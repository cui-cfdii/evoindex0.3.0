/**
 * EvoIndex 3.0 — 文档管理器 (CRUD)
 *
 * 知识库增删改查操作，维护树+向量+图三套索引的一致性。
 *
 * 操作路径:
 *   addDoc    → 解析MD→建树节点→嵌入→写LanceDB→更新图
 *   removeDoc → 定位节点→删LanceDB→删图实体→删树节点
 *   updateDoc → removeDoc + addDoc (简化)
 *   listDocs  → 遍历树根节点
 *
 * @module doc_manager
 * @version 3.0.0
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { HybridQueryEngineV3, QueryEngineConfig } from './hybrid_query.mjs';
import { VectorStore, VectorStoreConfig } from './lancedb_store.mjs';
import { GraphStore, GraphStoreConfig } from './graph_store.mjs';
import { EmbeddingClient } from '../utils/embedding_client.mjs';

/**
 * MD 解析：从 markdown 文本提取标题树
 */
function parseMarkdownTree(text, filename = 'untitled') {
  const lines = text.split('\n');
  const HEADING_RE = /^(#{1,6})\s+(.+)$/;

  // 收集标题位置
  const headings = [];
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(HEADING_RE);
    if (m) headings.push({ line: i, level: m[1].length, title: m[2].trim() });
  }

  if (headings.length === 0) {
    return {
      title: filename,
      level: 0,
      content: text.slice(0, 2000),
      children: [],
      _id: crypto.createHash('md5').update(filename).digest('hex').slice(0, 12),
    };
  }

  // 构建树
  const docChildren = [];
  const stack = [];

  for (let idx = 0; idx < headings.length; idx++) {
    const { line, level, title } = headings[idx];

    // 内容 = 当前标题后到下一个标题前
    const contentStart = line + 1;
    const contentEnd = idx + 1 < headings.length ? headings[idx + 1].line : lines.length;
    const content = lines.slice(contentStart, contentEnd).join('\n').trim().slice(0, 2000);

    const node = {
      title,
      level,
      content,
      children: [],
      _id: crypto.createHash('md5').update(`${filename}:${title}:${line}`).digest('hex').slice(0, 12),
    };

    // 找父节点
    while (stack.length > 0 && stack[stack.length - 1].level >= level) {
      stack.pop();
    }

    if (stack.length > 0) {
      node.level = stack[stack.length - 1].level + 1;
      stack[stack.length - 1].children.push(node);
    } else {
      node.level = 1;
      docChildren.push(node);
    }

    stack.push(node);
  }

  return {
    title: filename,
    level: 0,
    content: text.slice(0, 500),
    children: docChildren,
    _id: crypto.createHash('md5').update(filename).digest('hex').slice(0, 12),
  };
}

/**
 * 文档管理器配置
 */
export class DocManagerConfig {
  constructor(options = {}) {
    this.dataDir = options.dataDir || null;
  }
}

/**
 * 文档管理器
 */
export class DocManager {
  constructor(engine, config = new DocManagerConfig()) {
    this.engine = engine instanceof HybridQueryEngineV3
      ? engine
      : new HybridQueryEngineV3(engine); // 兼容传入 config
    this.config = config;
  }

  /**
   * 添加文档
   * @param {string} source - 文件路径或 markdown 内容
   * @param {object} options
   * @returns {Promise<{docId, nodeCount, embedCount}>}
   */
  async addDoc(source, options = {}) {
    if (!this.engine.index) throw new Error('请先 loadIndex()');

    let text, filename;

    // 判断是文件路径还是内容
    if (fs.existsSync(source)) {
      text = fs.readFileSync(source, 'utf-8');
      filename = path.basename(source, '.md').replace(/\.md$/, '');
    } else {
      text = source;
      filename = options.title || 'untitled';
    }

    // 1. 解析 markdown
    const docNode = parseMarkdownTree(text, filename);

    // 2. 插入到树索引
    this._insertDocNode(docNode);

    // 3. 提取所有节点并嵌入
    const allNodes = this._flatten(docNode);
    const embedClient = this.engine.config.embeddingClient;

    console.log(`📄 添加文档: ${filename} (${allNodes.length} 节点)`);

    let embedCount = 0;
    for (let i = 0; i < allNodes.length; i += 10) {
      const batch = allNodes.slice(i, i + 10);
      const texts = batch.map(n => `${n.title}\n${(n.content || '').slice(0, 500)}`);

      try {
        const vectors = await embedClient.embedBatch(texts);
        const items = batch.map((node, j) => ({
          id: node._id,
          vector: vectors[j],
          metadata: {
            title: node.title,
            path: `${filename} > ${node.title}`,
            level: node.level || 0,
          },
        }));
        await this.engine.config.vectorStore.addBatch(items);
        embedCount += items.length;
      } catch (err) {
        console.warn(`  ⚠️ 嵌入批次失败: ${err.message}`);
      }

      if (options.delay > 0) {
        await new Promise(r => setTimeout(r, options.delay));
      }
    }

    // 4. 增量更新图
    try {
      await this.engine.config.graphStore.buildFromTree(this.engine.index);
    } catch (err) {
      console.warn(`  ⚠️ 图更新跳过: ${err.message}`);
    }

    // 5. 更新统计
    if (this.engine.index.stats) {
      this.engine.index.stats.totalNodes = (this.engine.index.stats.totalNodes || 0) + allNodes.length;
      this.engine.index.stats.totalDocs = (this.engine.index.stats.totalDocs || 0) + 1;
    }

    return {
      docId: docNode._id,
      title: filename,
      nodeCount: allNodes.length,
      embedCount,
    };
  }

  /**
   * 删除文档
   * @param {string} query - 文档标题或 ID (模糊匹配)
   * @returns {Promise<{removed, nodeCount}>}
   */
  async removeDoc(query) {
    if (!this.engine.index) throw new Error('请先 loadIndex()');

    const root = this.engine.index.root || this.engine.index.tree?.root;
    if (!root) return { removed: 0, nodeCount: 0 };

    // 查找匹配的文档节点
    const matches = [];
    this._findDocs(root, query, matches);

    if (matches.length === 0) {
      console.log(`⚠️ 未找到匹配文档: "${query}"`);
      return { removed: 0, nodeCount: 0 };
    }

    let totalNodes = 0;
    for (const { parent, index, node } of matches) {
      const nodeCount = this._countNodes(node);
      totalNodes += nodeCount;

      // 从树中移除
      parent.children.splice(index, 1);

      // 从向量存储中删除 (批量标记)
      const allNodes = this._flatten(node);
      for (const n of allNodes) {
        // LanceDB 不支持直接删除，用标记
        try {
          await this.engine.config.vectorStore.add(n._id, new Array(768).fill(0), {
            title: '__DELETED__',
            path: '',
            level: -1,
          });
        } catch (_) {}
      }

      console.log(`🗑️ 删除文档: ${node.title} (${nodeCount} 节点)`);
    }

    // 重建图
    try {
      await this.engine.config.graphStore.buildFromTree(this.engine.index);
    } catch (_) {}

    // 更新统计
    if (this.engine.index.stats) {
      this.engine.index.stats.totalNodes = Math.max(0, (this.engine.index.stats.totalNodes || 0) - totalNodes);
      this.engine.index.stats.totalDocs = Math.max(0, (this.engine.index.stats.totalDocs || 0) - matches.length);
    }

    return { removed: matches.length, nodeCount: totalNodes };
  }

  /**
   * 更新文档
   * @param {string} query - 文档标题 (匹配)
   * @param {string} newContent - 新内容 (文件路径或 markdown)
   */
  async updateDoc(query, newContent) {
    // 先删后加
    const removed = await this.removeDoc(query);
    if (removed.removed === 0) {
      throw new Error(`未找到文档: "${query}"`);
    }

    let text;
    if (fs.existsSync(newContent)) {
      text = fs.readFileSync(newContent, 'utf-8');
    } else {
      text = newContent;
    }

    return await this.addDoc(text, { title: query });
  }

  /**
   * 列出文档
   * @returns {Array<{title, nodeCount}>}
   */
  listDocs() {
    if (!this.engine.index) return [];

    const root = this.engine.index.root || this.engine.index.tree?.root;
    if (!root || !root.children) return [];

    return root.children.map(doc => ({
      title: doc.title,
      id: doc._id,
      nodeCount: this._countNodes(doc),
      level: doc.level,
    }));
  }

  /**
   * 保存索引到文件
   */
  saveIndex(filepath) {
    if (!this.engine.index) throw new Error('无索引数据');
    fs.writeFileSync(filepath, JSON.stringify(this.engine.index, null, 2), 'utf-8');
    console.log(`💾 索引已保存: ${filepath}`);
  }

  // ─── 内部方法 ─────────────────

  _insertDocNode(docNode) {
    const root = this.engine.index.root || this.engine.index.tree?.root;
    if (!root) {
      this.engine.index.root = { title: 'Root', level: 0, children: [], _id: 'root' };
    }
    const targetRoot = this.engine.index.root || this.engine.index.tree.root;
    if (!targetRoot.children) targetRoot.children = [];
    targetRoot.children.push(docNode);
  }

  _findDocs(node, query, results, parent = null, index = -1) {
    if (!node) return;

    const qLower = query.toLowerCase();
    const titleLower = (node.title || '').toLowerCase();
    const idLower = (node._id || '').toLowerCase();

    if (titleLower.includes(qLower) || idLower === qLower) {
      if (parent && index >= 0) {
        results.push({ parent, index, node });
      }
    }

    if (node.children) {
      for (let i = 0; i < node.children.length; i++) {
        this._findDocs(node.children[i], query, results, node, i);
      }
    }
  }

  _flatten(node) {
    const nodes = [node];
    if (node.children) {
      for (const child of node.children) {
        nodes.push(...this._flatten(child));
      }
    }
    return nodes;
  }

  _countNodes(node) {
    let n = 1;
    if (node.children) {
      for (const child of node.children) n += this._countNodes(child);
    }
    return n;
  }
}
