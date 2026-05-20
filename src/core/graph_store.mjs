/**
 * EvoIndex 3.0 — 图存储模块
 *
 * 从树索引构建实体关系图，检测社区，支持子图查询。
 * 纯本地运行，不依赖 LLM（规则实体提取 + graphology 社区检测）。
 *
 * @module graph_store
 * @version 3.0.0
 */

import Graph from 'graphology';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * 图存储配置
 */
export class GraphStoreConfig {
  constructor(options = {}) {
    /** 存储目录 */
    this.dataDir = options.dataDir ||
      path.join(__dirname, '../../data/graph');
    /** 最小社区大小 */
    this.minCommunitySize = options.minCommunitySize || 5;
    /** 最大社区层级 */
    this.maxLevels = options.maxLevels || 3;
    /** 是否启用图 */
    this.enabled = options.enabled !== false;
  }
}

/**
 * 实体规则提取器
 * 从文本中提取中文/英文实体，不依赖 LLM
 */
class RuleEntityExtractor {
  constructor() {
    // 中文实体模式
    this.patterns = [
      // 技术概念: 大模型|RAG|GraphRAG|向量|嵌入|微调|LLM|GPT|BERT|Transformer...
      { type: '技术', re: /\b(RAG|GraphRAG|LLM|GPT|BERT|Transformer|LoRA|QLoRA|RLHF|DPO|MoE|KV.?Cache|Flash.?Attention|向量数据库|知识图谱|语义检索|混合检索|ANN|HNSW|IVF|PQ|LanceDB|Chroma|Pinecone|Weaviate|Milvus|Faiss|Qdrant|Ollama|LM.?Studio|vLLM|LangChain|LlamaIndex)\b/gi },
      // 模型名: Qwen|DeepSeek|GLM|ChatGLM|Baichuan|Yi|Mistral|Llama|Gemma|Claude|Phi|nomic|bge
      { type: '模型', re: /\b(Qwen\d*|DeepSeek|GLM\d*|ChatGLM\d*|Baichuan\d*|Yi-\d+|Mistral|Llama-?\d|Gemma|Claude|Phi-\d|nomic|bge-\w+|all-MiniLM|text2vec|m3e)\b/gi },
      // 框架/库: PyTorch|TensorFlow|HuggingFace|Transformers|LangChain|graphology|FastAPI|Spring|Django
      { type: '框架', re: /\b(PyTorch|TensorFlow|HuggingFace|Transformers|LangChain|graphology|FastAPI|Spring|Django|Next\.js|React|Vue|Flask)\b/gi },
      // 医药监管实体 (针对 pharma_regulatory 文档集)
      { type: '药品', re: /\b(阿司匹林|二甲双胍|胰岛素|利拉鲁肽|达格列净|恩格列净|PD-1|CAR-T|ADC|mRNA|siRNA|ASO|单克隆抗体|双特异性抗体)\b/gi },
      { type: '疾病', re: /\b(糖尿病|高血压|冠心病|脑卒中|阿尔茨海默|帕金森|肿瘤|白血病|淋巴瘤|COVID-19|肺动脉高压)\b/gi },
      { type: '靶点', re: /\b(GLP-1|SGLT2|DPP-4|EGFR|HER2|PD-L1|CTLA-4|VEGF|TNF-α|JAK|BTK)\b/gi },
      { type: '监管', re: /\b(FDA|EMA|NMPA|CDE|ICH|GCP|GLP|GMP|NDA|IND|ANDA|BLA|孤儿药|突破性疗法|加速审批)\b/gi },
      // 中文概念
      { type: '概念', re: /(自然语言处理|机器(学习|翻译|阅读)|深度(学习|神经网络)|强化学习|迁移学习|联邦学习|知识蒸馏|模型压缩|量化|剪枝|推理优化|语义理解|文本生成|代码生成|多模态|视觉语言)/g },
      // 英文概念（大小写混合）
      { type: '概念_en', re: /\b(Retrieval.?Augmented.?Generation|Semantic.?Search|Vector.?Database|Knowledge.?Graph|Hybrid.?Retrieval|Self.?Attention|Cross.?Attention|Multi.?Head.?Attention|Chain.?of.?Thought|Tree.?of.?Thought|Retrieval.?QA|Embedding.?Model|Fine.?Tuning|Pre.?Training|In.?Context.?Learning)\b/gi },
    ];
  }

  extract(text) {
    const entities = new Map();
    const relationships = [];

    for (const { type, re } of this.patterns) {
      const matches = text.matchAll(re);
      for (const match of matches) {
        const name = match[0];
        if (!entities.has(name)) {
          entities.set(name, {
            name,
            type: this._normalizeType(type),
            count: 0,
          });
        }
        entities.get(name).count++;
      }
    }

    // 生成简单关系：同类型实体间建立关联
    const typeGroups = new Map();
    for (const [name, entity] of entities) {
      if (!typeGroups.has(entity.type)) {
        typeGroups.set(entity.type, []);
      }
      typeGroups.get(entity.type).push(name);
    }

    for (const [type, names] of typeGroups) {
      for (let i = 0; i < names.length; i++) {
        for (let j = i + 1; j < names.length; j++) {
          relationships.push({
            source: names[i],
            target: names[j],
            type: '相关',
            weight: 0.5,
          });
        }
      }
    }

    return {
      entities: [...entities.values()],
      relationships,
    };
  }

  _normalizeType(type) {
    if (type === '概念_en') return '概念';
    return type;
  }
}

/**
 * 图存储
 * 管理知识图谱的构建、存储和查询
 */
export class GraphStore {
  constructor(config = new GraphStoreConfig()) {
    this.config = config;
    this.graph = new Graph();
    this.communities = [];
    this._nodeToEntity = new Map();  // 树节点ID → 实体列表
    this._entityToNodes = new Map(); // 实体名 → 树节点ID列表
    this._initialized = false;
    this._dirty = false;
  }

  /**
   * 从树索引构建图
   * @param {object} treeIndex - EvoIndex 树索引对象
   * @param {object} options
   */
  async buildFromTree(treeIndex, options = {}) {
    if (!this.config.enabled) {
      console.log('⚠️ 图存储已禁用');
      return { nodes: 0, edges: 0, communities: 0 };
    }

    console.log('🔗 从树索引构建知识图谱...');

    const root = treeIndex.root || treeIndex.tree?.root;
    if (!root) {
      console.warn('⚠️ 树索引为空，跳图构建');
      return { nodes: 0, edges: 0, communities: 0 };
    }

    // 1. 规则实体提取
    const extractor = new RuleEntityExtractor();
    const allNodes = this._flattenTree(root);
    console.log(`  扫描 ${allNodes.length} 个树节点...`);

    const allEntities = new Map();  // 全局去重实体
    const allRelationships = [];

    let entityCount = 0;
    for (const node of allNodes) {
      const text = `${node.title || ''} ${(node.content || '').slice(0, 1000)}`;
      const result = extractor.extract(text);

      if (result.entities.length === 0) continue;

      // 记录节点-实体关联
      const nodeEntities = [];
      for (const entity of result.entities) {
        if (!allEntities.has(entity.name)) {
          allEntities.set(entity.name, { ...entity, nodes: [] });
          entityCount++;
        }
        allEntities.get(entity.name).nodes.push(node._id || node.title);
        nodeEntities.push(entity.name);
      }
      this._nodeToEntity.set(node._id || node.title, nodeEntities);

      allRelationships.push(...result.relationships);
    }

    console.log(`  提取 ${entityCount} 个实体, ${allRelationships.length} 条关系`);

    // 2. 构建图
    this.graph = new Graph();

    // 添加实体节点
    for (const [name, entity] of allEntities) {
      this.graph.addNode(name, {
        label: name,
        type: entity.type,
        count: entity.count,
        docNodes: entity.nodes,
        size: Math.min(10, 2 + entity.nodes.length),
      });
      for (const nodeId of entity.nodes) {
        if (!this._entityToNodes.has(name)) {
          this._entityToNodes.set(name, []);
        }
        this._entityToNodes.get(name).push(nodeId);
      }
    }

    // 添加关系边
    let edgeCount = 0;
    const addedEdges = new Set();
    for (const rel of allRelationships) {
      if (!this.graph.hasNode(rel.source) || !this.graph.hasNode(rel.target)) continue;

      const edgeKey = [rel.source, rel.target].sort().join('||');
      if (addedEdges.has(edgeKey)) continue;
      addedEdges.add(edgeKey);

      this.graph.addEdge(rel.source, rel.target, {
        type: rel.type,
        weight: rel.weight || 1,
      });
      edgeCount++;
    }

    console.log(`  图: ${this.graph.order} 节点, ${edgeCount} 边`);

    // 3. 社区检测
    this.communities = this._detectCommunities();
    console.log(`  社区: ${this.communities.length} 个`);

    this._initialized = true;
    this._dirty = false;

    return {
      nodes: this.graph.order,
      edges: edgeCount,
      communities: this.communities.length,
    };
  }

  /**
   * 图增强查询 — 从查询关键词扩展相关实体
   * @param {string} query - 用户查询
   * @param {number} topK - 返回相关实体数
   * @returns {Array<{id, score, metadata}>}
   */
  enhance(query, topK = 20) {
    if (!this._initialized || !this.config.enabled) return [];

    // 1. 在图中匹配查询中的实体
    const matchedEntities = [];
    for (const node of this.graph.nodes()) {
      const queryLower = query.toLowerCase();
      const nodeLower = node.toLowerCase();

      // 模糊匹配
      if (queryLower.includes(nodeLower) || nodeLower.includes(queryLower)) {
        matchedEntities.push(node);
        continue;
      }

      // 字符级模糊匹配（中文）
      const queryChars = [...query];
      const nodeChars = [...node];
      let matchCount = 0;
      for (const qc of queryChars) {
        if (nodeChars.includes(qc)) matchCount++;
      }
      if (matchCount >= Math.min(3, queryChars.length)) {
        matchedEntities.push(node);
      }
    }

    // 2. 子图展开：从匹配实体出发，沿边扩展1-2跳
    const expandedNodes = new Set(matchedEntities);
    for (const entity of matchedEntities) {
      // 1-hop neighbors
      for (const neighbor of this.graph.neighbors(entity)) {
        expandedNodes.add(neighbor);
        // 2-hop
        for (const n2 of this.graph.neighbors(neighbor)) {
          if (expandedNodes.size >= topK * 3) break;
          expandedNodes.add(n2);
        }
        if (expandedNodes.size >= topK * 3) break;
      }
    }

    // 3. 收集关联的树节点
    const relatedNodeIds = new Set();
    for (const entity of expandedNodes) {
      const nodeIds = this._entityToNodes.get(entity) || [];
      for (const nid of nodeIds) {
        relatedNodeIds.add(nid);
      }
    }

    // 4. 计算得分（基于图结构）
    const results = [];
    for (const nodeId of relatedNodeIds) {
      // 得分 = 关联实体数 / 总匹配实体数
      const entities = this._nodeToEntity.get(nodeId) || [];
      const matchRatio = entities.filter(e => expandedNodes.has(e)).length / Math.max(1, expandedNodes.size);
      results.push({
        id: nodeId,
        score: Math.min(1.0, matchRatio * 3), // 放大但封顶
        metadata: {
          type: 'graph',
          entities: entities.slice(0, 5),
        },
      });
    }

    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * 增量更新 — 单个节点变更时更新图
   * @param {object} node - 变更的树节点
   * @param {'add'|'update'|'remove'} action
   */
  async incrementalUpdate(node, action = 'update') {
    if (!this._initialized) return;

    const nodeId = node._id || node.title;
    const text = `${node.title || ''} ${(node.content || '').slice(0, 1000)}`;

    // 1. 移除旧关联
    const oldEntities = this._nodeToEntity.get(nodeId) || [];
    for (const entityName of oldEntities) {
      const nodes = this._entityToNodes.get(entityName) || [];
      this._entityToNodes.set(entityName, nodes.filter(id => id !== nodeId));
      // 若无其他节点引用，移除实体
      if (this._entityToNodes.get(entityName)?.length === 0) {
        this._entityToNodes.delete(entityName);
        try { this.graph.dropNode(entityName); } catch (_) {}
      }
    }

    if (action === 'remove') {
      this._nodeToEntity.delete(nodeId);
      this._dirty = true;
      return;
    }

    // 2. 提取新实体
    const extractor = new RuleEntityExtractor();
    const { entities } = extractor.extract(text);
    const newEntityNames = entities.map(e => e.name);

    this._nodeToEntity.set(nodeId, newEntityNames);

    // 3. 更新图
    for (const entity of entities) {
      if (!this.graph.hasNode(entity.name)) {
        this.graph.addNode(entity.name, {
          label: entity.name,
          type: entity.type,
          count: entity.count,
          size: 2,
        });
      }

      if (!this._entityToNodes.has(entity.name)) {
        this._entityToNodes.set(entity.name, []);
      }
      if (!this._entityToNodes.get(entity.name).includes(nodeId)) {
        this._entityToNodes.get(entity.name).push(nodeId);
      }
    }

    this._dirty = true;
  }

  /**
   * 保存图到磁盘
   */
  async save() {
    if (!fs.existsSync(this.config.dataDir)) {
      fs.mkdirSync(this.config.dataDir, { recursive: true });
    }

    const data = {
      communities: this.communities,
      nodeToEntity: [...this._nodeToEntity],
      entityToNodes: [...this._entityToNodes],
      graphData: this.graph.export(),
    };

    fs.writeFileSync(
      path.join(this.config.dataDir, 'graph_backup.json'),
      JSON.stringify(data, null, 2),
      'utf-8'
    );

    this._dirty = false;
    console.log('💾 图已保存');
  }

  /**
   * 从磁盘加载图
   */
  async load() {
    const backupFile = path.join(this.config.dataDir, 'graph_backup.json');
    if (!fs.existsSync(backupFile)) return false;

    try {
      const data = JSON.parse(fs.readFileSync(backupFile, 'utf-8'));
      this.communities = data.communities || [];
      this._nodeToEntity = new Map(data.nodeToEntity || []);
      this._entityToNodes = new Map(data.entityToNodes || []);

      this.graph = new Graph();
      if (data.graphData) {
        this.graph.import(data.graphData);
      }

      this._initialized = true;
      console.log(`📂 图已加载: ${this.graph.order} 节点, ${this.communities.length} 社区`);
      return true;
    } catch (err) {
      console.warn(`⚠️ 图加载失败: ${err.message}`);
      return false;
    }
  }

  getStats() {
    return {
      enabled: this.config.enabled,
      initialized: this._initialized,
      nodes: this._initialized ? this.graph.order : 0,
      edges: this._initialized ? this.graph.size : 0,
      communities: this.communities.length,
      dirty: this._dirty,
    };
  }

  // ─── 内部方法 ───────────────────────────────

  _flattenTree(node) {
    const nodes = [node];
    if (node.children) {
      for (const child of node.children) {
        nodes.push(...this._flattenTree(child));
      }
    }
    return nodes;
  }

  /**
   * 层级社区检测 (Louvain-like)
   */
  _detectCommunities() {
    const communities = [];

    // 方案：按实体类型分组作为初始社区
    const typeGroups = new Map();
    for (const node of this.graph.nodes()) {
      const attrs = this.graph.getNodeAttributes(node);
      const type = attrs.type || 'other';
      if (!typeGroups.has(type)) {
        typeGroups.set(type, []);
      }
      typeGroups.get(type).push(node);
    }

    for (const [type, nodes] of typeGroups) {
      if (nodes.length >= this.config.minCommunitySize) {
        communities.push({
          id: `community_${type}`,
          type,
          entities: nodes,
          size: nodes.length,
          summary: `${type}相关实体群: ${nodes.slice(0, 10).join(', ')}${nodes.length > 10 ? '...' : ''}`,
        });
      }
    }

    // 对大类做子社区拆分（按连通分量）
    const finalCommunities = [];
    for (const comm of communities) {
      if (comm.size <= this.config.minCommunitySize * 3) {
        finalCommunities.push(comm);
        continue;
      }

      // 大社区：按连通分量拆
      const subgraphs = this._connectedComponents(comm.entities);
      for (const [i, subNodes] of subgraphs.entries()) {
        finalCommunities.push({
          id: `community_${comm.type}_${i}`,
          type: comm.type,
          entities: subNodes,
          size: subNodes.length,
          summary: `${comm.type}子群${i + 1}: ${subNodes.slice(0, 8).join(', ')}`,
        });
      }
    }

    return finalCommunities;
  }

  _connectedComponents(nodes) {
    const visited = new Set();
    const components = [];
    const nodeSet = new Set(nodes);

    for (const start of nodes) {
      if (visited.has(start)) continue;

      const component = [];
      const queue = [start];
      while (queue.length > 0) {
        const current = queue.shift();
        if (visited.has(current)) continue;
        if (!nodeSet.has(current)) continue;
        visited.add(current);
        component.push(current);

        for (const neighbor of this.graph.neighbors(current)) {
          if (!visited.has(neighbor) && nodeSet.has(neighbor)) {
            queue.push(neighbor);
          }
        }
      }

      if (component.length > 0) {
        components.push(component);
      }
    }

    return components;
  }
}

/** 默认单例 */
export const graphStore = new GraphStore();
