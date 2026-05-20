/**
 * EvoIndex 3.0 — 混合查询引擎
 *
 * 集成 树索引(结构定位) + 向量检索(语义相似) + 图增强(关系推理) + RRF 三路融合
 *
 * @module hybrid_query
 * @version 3.0.0
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { EmbeddingClient, embeddingClient } from '../utils/embedding_client.mjs';
import { VectorStore, vectorStore } from './lancedb_store.mjs';
import { RRFFusion, rrfFusion } from './rrf_fusion.mjs';
import { GraphStore, graphStore } from './graph_store.mjs';
import { QueryCache, queryCache } from './query_cache.mjs';
import { routeQuery, nodeMatchesDomain } from './query_router.mjs';
import {
  titleMatchScore,
  entityMatchScore,
  communityRelevanceScore,
  contentMatchScore,
  hybridScore as _legacyHybridScore,
} from '../utils/scoring.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * 混合查询引擎配置
 */
export class QueryEngineConfig {
  constructor(options = {}) {
    /** RRF 融合模式 */
    this.fusionMode = options.fusionMode || 'weighted';

    /** RRF 权重 — 树:结构 向量:语义 图:关系 */
    this.fusionWeights = options.fusionWeights || { tree: 0.45, vector: 0.45, graph: 0.1 };

    /** 向量检索 Top-K (在 RRF 融合前) */
    this.vectorTopK = options.vectorTopK || 50;

    /** 树检索 Top-K */
    this.treeTopK = options.treeTopK || 50;

    /** 图增强 Top-K */
    this.graphTopK = options.graphTopK || 20;

    /** 最终返回 Top-K */
    this.finalTopK = options.finalTopK || 10;

    /** 置信度阈值 — 高于此值直接返回，不触发 LLM */
    this.confidenceThreshold = options.confidenceThreshold || 0.7;

    /** 向量存储实例 */
    this.vectorStore = options.vectorStore || vectorStore;

    /** 嵌入客户端实例 */
    this.embeddingClient = options.embeddingClient || embeddingClient;

    /** RRF 融合实例 */
    this.rrfFusion = options.rrfFusion || rrfFusion;

    /** 图存储实例 (P1) */
    this.graphStore = options.graphStore || graphStore;

    /** 查询缓存实例 (P2) */
    this.queryCache = options.queryCache || queryCache;
  }
}

/**
 * 混合查询引擎
 */
export class HybridQueryEngineV3 {
  constructor(config = new QueryEngineConfig()) {
    this.config = config;
    this.index = null;
    this.communitySummaries = [];
    this._initialized = false;
  }

  /**
   * 加载索引文件
   * @param {string} indexPath - 索引 JSON 文件路径
   */
  async loadIndex(indexPath) {
    if (!fs.existsSync(indexPath)) {
      throw new Error(`索引文件不存在: ${indexPath}`);
    }

    const content = fs.readFileSync(indexPath, 'utf-8');
    this.index = JSON.parse(content);

    // 加载社区摘要
    if (this.index.communities) {
      this.communitySummaries = this._flattenCommunities(this.index.communities);
    }

    // 初始化向量存储
    if (!this._initialized) {
      await this.config.vectorStore.init();
      this._initialized = true;
    }

    console.log(`✅ 索引已加载: ${indexPath}`);
    console.log(`   节点: ${this.index.stats?.totalNodes || '?'} | 实体: ${this.index.stats?.totalEntities || '?'}`);
    console.log(`   向量存储模式: ${this.config.vectorStore.getStats().mode}`);

    return this;
  }

  /**
   * 构建向量索引
   * 递归遍历树索引的所有节点，为每个节点生成嵌入并存储
   */
  async buildVectorIndex(options = {}) {
    if (!this.index) throw new Error('请先 loadIndex()');

    const nodes = this._flattenTree(this.index.root || this.index.tree?.root);
    console.log(`🔨 构建向量索引: ${nodes.length} 个节点...`);

    let count = 0;
    const batchSize = options.batchSize || 10;

    for (let i = 0; i < nodes.length; i += batchSize) {
      const batch = nodes.slice(i, i + batchSize);
      const texts = batch.map(n => `${n.title}\n${(n.content || '').slice(0, 2000)}`);

      try {
        const vectors = await this.config.embeddingClient.embedBatch(texts);
        const items = batch.map((node, j) => ({
          id: node._id || `node-${i + j}`,
          vector: vectors[j],
          metadata: {
            title: node.title,
            path: this._getNodePath(node),
            level: node.level || 0,
          },
        }));

        await this.config.vectorStore.addBatch(items);
        count += items.length;
        console.log(`  进度: ${count}/${nodes.length}`);
      } catch (err) {
        console.error(`  批次 ${i}-${i + batchSize} 失败: ${err.message}`);
      }

      // 避免请求过快
      if (options.delay > 0) {
        await new Promise(r => setTimeout(r, options.delay));
      }
    }

    console.log(`✅ 向量索引构建完成: ${count} 条`);
  }

  /**
   * 构建图索引 (P1)
   * 从树索引导出实体关系图，检测社区
   */
  async buildGraphIndex(options = {}) {
    if (!this.index) throw new Error('请先 loadIndex()');
    return await this.config.graphStore.buildFromTree(this.index, options);
  }

  /**
   * 查询
   * @param {string} query - 用户查询
   * @param {object} options - 查询选项
   * @returns {Promise<object>} { results, confidence, sources }
   */
  async query(query, options = {}) {
    if (!this.index) throw new Error('请先 loadIndex()');

    const topK = options.topK || this.config.finalTopK;

    // P2: 检查查询缓存
    if (!options.skipCache) {
      const cached = await this.config.queryCache.get(query);
      if (cached) {
        console.log(`⚡ 缓存命中: "${query}" → ${cached.results?.length || '?'} 结果`);
        return { ...cached, fromCache: true };
      }
    }

    console.log(`🔍 查询: "${query}"`);

    // Layer 0: 查询意图路由
    const domain = options.domain || routeQuery(query);
    if (domain !== 'all') {
      console.log(`  🧭 路由: ${domain}`);
    }

    // Step 1: 树索引召回
    const treeTopK = options.treeTopK || this.config.treeTopK;
    const treeResults = this._treeSearch(query, treeTopK, domain);

    // Step 2: 向量语义召回
    const vectorResults = await this._vectorSearch(query, this.config.vectorTopK);

    // Step 3: 图增强召回 (P1)
    const graphResults = this.config.graphStore.getStats().initialized
      ? this.config.graphStore.enhance(query, this.config.graphTopK)
      : [];

    // Step 4: RRF 三路融合
    const fusionConfig = {
      mode: this.config.fusionMode,
      weights: this.config.fusionWeights,
    };
    const rrf = new RRFFusion(fusionConfig);
    const fusedResults = rrf.fuse(treeResults, vectorResults, graphResults, topK);

    // Step 5: 计算置信度
    const confidence = rrf.confidence(fusedResults, Math.min(5, topK));

    // Step 6: 判断是否需要 LLM 精排
    const needsLLM = confidence < this.config.confidenceThreshold;

    console.log(`✅ 检索完成: ${fusedResults.length} 结果 | 置信度: ${(confidence * 100).toFixed(1)}% | LLM: ${needsLLM ? '需要' : '跳过'}`);

    const result = {
      results: fusedResults,
      confidence,
      needsLLM,
      sourceCounts: {
        tree: treeResults.length,
        vector: vectorResults.length,
        graph: graphResults.length,
        fused: fusedResults.length,
      },
      query,
    };

    // P2: 存入缓存
    await this.config.queryCache.set(query, result);

    return result;
  }

  /**
   * 树索引搜索（复用 2.0 逻辑）
   */
  _treeSearch(query, topK, domain = 'all') {
    const results = [];
    const root = this.index.root || this.index.tree?.root;
    if (!root) return results;

    this._searchTreeRecursive(root, query, results, domain);

    // 按混合评分排序
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  _searchTreeRecursive(node, query, results, domain = 'all') {
    if (!node) return;

    // 领域过滤 (Layer 0 路由)
    if (domain !== 'all' && !nodeMatchesDomain(node, domain)) {
      // 仍然搜索子节点（文档内可能有跨领域内容）
      if (node.children) {
        for (const child of node.children) {
          this._searchTreeRecursive(child, query, results, domain);
        }
      }
      return;
    }

    // 混合评分（标题+实体+社区）
    let score = _legacyHybridScore(
      { ...node, communityId: this._findCommunity(node) },
      query,
      { communitySummaries: this.communitySummaries }
    );

    // 降级：关键词不匹配时，用内容匹配补充
    if (score === 0 && node.content) {
      score = contentMatchScore(node.content, query) * 0.5;
    }

    // 再降级：中文字符级模糊匹配
    if (score === 0 && node.title) {
      const queryChars = [...query];
      const titleChars = [...node.title];
      let matchCount = 0;
      for (const qc of queryChars) {
        if (titleChars.includes(qc)) matchCount++;
      }
      if (matchCount >= Math.min(2, queryChars.length)) {
        score = (matchCount / queryChars.length) * 0.3;
      }
    }

    if (score > 0.02) {  // 最低阈值, 过滤纯噪音
      results.push({ id: node._id || node.title, score, node });
    }
    if (node.children) {
      for (const child of node.children) {
        this._searchTreeRecursive(child, query, results);
      }
    }
  }

  /**
   * 向量语义搜索
   */
  async _vectorSearch(query, topK) {
    try {
      const queryVec = await this.config.embeddingClient.embed(query);
      return await this.config.vectorStore.search(queryVec, topK);
    } catch (err) {
      console.warn(`⚠️  向量搜索失败: ${err.message}，返回空结果`);
      return [];
    }
  }

  // ─── 工具函数 ───────────────────────────────────

  _flattenTree(node) {
    const nodes = [node];
    if (node.children) {
      for (const child of node.children) {
        nodes.push(...this._flattenTree(child));
      }
    }
    return nodes;
  }

  _flattenCommunities(communities, level = 0) {
    const flat = [];
    for (const c of communities) {
      flat.push({ ...c, level });
      if (c.children) {
        flat.push(...this._flattenCommunities(c.children, level + 1));
      }
    }
    return flat;
  }

  _findCommunity(node) {
    if (!this.communitySummaries.length) return null;
    const text = `${node.title} ${node.content || ''}`.toLowerCase();
    let best = null, bestScore = 0;
    for (const c of this.communitySummaries) {
      let score = 0;
      if (c.summary) {
        for (const w of c.summary.toLowerCase().split(/\s+/)) {
          if (text.includes(w)) score += 0.1;
        }
      }
      if (c.entities) {
        for (const e of c.entities) {
          if (text.includes(e.name?.toLowerCase())) score += 0.3;
        }
      }
      if (score > bestScore) { bestScore = score; best = c.id; }
    }
    return best;
  }

  _getNodePath(node) {
    // 简单路径：用层级和标题构造
    const level = node.level || 0;
    const prefix = '#'.repeat(Math.min(level + 1, 6));
    return `${prefix} ${node.title}`;
  }

  _displayResults(results, options = {}) {
    console.log(`\n📊 排序结果 (Top ${results.length}):\n`);
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      console.log(`${i + 1}. [${(r.score * 100).toFixed(1)}%] ${r.metadata?.title || r.id}`);
      if (options.showSources && r.sourceScores) {
        const sources = Object.entries(r.sourceScores)
          .map(([k, v]) => `${k}:${(v * 100).toFixed(0)}%`)
          .join(' ');
        console.log(`   来源: ${sources} | 命中: ${r.hits}路`);
      }
    }
  }
}

/** 默认实例 */
export const hybridQueryEngine = new HybridQueryEngineV3();
