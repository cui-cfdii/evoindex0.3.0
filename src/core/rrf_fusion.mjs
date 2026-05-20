/**
 * EvoIndex 3.0 — RRF 多路融合排序
 *
 * 融合树索引（结构定位）+ 向量检索（语义相似）+ 图增强（关系扩展）三路信号
 * 使用 Reciprocal Rank Fusion (RRF) + 线性加权两种模式
 *
 * @module rrf_fusion
 * @version 3.0.0
 */

/**
 * RRF 融合配置
 */
export class RRFConfig {
  constructor(options = {}) {
    /** RRF 参数 k (默认 60) */
    this.rrfK = options.rrfK || 60;

    /** 融合模式: 'rrf' | 'weighted' | 'hybrid' */
    this.mode = options.mode || 'weighted';

    /** 权重配置 */
    this.weights = {
      tree: options.treeWeight ?? 0.4,
      vector: options.vectorWeight ?? 0.6,
      graph: options.graphWeight ?? 0.0, // 图不参与 RRF 主融合
      ...options.weights,
    };

    /** 归一化权重 */
    this._normalized = false;
    this._normalize();
  }

  _normalize() {
    const w = this.weights;
    const total = w.tree + w.vector + w.graph;
    if (total > 0 && total !== 1) {
      w.tree /= total;
      w.vector /= total;
      w.graph /= total;
    }
    this._normalized = true;
  }
}

/**
 * RRF 融合引擎
 *
 * 输入三路排序结果，输出融合后的统一排序。
 *
 * 用法:
 *   const engine = new RRFFusion();
 *   const results = engine.fuse(treeResults, vectorResults, graphResults, topK);
 */
export class RRFFusion {
  constructor(config = new RRFConfig()) {
    this.config = config;
  }

  /**
   * 融合三路结果
   *
   * @param {Array<{id, score, ...}>} treeResults - 树索引结果
   * @param {Array<{id, score, ...}>} vectorResults - 向量检索结果
   * @param {Array<{id, score, ...}>} graphResults - 图增强结果 (可选)
   * @param {number} topK - 返回数量 (默认 10)
   * @returns {Array<{id, score, sources, ...}>} 融合后结果
   */
  fuse(treeResults = [], vectorResults = [], graphResults = [], topK = 10) {
    const mode = this.config.mode;

    if (mode === 'rrf') {
      return this._fuseRRF(treeResults, vectorResults, graphResults, topK);
    } else if (mode === 'weighted') {
      return this._fuseWeighted(treeResults, vectorResults, graphResults, topK);
    } else {
      // hybrid: RRF 初排 + weighted 精排
      const rrfResults = this._fuseRRF(treeResults, vectorResults, graphResults, topK * 2);
      return this._refineWeighted(rrfResults, topK);
    }
  }

  /**
   * RRF 融合
   *
   * 公式: RRF(d) = Σ 1/(k + rank_i(d))
   * 其中 rank_i(d) 是文档 d 在第 i 路中的排名
   */
  _fuseRRF(treeResults, vectorResults, graphResults, topK) {
    const k = this.config.rrfK;
    const scores = new Map();
    const sourceMap = new Map();
    const metaMap = new Map();

    treeResults.forEach((r, i) => {
      const rrf = 1 / (k + i + 1);
      scores.set(r.id, (scores.get(r.id) || 0) + rrf * this.config.weights.tree);
      this._trackSource(sourceMap, r.id, 'tree', rrf);
      if (!metaMap.has(r.id) && (r.metadata || r.node)) {
        const existing = metaMap.get(r.id) || {};
        Object.assign(existing, r.metadata || { title: r.node?.title || r.id });
        if (r.node) existing._node = r.node;
        metaMap.set(r.id, existing);
      }
    });

    vectorResults.forEach((r, i) => {
      const rrf = 1 / (k + i + 1);
      scores.set(r.id, (scores.get(r.id) || 0) + rrf * this.config.weights.vector);
      this._trackSource(sourceMap, r.id, 'vector', rrf);
      if (!metaMap.has(r.id) && r.metadata) metaMap.set(r.id, r.metadata);
    });

    graphResults.forEach((r, i) => {
      const rrf = 1 / (k + i + 1);
      scores.set(r.id, (scores.get(r.id) || 0) + rrf * this.config.weights.graph);
      this._trackSource(sourceMap, r.id, 'graph', rrf);
      if (!metaMap.has(r.id)) metaMap.set(r.id, r.metadata || { title: r.id });
    });

    const ranked = Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK);

    return ranked.map(([id, score]) => ({
      id,
      score: this._normalizeScore(score, treeResults.length + vectorResults.length),
      sourceScores: sourceMap.get(id),
      hits: this._countSources(sourceMap.get(id)),
      metadata: metaMap.get(id) || { title: id },
    }));
  }

  /**
   * 线性加权融合
   *
   * 差分隐私启发的归一化 + 加权组合
   */
  _fuseWeighted(treeResults, vectorResults, graphResults, topK) {
    const w = this.config.weights;
    const scores = new Map();
    const sourceMap = new Map();
    const metaMap = new Map(); // 保留 metadata

    // 归一到 [0, 1]
    const treeNorm = this._maxScore(treeResults);
    const vecNorm = this._maxScore(vectorResults);
    const graphNorm = this._maxScore(graphResults);

    treeResults.forEach(r => {
      const norm = treeNorm > 0 ? r.score / treeNorm : 0;
      scores.set(r.id, (scores.get(r.id) || 0) + norm * w.tree);
      this._trackSource(sourceMap, r.id, 'tree', norm);
      if (!metaMap.has(r.id) && (r.metadata || r.node)) {
        const existing = metaMap.get(r.id) || {};
        Object.assign(existing, r.metadata || { title: r.node?.title || r.id });
        if (r.node) existing._node = r.node;
        metaMap.set(r.id, existing);
      }
    });

    vectorResults.forEach(r => {
      const norm = vecNorm > 0 ? r.score / vecNorm : 0;
      scores.set(r.id, (scores.get(r.id) || 0) + norm * w.vector);
      this._trackSource(sourceMap, r.id, 'vector', norm);
      if (!metaMap.has(r.id) && r.metadata) {
        metaMap.set(r.id, r.metadata);
      }
    });

    graphResults.forEach(r => {
      const norm = graphNorm > 0 ? r.score / graphNorm : 0;
      scores.set(r.id, (scores.get(r.id) || 0) + norm * w.graph);
      this._trackSource(sourceMap, r.id, 'graph', norm);
      if (!metaMap.has(r.id)) {
        metaMap.set(r.id, r.metadata || { title: r.id, type: 'graph' });
      }
    });

    const ranked = Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK);

    return ranked.map(([id, score]) => ({
      id,
      score: Math.min(score, 1.0),
      sourceScores: sourceMap.get(id),
      hits: this._countSources(sourceMap.get(id)),
      metadata: metaMap.get(id) || { title: id },
    }));
  }

  /**
   * RRF + weighted 混合精排
   */
  _refineWeighted(rrfResults, topK) {
    // 在 RRF 粗排结果上应用加权精排
    const sorted = rrfResults
      .sort((a, b) => {
        // 偏好被多路命中的结果
        const aBoost = a.hits > 1 ? 1.1 : 1.0;
        const bBoost = b.hits > 1 ? 1.1 : 1.0;
        return (b.score * bBoost) - (a.score * aBoost);
      })
      .slice(0, topK);
    return sorted;
  }

  /**
   * 计算置信度
   *
   * @param {Array} fusedResults - 融合后的结果
   * @param {number} topK - 考虑的 Top-K
   * @returns {number} 置信度 [0, 1]
   */
  confidence(fusedResults, topK = 5) {
    if (!fusedResults || fusedResults.length === 0) return 0;

    const top = fusedResults.slice(0, topK);

    // 三个信号：
    // 1. Top-1 得分是否显著高于 Top-2 (gap)
    // 2. 多路命中比例
    // 3. 平均得分
    const topScore = top[0]?.score || 0;
    const secondScore = top[1]?.score || 0;
    const gap = topScore > 0 ? (topScore - secondScore) / topScore : 0;

    const multiHit = top.filter(r => r.hits > 1).length / top.length;
    const avgScore = top.reduce((s, r) => s + r.score, 0) / top.length;

    return (gap * 0.3 + multiHit * 0.4 + avgScore * 0.3);
  }

  // ─── 工具函数 ───────────────────────────────────

  _trackSource(map, id, source, score) {
    if (!map.has(id)) map.set(id, {});
    map.get(id)[source] = score;
  }

  _countSources(sourceInfo) {
    return sourceInfo ? Object.keys(sourceInfo).length : 1;
  }

  _maxScore(results) {
    if (!results || results.length === 0) return 0;
    return Math.max(...results.map(r => r.score || 0));
  }

  _normalizeScore(score, maxPossible) {
    // 简单的 min-max 风格归一化
    const theoretical = maxPossible > 0 ? 3 / (this.config.rrfK + 1) : 1;
    return Math.min(score / theoretical, 1.0);
  }
}

/** 默认实例 */
export const rrfFusion = new RRFFusion();
