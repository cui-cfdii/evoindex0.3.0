/**
 * EvoIndex 3.0 — 查询记忆缓存 (P2 "LLMWiki")
 *
 * 缓存查询→结果的映射，通过嵌入相似度实现模糊匹配。
 * 重复/相似查询 <5ms 直达，跳过树+向量+图全链路。
 *
 * 两层缓存:
 *   L1: 精确匹配 (query string hash) → <1ms
 *   L2: 嵌入相似匹配 (cosine > 0.95) → <5ms
 *
 * @module query_cache
 * @version 3.0.0
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { embeddingClient } from '../utils/embedding_client.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * 查询缓存配置
 */
export class QueryCacheConfig {
  constructor(options = {}) {
    /** 最大缓存条目 */
    this.maxSize = options.maxSize || 1000;
    /** L2 相似度阈值 */
    this.similarityThreshold = options.similarityThreshold || 0.92;
    /** 是否启用 */
    this.enabled = options.enabled !== false;
    /** 持久化路径 */
    this.cachePath = options.cachePath ||
      path.join(__dirname, '../../data/cache/query_cache.json');
    /** TTL (毫秒, 0=永久) */
    this.ttl = options.ttl || 0;
  }
}

/**
 * 查询记忆缓存
 */
export class QueryCache {
  constructor(config = new QueryCacheConfig()) {
    this.config = config;

    /** L1: hash → result */
    this._exact = new Map();

    /** L2: 嵌入向量列表 + 结果 */
    this._embeddings = [];    // { query, vector, result, timestamp }

    /** 访问顺序 (用于 LRU 淘汰) */
    this._accessOrder = [];

    this._stats = { hits: 0, misses: 0, l1Hits: 0, l2Hits: 0 };
  }

  /**
   * 查询缓存
   * @param {string} query - 用户查询
   * @returns {object|null} 缓存的结果 或 null
   */
  async get(query) {
    if (!this.config.enabled) return null;

    const key = this._hash(query);

    // L1: 精确匹配
    if (this._exact.has(key)) {
      const entry = this._exact.get(key);
      if (!this._isExpired(entry)) {
        this._stats.hits++;
        this._stats.l1Hits++;
        this._touch(key);
        return entry.result;
      }
      this._exact.delete(key);
    }

    // L2: 嵌入相似匹配
    if (this._embeddings.length > 0) {
      try {
        const queryVec = await embeddingClient.embed(query);
        let bestScore = 0;
        let bestEntry = null;
        let bestIdx = -1;

        for (let i = 0; i < this._embeddings.length; i++) {
          const entry = this._embeddings[i];
          const sim = this._cosine(queryVec, entry.vector);
          if (sim > bestScore && sim >= this.config.similarityThreshold) {
            bestScore = sim;
            bestEntry = entry;
            bestIdx = i;
          }
        }

        if (bestEntry && !this._isExpired(bestEntry)) {
          this._stats.hits++;
          this._stats.l2Hits++;
          this._touch(this._hash(bestEntry.query));
          return bestEntry.result;
        }
      } catch (err) {
        // 嵌入失败，跳过 L2
      }
    }

    this._stats.misses++;
    return null;
  }

  /**
   * 存入缓存
   * @param {string} query - 查询
   * @param {object} result - 查询结果
   * @param {number[]} [vector] - 预计算的嵌入向量 (可选)
   */
  async set(query, result, vector = null) {
    if (!this.config.enabled) return;

    const key = this._hash(query);

    // L1
    this._exact.set(key, {
      result,
      timestamp: Date.now(),
    });
    this._touch(key);

    // L2 (异步嵌入)
    if (!vector) {
      try {
        vector = await embeddingClient.embed(query);
      } catch {
        vector = null;
      }
    }

    if (vector) {
      this._embeddings.push({
        query,
        vector,
        result,
        timestamp: Date.now(),
      });

      // 限制 L2 大小
      if (this._embeddings.length > this.config.maxSize / 2) {
        this._embeddings.shift();
      }
    }

    // LRU 淘汰
    while (this._exact.size > this.config.maxSize) {
      const oldest = this._accessOrder.shift();
      if (oldest) this._exact.delete(oldest);
    }
  }

  /**
   * 获取统计
   */
  getStats() {
    const total = this._stats.hits + this._stats.misses;
    return {
      total,
      hits: this._stats.hits,
      misses: this._stats.misses,
      hitRate: total > 0 ? (this._stats.hits / total * 100).toFixed(1) + '%' : 'N/A',
      l1Hits: this._stats.l1Hits,
      l2Hits: this._stats.l2Hits,
      l1Size: this._exact.size,
      l2Size: this._embeddings.length,
      enabled: this.config.enabled,
    };
  }

  /**
   * 持久化到磁盘
   */
  async save() {
    const dir = path.dirname(this.config.cachePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const data = {
      exact: [...this._exact].map(([k, v]) => [k, { result: v.result, timestamp: v.timestamp }]),
      embeddings: this._embeddings.map(e => ({
        query: e.query,
        vector: e.vector,
        timestamp: e.timestamp,
      })),
      stats: this._stats,
    };

    fs.writeFileSync(this.config.cachePath, JSON.stringify(data), 'utf-8');
  }

  /**
   * 从磁盘加载
   */
  async load() {
    if (!fs.existsSync(this.config.cachePath)) return false;

    try {
      const data = JSON.parse(fs.readFileSync(this.config.cachePath, 'utf-8'));

      if (data.exact) {
        for (const [k, v] of data.exact) {
          this._exact.set(k, v);
          this._accessOrder.push(k);
        }
      }

      if (data.embeddings) {
        this._embeddings = data.embeddings;
      }

      if (data.stats) {
        this._stats = data.stats;
      }

      console.log(`📂 缓存已加载: ${this._exact.size} 条精确, ${this._embeddings.length} 条语义`);
      return true;
    } catch (err) {
      console.warn(`⚠️ 缓存加载失败: ${err.message}`);
      return false;
    }
  }

  /**
   * 清空缓存
   */
  clear() {
    this._exact.clear();
    this._embeddings = [];
    this._accessOrder = [];
    this._stats = { hits: 0, misses: 0, l1Hits: 0, l2Hits: 0 };
  }

  // ─── 内部 ─────────────────

  _hash(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
      h = ((h << 5) - h) + str.charCodeAt(i);
      h |= 0; // 32-bit
    }
    return 'q_' + Math.abs(h);
  }

  _touch(key) {
    const idx = this._accessOrder.indexOf(key);
    if (idx >= 0) this._accessOrder.splice(idx, 1);
    this._accessOrder.push(key);
  }

  _isExpired(entry) {
    if (this.config.ttl <= 0) return false;
    return Date.now() - entry.timestamp > this.config.ttl;
  }

  _cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }
}

/** 默认单例 */
export const queryCache = new QueryCache();
