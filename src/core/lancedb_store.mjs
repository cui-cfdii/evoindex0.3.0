/**
 * EvoIndex 3.0 — LanceDB 向量存储
 *
 * 管理文档嵌入向量的持久化存储和 ANN 检索。
 * 主路径：vectordb (LanceDB Node.js SDK)
 * 降级路径：内存向量 + 余弦相似度 + JSON 文件持久化
 *
 * @module lancedb_store
 * @version 3.0.0
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * 向量存储配置
 */
export class VectorStoreConfig {
  constructor(options = {}) {
    /** 存储目录 */
    this.dataDir = options.dataDir ||
      path.join(__dirname, '../../data/vectors');

    /** 向量维度 (nomic-embed: 768) */
    this.dimension = options.dimension || 768;

    /** 集合/表名 */
    this.collectionName = options.collectionName || 'evoindex_docs';

    /** 降级模式：'lancedb' | 'memory' | 'auto' */
    this.mode = options.mode || 'auto';

    /** 缓存最近检索结果数 */
    this.cacheSize = options.cacheSize || 100;
  }
}

/**
 * 向量存储
 *
 * 用法:
 *   const store = new VectorStore();
 *   await store.init();
 *   await store.add('doc-1', [0.1, 0.2, ...], { title: '...' });
 *   const results = await store.search(queryVec, 10);
 */
export class VectorStore {
  constructor(config = new VectorStoreConfig()) {
    this.config = config;
    this._db = null;          // LanceDB 连接
    this._table = null;       // LanceDB 表
    this._memoryVectors = []; // 降级模式：内存向量
    this._memoryMeta = new Map();
    this._mode = null;
    this._cache = new Map();  // 简单 LRU 缓存
  }

  /**
   * 初始化向量存储
   */
  async init() {
    // 确保数据目录存在
    if (!fs.existsSync(this.config.dataDir)) {
      fs.mkdirSync(this.config.dataDir, { recursive: true });
    }

    // 尝试 LanceDB
    if (this.config.mode === 'auto' || this.config.mode === 'lancedb') {
      try {
        await this._initLanceDB();
        this._mode = 'lancedb';
        console.log('✅ LanceDB 向量存储已就绪');
        return;
      } catch (err) {
        console.warn(`⚠️  LanceDB 不可用 (${err.message})，降级到内存模式`);
      }
    }

    // 降级：内存模式
    await this._initMemory();
    this._mode = 'memory';
    console.log('✅ 内存向量存储已就绪 (降级模式)');
  }

  /**
   * 添加文档向量
   * @param {string} id - 文档 ID
   * @param {number[]} vector - 嵌入向量
   * @param {object} metadata - 元数据 (title, path, level 等)
   */
  async add(id, vector, metadata = {}) {
    if (this._mode === 'lancedb') {
      await this._addLanceDB(id, vector, metadata);
    } else {
      await this._addMemory(id, vector, metadata);
    }
    // 清除相关缓存
    this._cache.clear();
  }

  /**
   * 批量添加
   * @param {Array<{id, vector, metadata}>} items
   */
  async addBatch(items) {
    for (const item of items) {
      await this.add(item.id, item.vector, item.metadata || {});
    }
  }

  /**
   * 语义搜索
   * @param {number[]} queryVec - 查询向量
   * @param {number} topK - 返回数量
   * @returns {Promise<Array<{id, score, metadata}>>}
   */
  async search(queryVec, topK = 10) {
    // 检查缓存
    const cacheKey = `${queryVec.slice(0, 10).join(',')}_${topK}`;
    if (this._cache.has(cacheKey)) {
      return this._cache.get(cacheKey);
    }

    let results;
    if (this._mode === 'lancedb') {
      results = await this._searchLanceDB(queryVec, topK);
    } else {
      results = this._searchMemory(queryVec, topK);
    }

    // 缓存结果
    if (this._cache.size >= this.config.cacheSize) {
      const firstKey = this._cache.keys().next().value;
      this._cache.delete(firstKey);
    }
    this._cache.set(cacheKey, results);

    return results;
  }

  /**
   * 获取存储统计
   */
  getStats() {
    return {
      mode: this._mode,
      count: this._mode === 'memory' ? this._memoryVectors.length : 'N/A',
      dimension: this.config.dimension,
      dataDir: this.config.dataDir,
    };
  }

  // ─── LanceDB 实现 ───────────────────────────────

  async _initLanceDB() {
    const lancedb = await import('vectordb');

    const dbPath = path.join(this.config.dataDir, 'lancedb');
    this._db = await lancedb.connect(dbPath);

    // 尝试打开已有表，不存在则创建
    try {
      this._table = await this._db.openTable(this.config.collectionName);
    } catch {
      // 表不存在，稍后在第一次 add 时创建
      this._table = null;
    }
  }

  async _ensureTable() {
    if (this._table) return;
    this._table = await this._db.createTable(this.config.collectionName, [
      { id: '_init_', vector: new Array(this.config.dimension).fill(0), title: '', path: '', level: 0 },
    ]);
  }

  async _addLanceDB(id, vector, metadata) {
    await this._ensureTable();
    await this._table.add([{
      id,
      vector,
      title: metadata.title || '',
      path: metadata.path || '',
      level: metadata.level || 0,
    }]);
  }

  async _searchLanceDB(queryVec, topK) {
    await this._ensureTable();
    const results = await this._table.search(queryVec).limit(topK).execute();
    return results
      .filter(r => r.id !== '_init_')
      .map(r => ({
        id: r.id,
        score: 1 - (r._distance || 0), // 距离转相似度
        metadata: { title: r.title, path: r.path, level: r.level },
      }));
  }

  // ─── 内存降级实现 ───────────────────────────────

  async _initMemory() {
    // 从 JSON 文件恢复
    const backupFile = path.join(this.config.dataDir, 'vectors_backup.json');
    if (fs.existsSync(backupFile)) {
      try {
        const data = JSON.parse(fs.readFileSync(backupFile, 'utf-8'));
        this._memoryVectors = data.vectors || [];
        for (const [id, meta] of Object.entries(data.metadata || {})) {
          this._memoryMeta.set(id, meta);
        }
        console.log(`  从备份恢复 ${this._memoryVectors.length} 条向量`);
      } catch (_) { /* 文件损坏，从零开始 */ }
    }
  }

  async _addMemory(id, vector, metadata) {
    // 更新或追加
    const idx = this._memoryVectors.findIndex(v => v.id === id);
    if (idx >= 0) {
      this._memoryVectors[idx] = { id, vector };
    } else {
      this._memoryVectors.push({ id, vector });
    }
    this._memoryMeta.set(id, metadata);

    // 定期持久化（每 50 条）
    if (this._memoryVectors.length % 50 === 0) {
      await this._saveToDisk();
    }
  }

  _searchMemory(queryVec, topK) {
    const scored = this._memoryVectors.map(item => ({
      id: item.id,
      score: this._cosineSimilarity(queryVec, item.vector),
      metadata: this._memoryMeta.get(item.id) || {},
    }));

    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * 余弦相似度
   */
  _cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * 持久化到磁盘
   */
  async _saveToDisk() {
    const backupFile = path.join(this.config.dataDir, 'vectors_backup.json');
    const metadata = {};
    for (const [id, meta] of this._memoryMeta) {
      metadata[id] = meta;
    }
    fs.writeFileSync(backupFile, JSON.stringify({
      vectors: this._memoryVectors,
      metadata,
    }, null, 2), 'utf-8');
  }
}

/** 默认单例 */
export const vectorStore = new VectorStore();
