/**
 * EvoIndex 3.0 — Embedding 客户端
 *
 * 调用本地 nomic-embed-text-v1.5 生成文本嵌入向量
 * 支持 LM Studio (OpenAI-compatible API) 和 Ollama 双后端
 *
 * @module embedding_client
 * @version 3.0.0
 */

/**
 * Embedding 客户端配置
 */
export class EmbeddingConfig {
  constructor(options = {}) {
    /** LM Studio 端点（默认） */
    this.lmStudioURL = options.lmStudioURL ||
      process.env.LM_STUDIO_URL || 'http://127.0.0.1:1234/v1';

    /** LM Studio 备用端点（WSL→Windows 桥接） */
    this.lmStudioFallbackURL = options.lmStudioFallbackURL ||
      process.env.LM_STUDIO_FALLBACK_URL || '';

    /** Ollama 端点（备选） */
    this.ollamaURL = options.ollamaURL ||
      process.env.OLLAMA_URL || 'http://172.16.8.69:11434/api';

    /** 嵌入模型名称 */
    this.model = options.model || 'nomic-embed-text-v1.5';

    /** 后端选择：'lmstudio' | 'ollama' | 'auto' */
    this.backend = options.backend || 'auto';

    /** 请求超时 (ms) */
    this.timeout = options.timeout || 15000;

    /** 最大重试次数 */
    this.maxRetries = options.maxRetries || 2;
  }
}

/**
 * Embedding 客户端
 *
 * 用法:
 *   const client = new EmbeddingClient();
 *   const vec = await client.embed('肺结核的治疗方案');
 *   const vecs = await client.embedBatch(['文本1', '文本2', '文本3']);
 */
export class EmbeddingClient {
  constructor(config = new EmbeddingConfig()) {
    this.config = config;
    this._activeBackend = null;
    this._healthy = false;
  }

  /**
   * 健康检查 — 自动探测可用后端
   * @returns {Promise<string>} 可用后端名称
   */
  async detectBackend() {
    if (this.config.backend !== 'auto') {
      return this.config.backend;
    }

    // 先试 LM Studio
    try {
      const resp = await fetch(`${this.config.lmStudioURL}/models`, {
        signal: AbortSignal.timeout(3000),
      });
      if (resp.ok) {
        this._activeBackend = 'lmstudio';
        this._healthy = true;
        return 'lmstudio';
      }
    } catch (_) { /* LM Studio 不可用，试 Ollama */ }

    // 再试 Ollama
    try {
      const resp = await fetch(`${this.config.ollamaURL}/tags`, {
        signal: AbortSignal.timeout(3000),
      });
      if (resp.ok) {
        this._activeBackend = 'ollama';
        this._healthy = true;
        return 'ollama';
      }
    } catch (_) { /* 两个都不可用 */ }

    throw new Error('No embedding backend available (tried LM Studio + Ollama)');
  }

  /**
   * 单文本嵌入
   * @param {string} text - 输入文本
   * @returns {Promise<number[]>} 嵌入向量
   */
  async embed(text) {
    if (!this._activeBackend) {
      await this.detectBackend();
    }

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await this._callEmbedAPI(text);
      } catch (err) {
        if (attempt === this.config.maxRetries) throw err;
        // 重试前等一会
        await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  }

  /**
   * 批量嵌入
   * @param {string[]} texts - 文本数组
   * @returns {Promise<number[][]>} 嵌入向量数组
   */
  async embedBatch(texts) {
    const vectors = [];
    for (const text of texts) {
      vectors.push(await this.embed(text));
    }
    return vectors;
  }

  /**
   * 内部：调用嵌入 API
   */
  async _callEmbedAPI(text) {
    if (this._activeBackend === 'lmstudio') {
      return this._embedLMStudio(text);
    } else {
      return this._embedOllama(text);
    }
  }

  /**
   * LM Studio OpenAI-compatible embeddings API
   */
  async _embedLMStudio(text) {
    const urls = [this.config.lmStudioURL];
    if (this.config.lmStudioFallbackURL) {
      urls.push(this.config.lmStudioFallbackURL);
    }

    let lastError;
    for (const baseURL of urls) {
      try {
        const url = `${baseURL}/embeddings`;
        const resp = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: this.config.model,
            input: text,
          }),
          signal: AbortSignal.timeout(this.config.timeout),
        });

        if (!resp.ok) {
          const err = await resp.text();
          throw new Error(`LM Studio embedding error ${resp.status}: ${err}`);
        }

        const data = await resp.json();
        return data.data?.[0]?.embedding || data.embedding || [];
      } catch (err) {
        lastError = err;
        // 继续尝试下一个 URL
      }
    }
    throw lastError || new Error('All LM Studio endpoints failed');
  }

  /**
   * Ollama embeddings API
   */
  async _embedOllama(text) {
    const url = `${this.config.ollamaURL}/embeddings`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.config.model,
        prompt: text,
      }),
      signal: AbortSignal.timeout(this.config.timeout),
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`Ollama embedding error ${resp.status}: ${err}`);
    }

    const data = await resp.json();
    return data.embedding || [];
  }

  /**
   * 获取向量维度
   */
  async getDimension() {
    const vec = await this.embed('test');
    return vec.length;
  }

  /**
   * 客户端状态
   */
  getStatus() {
    return {
      backend: this._activeBackend || 'unknown',
      healthy: this._healthy,
      model: this.config.model,
    };
  }
}

/** 默认单例 */
export const embeddingClient = new EmbeddingClient();
