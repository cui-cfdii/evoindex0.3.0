/**
 * EvoIndex 3.0 — 后端健康监控 + 运行时切换 (P3)
 *
 * 增强 EmbeddingClient:
 *   - 周期性健康检查
 *   - 自动故障切换
 *   - 延迟统计
 *   - 断路器模式
 *
 * @module backend_monitor
 * @version 3.0.0
 */

/**
 * 后端健康状态
 */
class BackendHealth {
  constructor(name, url) {
    this.name = name;
    this.url = url;
    this.healthy = true;
    this.lastCheck = 0;
    this.lastLatency = 0;
    this.failCount = 0;
    this.successCount = 0;
    this.circuitOpen = false;
    this.circuitOpenUntil = 0;
  }

  recordSuccess(latency) {
    this.healthy = true;
    this.lastLatency = latency;
    this.successCount++;
    this.failCount = Math.max(0, this.failCount - 1);
    this.circuitOpen = false;
  }

  recordFailure() {
    this.failCount++;
    this.lastCheck = Date.now();

    // 断路器: 连续 3 次失败则熔断 30 秒
    if (this.failCount >= 3 && !this.circuitOpen) {
      this.circuitOpen = true;
      this.circuitOpenUntil = Date.now() + 30000;
      this.healthy = false;
      console.warn(`🔴 后端 ${this.name} 熔断 30s (连续 ${this.failCount} 次失败)`);
    }
  }

  canTry() {
    if (!this.circuitOpen) return true;
    if (Date.now() > this.circuitOpenUntil) {
      // 半开状态：允许试探
      this.circuitOpen = false;
      return true;
    }
    return false;
  }

  getStats() {
    return {
      name: this.name,
      healthy: this.healthy,
      latency: this.lastLatency,
      successCount: this.successCount,
      failCount: this.failCount,
      circuitOpen: this.circuitOpen,
    };
  }
}

/**
 * 后端健康监控器
 *
 * 管理多后端的状态、自动故障切换和周期性探测
 */
export class BackendMonitor {
  constructor() {
    this._backends = new Map();
    this._active = null;
    this._fallbackOrder = []; // 优先顺序
    this._totalRequests = 0;
    this._totalFailures = 0;
    this._totalLatency = 0;
  }

  /**
   * 注册后端
   */
  register(name, url) {
    this._backends.set(name, new BackendHealth(name, url));
    if (!this._fallbackOrder.includes(name)) {
      this._fallbackOrder.push(name);
    }
    return this;
  }

  /**
   * 标记主后端
   */
  setActive(name) {
    this._active = name;
    return this;
  }

  /**
   * 记录成功调用
   */
  recordSuccess(backend, latency) {
    this._totalRequests++;
    this._totalLatency += latency;

    const health = this._backends.get(backend);
    if (health) health.recordSuccess(latency);
  }

  /**
   * 记录失败 + 自动切换
   * @returns {string|null} 新的活跃后端名, 或 null 表示全部不可用
   */
  recordFailure(backend) {
    this._totalRequests++;
    this._totalFailures++;

    const health = this._backends.get(backend);
    if (health) health.recordFailure();

    // 尝试切换到下一个健康后端
    return this._findHealthy();
  }

  /**
   * 获取当前活跃后端的 URL
   */
  getActiveURL() {
    const health = this._backends.get(this._active);
    return health?.url || null;
  }

  /**
   * 获取活跃后端名
   */
  getActiveName() {
    return this._active;
  }

  /**
   * 是否有可用后端
   */
  isAvailable() {
    return this._active !== null && this._findHealthy() !== null;
  }

  /**
   * 获取统计
   */
  getStats() {
    const backendStats = {};
    for (const [name, health] of this._backends) {
      backendStats[name] = health.getStats();
    }

    return {
      active: this._active,
      backends: backendStats,
      totalRequests: this._totalRequests,
      totalFailures: this._totalFailures,
      avgLatency: this._totalRequests > 0
        ? (this._totalLatency / this._totalRequests).toFixed(1) + 'ms'
        : 'N/A',
      failureRate: this._totalRequests > 0
        ? (this._totalFailures / this._totalRequests * 100).toFixed(1) + '%'
        : '0%',
    };
  }

  // ─── 内部 ─────────────────

  _findHealthy() {
    // 先检查当前活跃后端
    if (this._active) {
      const current = this._backends.get(this._active);
      if (current?.canTry() && current?.healthy) {
        return this._active;
      }
    }

    // 按优先级查找
    for (const name of this._fallbackOrder) {
      if (name === this._active) continue; // 已检查过
      const health = this._backends.get(name);
      if (health?.canTry() && health?.healthy) {
        console.log(`🔄 后端切换: ${this._active || 'none'} → ${name}`);
        this._active = name;
        return name;
      }
    }

    // 全部不可用
    console.error('🔴 所有后端不可用!');
    this._active = null;
    return null;
  }
}

/** 全局监控实例 */
export const backendMonitor = new BackendMonitor();
