/**
 * EvoIndex 3.0 — CMA-ES 权重自进化调优器 (P2)
 *
 * 用 CMA-ES 自动优化 RRF 三路融合权重 (tree/vector/graph)
 * 输入：一组测试查询 + ground truth (预期命中文档ID)
 * 输出：最优权重组合
 *
 * 简化版：3维参数空间，小种群快速收敛
 *
 * @module cmaes_tuner
 * @version 3.0.0
 */

/**
 * CMA-ES 调优器
 *
 * 用法:
 *   const tuner = new CMAESTuner(3);
 *   for (let gen = 0; gen < 10; gen++) {
 *     const population = tuner.sample();
 *     const scores = await evaluateAll(population); // 并行评估
 *     tuner.update(population, scores);
 *   }
 *   console.log('最优权重:', tuner.bestParams);
 */
export class CMAESTuner {
  constructor(dim = 3, config = {}) {
    this.dim = dim;

    // 种群参数
    this.lambda = config.lambda || 8;          // 每代候选数
    this.mu = config.mu || Math.floor(this.lambda / 2); // 存活数

    // 学习率
    this.sigma = config.sigma || 0.3;          // 初始步长
    this.cc = 4 / (dim + 4);                   // 秩1累积率
    this.cs = (this.mu + 2) / (dim + this.mu + 5);
    this.c1 = 2 / ((dim + 1.3) ** 2);
    this.cmu = Math.min(1 - this.c1, 2 * (this.mu - 2 + 1/this.mu) / ((dim + 2) ** 2 + this.mu));
    this.damps = 1 + 2 * Math.max(0, Math.sqrt((this.mu - 1) / (dim + 1)) - 1) + 2/this.mu;

    // 权重
    this.weights = [];
    for (let i = 0; i < this.mu; i++) {
      this.weights.push(Math.log(this.mu + 1) - Math.log(i + 1));
    }
    const sum = this.weights.reduce((s, w) => s + w, 0);
    this.weights = this.weights.map(w => w / sum);
    this.muEff = 1 / this.weights.reduce((s, w) => s + w * w, 0);

    // 状态变量
    this.mean = config.mean || new Array(dim).fill(0.33); // 初始均匀分布
    this.pc = new Array(dim).fill(0);
    this.ps = new Array(dim).fill(0);
    this.C = this._eye(dim);
    this.B = this._eye(dim);
    this.D = new Array(dim).fill(1);

    // 边界
    this.bounds = config.bounds || { min: 0.05, max: 0.9 };

    // 历史
    this.history = [];
    this.bestScore = -Infinity;
    this.bestParams = [...this.mean];
    this.generation = 0;
  }

  /**
   * 采样一代种群
   * @returns {Array<number[]>} 种群个体 (每个是 dim 维向量)
   */
  sample() {
    const population = [];
    const chol = this._cholesky(this.C);

    for (let i = 0; i < this.lambda; i++) {
      // z ~ N(0, I)
      const z = new Array(this.dim);
      for (let j = 0; j < this.dim; j++) {
        z[j] = this._randn();
      }

      // y = B * D * z
      const y = new Array(this.dim).fill(0);
      for (let r = 0; r < this.dim; r++) {
        for (let c = 0; c < this.dim; c++) {
          y[r] += chol[r][c] * z[c];
        }
      }

      // x = mean + sigma * y
      const x = new Array(this.dim);
      for (let j = 0; j < this.dim; j++) {
        x[j] = this.mean[j] + this.sigma * y[j];
        // 边界钳制
        x[j] = Math.max(this.bounds.min, Math.min(this.bounds.max, x[j]));
      }

      // 归一化到 sum=1
      const total = x.reduce((s, v) => s + Math.abs(v), 0);
      population.push(x.map(v => Math.abs(v) / total));
    }

    return population;
  }

  /**
   * 用评估分数更新分布
   * @param {Array<number[]>} population - 种群
   * @param {number[]} scores - 对应分数 (越高越好)
   */
  update(population, scores) {
    this.generation++;

    // 排序：按分数降序
    const ranked = population.map((p, i) => ({ params: p, score: scores[i] }))
      .sort((a, b) => b.score - a.score);

    const best = ranked[0];
    if (best.score > this.bestScore) {
      this.bestScore = best.score;
      this.bestParams = [...best.params];
    }

    this.history.push({
      gen: this.generation,
      bestScore: best.score,
      bestParams: [...best.params],
      meanScore: ranked.slice(0, this.mu).reduce((s, r) => s + r.score, 0) / this.mu,
    });

    // 取 top-mu 个体
    const survivors = ranked.slice(0, this.mu);

    // 计算加权均值
    const oldMean = [...this.mean];
    for (let j = 0; j < this.dim; j++) {
      this.mean[j] = 0;
      for (let i = 0; i < this.mu; i++) {
        this.mean[j] += this.weights[i] * survivors[i].params[j];
      }
    }

    // 更新累积路径
    const invSigma = 1 / Math.max(this.sigma, 1e-10);
    const yMean = new Array(this.dim);
    for (let j = 0; j < this.dim; j++) {
      yMean[j] = (this.mean[j] - oldMean[j]) * invSigma;
    }

    // ps 更新 (步长路径)
    const invSqrtC = this._invSqrtC();
    let cyMean = new Array(this.dim).fill(0);
    for (let r = 0; r < this.dim; r++) {
      for (let c = 0; c < this.dim; c++) {
        cyMean[r] += invSqrtC[r][c] * yMean[c];
      }
    }

    for (let j = 0; j < this.dim; j++) {
      this.ps[j] = (1 - this.cs) * this.ps[j] + Math.sqrt(this.cs * (2 - this.cs) * this.muEff) * cyMean[j];
    }

    // pc 更新 (协方差路径)
    for (let j = 0; j < this.dim; j++) {
      this.pc[j] = (1 - this.cc) * this.pc[j] + Math.sqrt(this.cc * (2 - this.cc) * this.muEff) * yMean[j];
    }

    // 协方差矩阵更新
    const hsig = this._norm(this.ps) / Math.sqrt(1 - (1 - this.cs) ** (2 * this.generation))
      / (1.5 + 1 / (this.dim + 1)) < 1.4 ? 1 : 0;
    const delta = (1 - hsig) * this.cc * (2 - this.cc);

    for (let i = 0; i < this.dim; i++) {
      for (let j = i; j < this.dim; j++) {
        this.C[i][j] = (1 - this.c1 - this.cmu) * this.C[i][j]
          + this.c1 * (this.pc[i] * this.pc[j] + delta * this.C[i][j]);

        for (let k = 0; k < this.mu; k++) {
          const z = new Array(this.dim);
          for (let d = 0; d < this.dim; d++) {
            z[d] = (survivors[k].params[d] - oldMean[d]) * invSigma;
          }
          this.C[i][j] += this.cmu * this.weights[k] * z[i] * z[j];
        }

        // 对称
        if (i !== j) this.C[j][i] = this.C[i][j];
      }
    }

    // 步长更新
    const expectedNorm = Math.sqrt(this.dim) * (1 - 1/(4*this.dim) + 1/(21*this.dim*this.dim));
    this.sigma *= Math.exp((this.cs / this.damps) * (this._norm(this.ps) / expectedNorm - 1));
    this.sigma = Math.max(0.01, Math.min(0.5, this.sigma));

    // 周期性重新归一化协方差
    if (this.generation % 5 === 0) {
      this._normalizeC();
    }

    return {
      bestScore: this.bestScore,
      bestParams: [...this.bestParams],
      meanScore: ranked[0].score,
    };
  }

  /**
   * 获取最优权重 (归一化)
   */
  getBestWeights() {
    const p = [...this.bestParams];
    const total = p.reduce((s, v) => s + v, 0);
    return {
      tree: p[0] / total,
      vector: p[1] / total,
      graph: p[2] / total,
    };
  }

  getHistory() {
    return this.history;
  }

  // ─── 线性代数工具 ──────────────────

  _eye(n) {
    const m = [];
    for (let i = 0; i < n; i++) {
      m[i] = new Array(n).fill(0);
      m[i][i] = 1;
    }
    return m;
  }

  _norm(v) {
    return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  }

  _randn() {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  _cholesky(A) {
    const n = A.length;
    const L = new Array(n).fill(null).map(() => new Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        if (i === j) {
          L[i][j] = Math.sqrt(Math.max(1e-10, A[i][i] - sum));
        } else {
          L[i][j] = (A[i][j] - sum) / Math.max(1e-10, L[j][j]);
        }
      }
    }
    return L;
  }

  _invSqrtC() {
    // 通过特征分解 C = B * diag(D^2) * B^T 来计算 C^{-1/2} = B * diag(1/D) * B^T
    // 简化：用 Cholesky 逆
    const chol = this._cholesky(this.C);
    const n = this.dim;
    // L^{-1} (下三角求逆)
    const Linv = new Array(n).fill(null).map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      Linv[i][i] = 1 / Math.max(1e-10, chol[i][i]);
      for (let j = 0; j < i; j++) {
        let sum = 0;
        for (let k = j; k < i; k++) {
          sum += chol[i][k] * Linv[k][j];
        }
        Linv[i][j] = -sum / Math.max(1e-10, chol[i][i]);
      }
    }
    return Linv;
  }

  _normalizeC() {
    // 保持协方差矩阵数值稳定
    const trace = this.C.reduce((s, row, i) => s + row[i], 0);
    const scale = this.dim / Math.max(trace, 1e-10);
    for (let i = 0; i < this.dim; i++) {
      for (let j = 0; j < this.dim; j++) {
        this.C[i][j] *= scale;
      }
    }
  }
}
