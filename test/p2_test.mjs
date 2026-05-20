/**
 * EvoIndex 3.0 — P2 验证测试
 *
 * 测试：
 *   1. CMA-ES 权重自进化 (自动优化 RRF 三路权重)
 *   2. 查询缓存命中率
 *   3. 重复查询延迟 (<5ms 目标)
 *
 * 用法: node test/p2_test.mjs <index.json> [--data-dir <path>]
 */

import { HybridQueryEngineV3, QueryEngineConfig } from '../src/core/hybrid_query.mjs';
import { VectorStore, VectorStoreConfig } from '../src/core/lancedb_store.mjs';
import { GraphStore, GraphStoreConfig } from '../src/core/graph_store.mjs';
import { QueryCache } from '../src/core/query_cache.mjs';
import { CMAESTuner } from '../src/core/cmaes_tuner.mjs';
import fs from 'fs';

async function main() {
  const args = process.argv.slice(2);
  let indexPath = null;
  let dataDir = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--data-dir' && i + 1 < args.length) {
      dataDir = args[++i];
    } else if (!indexPath) {
      indexPath = args[i];
    }
  }

  if (!indexPath) {
    console.log('用法: node test/p2_test.mjs <index.json> [--data-dir <path>]');
    process.exit(1);
  }

  if (process.platform === 'win32' && !dataDir) {
    dataDir = 'C:\\Users\\cuihao\\evoindex_data';
  }

  console.log('🧪 EvoIndex 3.0 P2 验证测试 — 自进化 + 记忆\n');
  console.log('═'.repeat(60));
  if (dataDir) console.log(`📁 数据目录: ${dataDir}\n`);

  // 1. 初始化引擎
  const vectorStore = new VectorStore(new VectorStoreConfig({ dataDir }));
  const gStore = new GraphStore(new GraphStoreConfig({ dataDir }));
  const cache = new QueryCache();

  const config = new QueryEngineConfig({
    vectorStore,
    graphStore: gStore,
    queryCache: cache,
    fusionWeights: { tree: 0.33, vector: 0.33, graph: 0.34 }, // 均匀初始值
  });
  const engine = new HybridQueryEngineV3(config);
  await engine.loadIndex(indexPath);

  // 构建图
  console.log('📌 Step 0: 构建图索引');
  const graphStats = await engine.buildGraphIndex();
  console.log(`  图: ${graphStats.nodes} 节点, ${graphStats.edges} 边, ${graphStats.communities} 社区\n`);

  // 2. CMA-ES 权重自进化
  console.log('📌 Step 1: CMA-ES 权重自进化');
  console.log('─'.repeat(50));

  // 训练查询集：query + 期望的排名提升指标
  const trainQueries = [
    { q: '什么是RAG', domain: 'llm' },
    { q: '检索增强生成', domain: 'llm' },
    { q: '大模型的微调方法', domain: 'llm' },
    { q: '向量数据库选型', domain: 'llm' },
    { q: 'GraphRAG的工作原理', domain: 'llm' },
    { q: '知识图谱构建', domain: 'llm' },
    { q: '嵌入模型对比', domain: 'llm' },
    { q: '混合检索策略', domain: 'llm' },
    { q: 'FDA审批流程', domain: 'pharma' },
    { q: 'GLP-1受体激动剂', domain: 'pharma' },
  ];

  /**
   * 评估函数：给定权重，返回 MRR 分数
   */
  async function evaluate(weights) {
    // 临时修改权重
    const saved = { ...engine.config.fusionWeights };
    engine.config.fusionWeights = { tree: weights[0], vector: weights[1], graph: weights[2] };
    cache.clear(); // 清除缓存避免干扰

    let totalMRR = 0;
    for (const { q } of trainQueries) {
      const result = await engine.query(q, { skipCache: true });
      totalMRR += result.results[0]?.score || 0;
    }

    // 恢复
    engine.config.fusionWeights = saved;
    return totalMRR / trainQueries.length;
  }

  // 初始化 CMA-ES
  const tuner = new CMAESTuner(3, {
    lambda: 8,
    sigma: 0.3,
    mean: [0.45, 0.45, 0.1], // P1 最优值作为起点
    bounds: { min: 0.05, max: 0.9 },
  });

  const generations = 8; // 快速收敛
  console.log(`  参数: ${tuner.dim}维, ${tuner.lambda}个体, ${generations}代\n`);

  for (let gen = 0; gen < generations; gen++) {
    const population = tuner.sample();
    const scores = [];
    const results = [];

    // 顺序评估（避免 LM Studio 过载）
    for (let i = 0; i < population.length; i++) {
      const score = await evaluate(population[i]);
      scores.push(score);

      const w = population[i];
      results.push({
        w: `${(w[0]*100).toFixed(0)}:${(w[1]*100).toFixed(0)}:${(w[2]*100).toFixed(0)}`,
        mrr: (score * 100).toFixed(1),
      });
    }

    const status = tuner.update(population, scores);

    // 显示进度
    const best = population[scores.indexOf(Math.max(...scores))];
    console.log(`  Gen ${gen + 1}/${generations} | 最优 MRR: ${(status.bestScore * 100).toFixed(1)}% | 权重: ${(best[0]*100).toFixed(0)}:${(best[1]*100).toFixed(0)}:${(best[2]*100).toFixed(0)}`);

    // 避免请求过快
    await new Promise(r => setTimeout(r, 200));
  }

  // 最优权重
  const bestWeights = tuner.getBestWeights();
  console.log(`\n  ✅ CMA-ES 收敛完成`);
  console.log(`  最优权重: 树:${(bestWeights.tree*100).toFixed(0)}% 向量:${(bestWeights.vector*100).toFixed(0)}% 图:${(bestWeights.graph*100).toFixed(0)}%`);

  // 3. 应用最优权重，重跑查询
  console.log('\n📌 Step 2: 应用最优权重验证');
  engine.config.fusionWeights = bestWeights;
  cache.clear();

  let optimizedMRR = 0;
  for (const { q } of trainQueries) {
    const result = await engine.query(q, { skipCache: true });
    const mrr = result.results[0]?.score || 0;
    optimizedMRR += mrr;
    console.log(`  "${q}" → Top-1: [${(mrr*100).toFixed(0)}%] ${result.results[0]?.metadata?.title || result.results[0]?.id || 'N/A'}`);
  }
  optimizedMRR /= trainQueries.length;
  console.log(`\n  优化后 MRR: ${(optimizedMRR * 100).toFixed(1)}%`);

  // 4. 查询缓存测试
  console.log('\n📌 Step 3: 查询缓存验证');

  // 先跑一遍预热缓存
  const cacheQuery = '什么是RAG - 重复查询测试';
  await engine.query(cacheQuery);

  // 测试重复查询延迟
  const repeatCount = 5;
  const latencies = [];
  for (let i = 0; i < repeatCount; i++) {
    const start = Date.now();
    const result = await engine.query(cacheQuery);
    const elapsed = Date.now() - start;
    latencies.push(elapsed);
    console.log(`  第${i+1}次: ${elapsed}ms ${result.fromCache ? '⚡缓存' : '🔍全链路'}`);
  }

  const avgLatency = latencies.reduce((s, l) => s + l, 0) / latencies.length;
  const minLatency = Math.min(...latencies);

  // 5. 模糊匹配测试
  const fuzzyQuery = '什么是RAG（模糊匹配测试）';
  const fuzzyResult = await engine.query(fuzzyQuery);
  console.log(`\n  模糊查询: "${fuzzyQuery}" → ${fuzzyResult.fromCache ? '⚡缓存命中' : '🔍全链路'}`);

  const cacheStats = cache.getStats();

  // 6. 结果汇总
  console.log('\n' + '═'.repeat(60));
  console.log('\n📊 P2 测试结果汇总\n');

  console.log('  ── CMA-ES 自进化 ──');
  console.log(`  最优权重:     树:${(bestWeights.tree*100).toFixed(0)}% 向量:${(bestWeights.vector*100).toFixed(0)}% 图:${(bestWeights.graph*100).toFixed(0)}%`);
  console.log(`  进化代数:     ${generations}`);
  console.log(`  优化后 MRR:   ${(optimizedMRR * 100).toFixed(1)}%`);
  console.log(`  进化历史:     ${tuner.getHistory().length} 代`);
  if (tuner.getHistory().length >= 2) {
    const first = tuner.getHistory()[0];
    const last = tuner.getHistory()[tuner.getHistory().length - 1];
    console.log(`  MRR 提升:     ${(first.bestScore*100).toFixed(1)}% → ${(last.bestScore*100).toFixed(1)}% (+${((last.bestScore-first.bestScore)*100).toFixed(1)}pp)`);
  }

  console.log('\n  ── 查询缓存 ──');
  console.log(`  命中率:       ${cacheStats.hitRate}`);
  console.log(`  L1 精确命中:  ${cacheStats.l1Hits}`);
  console.log(`  L2 语义命中:  ${cacheStats.l2Hits}`);
  console.log(`  缓存条目:     ${cacheStats.l1Size}`);
  console.log(`  重复查询延迟: ${minLatency}ms (min) / ${avgLatency.toFixed(1)}ms (avg)`);
  console.log(`  延迟目标:     ${minLatency < 5 ? '✅' : '⚠️'} <5ms`);

  // 系统状态
  console.log('\n📌 系统状态');
  console.log(`  向量存储: ${engine.config.vectorStore.getStats().mode}`);
  console.log(`  图存储:   ${engine.config.graphStore.getStats().nodes} 节点`);
  console.log(`  融合模式: ${engine.config.fusionMode}`);
  console.log(`  融合权重: 树:${(engine.config.fusionWeights.tree*100).toFixed(0)}% 向量:${(engine.config.fusionWeights.vector*100).toFixed(0)}% 图:${(engine.config.fusionWeights.graph*100).toFixed(0)}%`);

  console.log('\n✅ P2 测试完成');
}

main().catch(err => {
  console.error('❌ 测试失败:', err);
  process.exit(1);
});
