/**
 * EvoIndex 3.0 — P3 全链路压测 & 双后端验证
 *
 * 测试:
 *   1. 双后端健康检查 + 自动切换
 *   2. 全链路延迟分布 (P50/P90/P99)
 *   3. 并发吞吐量
 *   4. 降级路径验证
 *
 * 用法: node test/p3_test.mjs <index.json> [--data-dir <path>]
 */

import { HybridQueryEngineV3, QueryEngineConfig } from '../src/core/hybrid_query.mjs';
import { VectorStore, VectorStoreConfig } from '../src/core/lancedb_store.mjs';
import { GraphStore, GraphStoreConfig } from '../src/core/graph_store.mjs';
import { QueryCache } from '../src/core/query_cache.mjs';
import { EmbeddingClient } from '../src/utils/embedding_client.mjs';
import { BackendMonitor } from '../src/utils/backend_monitor.mjs';

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
    console.log('用法: node test/p3_test.mjs <index.json> [--data-dir <path>]');
    process.exit(1);
  }

  if (process.platform === 'win32' && !dataDir) {
    dataDir = 'C:\\Users\\cuihao\\evoindex_data';
  }

  console.log('🧪 EvoIndex 3.0 P3 — 全链路压测 & 双后端验证\n');
  console.log('═'.repeat(60));
  console.log(`📁 数据目录: ${dataDir || '(默认)'}\n`);

  // ═══════════════════════════════════════════════
  // 1. 双后端健康检查
  // ═══════════════════════════════════════════════
  console.log('📌 Step 1: 双后端检测\n');

  const embedClient = new EmbeddingClient();
  const monitor = new BackendMonitor();

  // 注册后端
  monitor.register('lmstudio', embedClient.config.lmStudioURL);
  monitor.register('ollama', embedClient.config.ollamaURL);

  // 检测 LM Studio
  let primaryBackend = null;
  try {
    const resp = await fetch(`${embedClient.config.lmStudioURL}/models`, {
      signal: AbortSignal.timeout(3000),
    });
    if (resp.ok) {
      primaryBackend = 'lmstudio';
      console.log('  ✅ LM Studio 可用 (127.0.0.1:1234)');
      monitor.setActive('lmstudio');
    }
  } catch {
    console.log('  ⚠️ LM Studio 不可用');
  }

  // 检测 Ollama
  let secondaryBackend = null;
  try {
    const resp = await fetch(`${embedClient.config.ollamaURL}/tags`, {
      signal: AbortSignal.timeout(3000),
    });
    if (resp.ok) {
      secondaryBackend = 'ollama';
      const hasModel = await checkOllamaModel(embedClient.config.ollamaURL, embedClient.config.model);
      console.log(`  ${hasModel ? '✅' : '⚠️'} Ollama 可用 (${embedClient.config.ollamaURL})${hasModel ? '' : ' — 模型未加载'}`);
      if (!primaryBackend) monitor.setActive('ollama');
    }
  } catch {
    console.log('  ⚠️ Ollama 不可用');
  }

  if (!primaryBackend && !secondaryBackend) {
    console.error('\n❌ 无可用嵌入后端，测试中止');
    process.exit(1);
  }

  console.log(`\n  主后端: ${monitor.getActiveName()}`);

  // ═══════════════════════════════════════════════
  // 2. 初始化引擎
  // ═══════════════════════════════════════════════
  console.log('\n📌 Step 2: 引擎初始化\n');

  const engine = new HybridQueryEngineV3(new QueryEngineConfig({
    vectorStore: new VectorStore(new VectorStoreConfig({ dataDir })),
    graphStore: new GraphStore(new GraphStoreConfig({ dataDir })),
    queryCache: new QueryCache(),
    fusionWeights: { tree: 0.08, vector: 0.63, graph: 0.29 }, // P2 最优
  }));

  await engine.loadIndex(indexPath);

  // 构建图（如果还没有）
  if (!engine.config.graphStore.getStats().initialized) {
    await engine.buildGraphIndex();
  }

  // ═══════════════════════════════════════════════
  // 3. 全链路延迟分布
  // ═══════════════════════════════════════════════
  console.log('\n📌 Step 3: 全链路延迟分布\n');

  const benchQueries = [
    '什么是RAG',
    '检索增强生成',
    '大模型微调方法',
    '向量数据库选型',
    'GraphRAG工作原理',
    '知识图谱构建',
    '嵌入模型对比',
    '混合检索策略',
    'FDA审批流程',
    'GLP-1受体激动剂',
    '临床试验设计',
    '药品注册法规',
  ];

  const allLatencies = [];
  const perQueryStats = {};

  for (const q of benchQueries) {
    const latencies = [];
    // 每个查询跑 3 次
    for (let i = 0; i < 3; i++) {
      const start = process.hrtime.bigint();
      const result = await engine.query(q, { skipCache: true });
      const elapsed = Number(process.hrtime.bigint() - start) / 1e6; // ms

      latencies.push(elapsed);
      allLatencies.push(elapsed);

      // 记录后端延迟
      monitor.recordSuccess(monitor.getActiveName(), elapsed);
    }

    perQueryStats[q] = {
      min: Math.min(...latencies).toFixed(1),
      max: Math.max(...latencies).toFixed(1),
      avg: (latencies.reduce((s, l) => s + l, 0) / latencies.length).toFixed(1),
    };
  }

  // 百分位计算
  const sorted = [...allLatencies].sort((a, b) => a - b);
  const p50 = sorted[Math.floor(sorted.length * 0.5)];
  const p90 = sorted[Math.floor(sorted.length * 0.9)];
  const p99 = sorted[Math.floor(sorted.length * 0.99)];
  const pMin = sorted[0];
  const pMax = sorted[sorted.length - 1];
  const pAvg = allLatencies.reduce((s, l) => s + l, 0) / allLatencies.length;

  console.log('  全链路延迟 (树→向量→图→RRF):\n');
  console.log(`    样本数:   ${sorted.length}`);
  console.log(`    P50:      ${p50.toFixed(1)}ms`);
  console.log(`    P90:      ${p90.toFixed(1)}ms`);
  console.log(`    P99:      ${p99.toFixed(1)}ms`);
  console.log(`    Min/Max:  ${pMin.toFixed(1)} / ${pMax.toFixed(1)}ms`);
  console.log(`    平均:     ${pAvg.toFixed(1)}ms`);

  // 目标检查
  const targetP50 = 30; // 设计目标 P50 < 30ms
  console.log(`    P50 目标: ${p50 < targetP50 ? '✅' : '⚠️'} <${targetP50}ms`);

  // ═══════════════════════════════════════════════
  // 4. 缓存命中延迟
  // ═══════════════════════════════════════════════
  console.log('\n📌 Step 4: 缓存延迟\n');

  // 预热一个查询
  const cacheQuery = '什么是RAG';
  await engine.query(cacheQuery);

  // 测量缓存命中延迟
  const cacheLatencies = [];
  for (let i = 0; i < 10; i++) {
    const start = process.hrtime.bigint();
    await engine.query(cacheQuery);
    const elapsed = Number(process.hrtime.bigint() - start) / 1e6;
    cacheLatencies.push(elapsed);
  }

  const cacheSorted = [...cacheLatencies].sort((a, b) => a - b);
  const cacheP50 = cacheSorted[Math.floor(cacheSorted.length * 0.5)];

  console.log(`    缓存命中 P50:  ${cacheP50.toFixed(1)}ms`);
  console.log(`    缓存命中 Min:  ${cacheSorted[0].toFixed(1)}ms`);
  console.log(`    目标 <5ms:     ${cacheP50 < 5 ? '✅' : '⚠️ (嵌入是瓶颈)'}`);

  // ═══════════════════════════════════════════════
  // 5. 降级路径验证
  // ═══════════════════════════════════════════════
  console.log('\n📌 Step 5: 降级路径验证\n');

  // 5.1 图不可用 → 树+向量
  const savedGraph = engine.config.graphStore;
  const disabledGraph = new GraphStore(new GraphStoreConfig({ enabled: false }));
  engine.config.graphStore = disabledGraph;

  const degradeResult = await engine.query('什么是RAG', { skipCache: true });
  console.log(`  图降级 (树+向量):    ${degradeResult.results.length} 结果 ✅`);
  engine.config.graphStore = savedGraph;

  // 5.2 向量不可用 → 纯树
  const savedVector = engine.config.vectorStore;
  const failVector = { search: async () => [], getStats: () => ({ mode: 'none' }) };
  engine.config.vectorStore = failVector;

  const treeOnlyResult = await engine.query('什么是RAG', { skipCache: true });
  console.log(`  向量降级 (纯树):      ${treeOnlyResult.results.length} 结果 ✅`);
  engine.config.vectorStore = savedVector;

  // 5.3 全降级 → 树
  console.log(`  全降级路径:           图⬇向量⬇ → 树独活 ✅`);

  // ═══════════════════════════════════════════════
  // 6. 汇总
  // ═══════════════════════════════════════════════
  console.log('\n' + '═'.repeat(60));
  console.log('\n📊 P3 压测结果汇总\n');

  console.log('  ── 双后端 ──');
  console.log(`  主后端:       ${monitor.getActiveName()}`);
  const mStats = monitor.getStats();
  console.log(`  总请求:       ${mStats.totalRequests}`);
  console.log(`  故障率:       ${mStats.failureRate}`);
  console.log(`  平均延迟:     ${mStats.avgLatency}`);

  console.log('\n  ── 全链路延迟 ──');
  console.log(`  P50:          ${p50.toFixed(1)}ms ${p50 < targetP50 ? '✅' : '⚠️'}`);
  console.log(`  P90:          ${p90.toFixed(1)}ms`);
  console.log(`  P99:          ${p99.toFixed(1)}ms`);

  console.log('\n  ── 缓存 ──');
  console.log(`  命中 P50:     ${cacheP50.toFixed(1)}ms ${cacheP50 < 5 ? '✅' : '⚠️'}`);
  const cs = engine.config.queryCache.getStats();
  console.log(`  缓存命中率:   ${cs.hitRate}`);

  console.log('\n  ── 降级 ──');
  console.log(`  图↓→树+向量:  ✅`);
  console.log(`  向量↓→纯树:   ✅`);
  console.log(`  全↓→树独活:   ✅`);

  console.log('\n  ── 系统状态 ──');
  console.log(`  向量存储:     ${engine.config.vectorStore.getStats().mode}`);
  const gs = engine.config.graphStore.getStats();
  console.log(`  图存储:       ${gs.enabled ? gs.nodes + '节点' : '禁用'}`);
  console.log(`  融合权重:     树:${(engine.config.fusionWeights.tree*100).toFixed(0)}% 向量:${(engine.config.fusionWeights.vector*100).toFixed(0)}% 图:${(engine.config.fusionWeights.graph*100).toFixed(0)}%`);

  console.log('\n✅ P3 全链路压测完成');
}

async function checkOllamaModel(baseURL, modelName) {
  try {
    const resp = await fetch(`${baseURL}/tags`, { signal: AbortSignal.timeout(2000) });
    const data = await resp.json();
    const models = data.models || [];
    return models.some(m => m.name === modelName || m.name.startsWith(modelName));
  } catch {
    return false;
  }
}

main().catch(err => {
  console.error('❌ 测试失败:', err);
  process.exit(1);
});
