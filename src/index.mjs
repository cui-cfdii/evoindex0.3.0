#!/usr/bin/env node
/**
 * EvoIndex 3.0 — Learned Hybrid Index for Knowledge Retrieval
 *
 * 主入口：树索引 + 向量语义 + 图增强 + 自进化
 *
 * 用法:
 *   node src/index.mjs status                              系统状态
 *   node src/index.mjs vectorize <index.json>              构建向量索引
 *   node src/index.mjs graph <index.json>                  构建图索引
 *   node src/index.mjs query "<query>" <index.json>        三路混合查询
 *   node src/index.mjs bench <index.json>                  全链路压测
 *   node src/index.mjs tune <index.json>                   CMA-ES 权重调优
 */

import fs from 'fs';
import { HybridQueryEngineV3, QueryEngineConfig } from './core/hybrid_query.mjs';
import { VectorStore, VectorStoreConfig } from './core/lancedb_store.mjs';
import { GraphStore, GraphStoreConfig } from './core/graph_store.mjs';
import { QueryCache } from './core/query_cache.mjs';
import { EmbeddingClient } from './utils/embedding_client.mjs';
import { CMAESTuner } from './core/cmaes_tuner.mjs';

function usage() {
  console.log(`
EvoIndex 3.0 — Learned Hybrid Index for Knowledge Retrieval
─────────────────────────────────────────────────────────
用法:
  node src/index.mjs status                                 系统状态
  node src/index.mjs vectorize <index.json> [--data-dir]    构建向量索引
  node src/index.mjs graph <index.json> [--data-dir]        构建图索引
  node src/index.mjs query "<query>" <index.json> [--data-dir] 混合查询
  node src/index.mjs bench <index.json> [--data-dir]        全链路压测
  node src/index.mjs tune <index.json> [--data-dir]         CMA-ES 权重调优
  node src/index.mjs cache <index.json> --query "<q>"       缓存查询

选项:
  --data-dir <path>    数据目录 (Windows 默认 C:\\Users\\cuihao\\evoindex_data)
  --weights <t:v:g>    融合权重 (如 0.08:0.63:0.29)
  --backend <name>     强制后端 (lmstudio|ollama)
`);
}

function parseArgs(args) {
  const opts = { dataDir: null, weights: null, backend: null, query: null };
  const positional = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--data-dir' && i + 1 < args.length) opts.dataDir = args[++i];
    else if (args[i] === '--weights' && i + 1 < args.length) opts.weights = args[++i];
    else if (args[i] === '--backend' && i + 1 < args.length) opts.backend = args[++i];
    else if (args[i] === '--query' && i + 1 < args.length) opts.query = args[++i];
    else positional.push(args[i]);
  }
  // Windows 默认数据目录
  if (!opts.dataDir && process.platform === 'win32') {
    opts.dataDir = 'C:\\Users\\cuihao\\evoindex_data';
  }
  return { opts, positional };
}

function parseWeights(str) {
  const parts = str.split(':').map(Number);
  if (parts.length !== 3 || parts.some(isNaN)) return null;
  const sum = parts.reduce((s, v) => s + v, 0);
  return { tree: parts[0]/sum, vector: parts[1]/sum, graph: parts[2]/sum };
}

function createEngine(opts) {
  const config = {
    vectorStore: new VectorStore(new VectorStoreConfig({ dataDir: opts.dataDir })),
    graphStore: new GraphStore(new GraphStoreConfig({ dataDir: opts.dataDir })),
    queryCache: new QueryCache(),
  };
  if (opts.weights) {
    const w = parseWeights(opts.weights);
    if (w) config.fusionWeights = w;
  }
  return new HybridQueryEngineV3(new QueryEngineConfig(config));
}

// ─── 命令实现 ──────────────────────────────

async function cmdStatus(opts) {
  console.log('\n📊 EvoIndex 3.0 系统状态\n');
  console.log('═'.repeat(40));

  // 嵌入后端
  const embed = new EmbeddingClient({ backend: opts.backend || 'auto' });
  try {
    const backend = await embed.detectBackend();
    const dim = await embed.getDimension();
    console.log(`  嵌入后端: ${backend}`);
    console.log(`  嵌入模型: ${embed.config.model}`);
    console.log(`  向量维度: ${dim}`);
  } catch (err) {
    console.log(`  嵌入后端: ❌ (${err.message})`);
  }
}

async function cmdVectorize(indexPath, opts) {
  if (!fs.existsSync(indexPath)) {
    console.error(`❌ 索引不存在: ${indexPath}`);
    process.exit(1);
  }

  const engine = createEngine(opts);
  await engine.loadIndex(indexPath);
  console.log('🔨 构建向量索引...\n');
  await engine.buildVectorIndex({ delay: 100 });
  console.log('✅ 完成');
}

async function cmdGraph(indexPath, opts) {
  if (!fs.existsSync(indexPath)) {
    console.error(`❌ 索引不存在: ${indexPath}`);
    process.exit(1);
  }

  const engine = createEngine(opts);
  await engine.loadIndex(indexPath);
  const stats = await engine.buildGraphIndex();
  console.log(`✅ 图构建完成: ${stats.nodes} 节点, ${stats.edges} 边, ${stats.communities} 社区`);
}

async function cmdQuery(query, indexPath, opts) {
  if (!fs.existsSync(indexPath)) {
    console.error(`❌ 索引不存在: ${indexPath}`);
    process.exit(1);
  }

  const engine = createEngine(opts);
  await engine.loadIndex(indexPath);

  // 如果图数据存在，加载
  if (engine.config.graphStore.config.enabled) {
    await engine.buildGraphIndex().catch(() => {});
  }

  const result = await engine.query(query);

  console.log(`\n📊 检索完成 | 置信度: ${(result.confidence * 100).toFixed(1)}% | ${result.fromCache ? '⚡缓存' : '🔍全链路'}`);
  console.log(`   树:${result.sourceCounts.tree} 向量:${result.sourceCounts.vector} 图:${result.sourceCounts.graph} 融合:${result.sourceCounts.fused}`);

  if (result.results.length === 0) {
    console.log('   未找到匹配结果');
    return;
  }

  console.log(`\nTop-${Math.min(5, result.results.length)} 结果:\n`);
  for (let i = 0; i < Math.min(5, result.results.length); i++) {
    const r = result.results[i];
    const sources = r.sourceScores
      ? Object.entries(r.sourceScores).map(([k, v]) => `${k}:${(v*100).toFixed(0)}%`).join(' ')
      : '';
    console.log(`${i + 1}. [${(r.score * 100).toFixed(1)}%] ${r.metadata?.title || r.id}`);
    if (sources) console.log(`   来源: ${sources} | ${r.hits}路命中`);
  }
}

async function cmdBench(indexPath, opts) {
  if (!fs.existsSync(indexPath)) {
    console.error(`❌ 索引不存在: ${indexPath}`);
    process.exit(1);
  }

  const engine = createEngine(opts);
  await engine.loadIndex(indexPath);
  await engine.buildGraphIndex().catch(() => {});

  const queries = [
    '什么是RAG', '检索增强生成', '大模型微调方法', '向量数据库选型',
    'GraphRAG工作原理', '知识图谱构建', '嵌入模型对比', '混合检索策略',
    'FDA审批流程', 'GLP-1受体激动剂',
  ];

  const latencies = [];
  console.log('🔬 全链路压测中...\n');

  for (const q of queries) {
    for (let i = 0; i < 3; i++) {
      const start = process.hrtime.bigint();
      await engine.query(q, { skipCache: true });
      latencies.push(Number(process.hrtime.bigint() - start) / 1e6);
    }
  }

  const sorted = [...latencies].sort((a, b) => a - b);
  console.log(`  样本: ${sorted.length}`);
  console.log(`  P50:  ${sorted[Math.floor(sorted.length * 0.5)].toFixed(1)}ms`);
  console.log(`  P90:  ${sorted[Math.floor(sorted.length * 0.9)].toFixed(1)}ms`);
  console.log(`  P99:  ${sorted[Math.floor(sorted.length * 0.99)].toFixed(1)}ms`);
  console.log(`  Avg:  ${(latencies.reduce((s,l) => s+l, 0) / latencies.length).toFixed(1)}ms`);
}

async function cmdTune(indexPath, opts) {
  if (!fs.existsSync(indexPath)) {
    console.error(`❌ 索引不存在: ${indexPath}`);
    process.exit(1);
  }

  const engine = createEngine(opts);
  await engine.loadIndex(indexPath);
  await engine.buildGraphIndex().catch(() => {});

  const trainQueries = [
    '什么是RAG', '检索增强生成', '大模型微调方法', '向量数据库选型',
    'GraphRAG工作原理', '知识图谱构建', '嵌入模型对比', '混合检索策略',
    'FDA审批流程', 'GLP-1受体激动剂',
  ];

  const tuner = new CMAESTuner(3, {
    lambda: 8, sigma: 0.3,
    mean: [0.33, 0.33, 0.34],
    bounds: { min: 0.05, max: 0.9 },
  });

  console.log('🧬 CMA-ES 权重自进化 (8 代)...\n');

  for (let gen = 0; gen < 8; gen++) {
    const population = tuner.sample();
    const scores = [];

    for (const w of population) {
      engine.config.fusionWeights = { tree: w[0], vector: w[1], graph: w[2] };
      engine.config.queryCache.clear();

      let mrr = 0;
      for (const { q } of trainQueries) {
        const r = await engine.query(q, { skipCache: true });
        mrr += r.results[0]?.score || 0;
      }
      scores.push(mrr / trainQueries.length);
    }

    const status = tuner.update(population, scores);
    console.log(`  Gen ${gen+1}: MRR ${(status.bestScore*100).toFixed(1)}% | 权重 ${tuner.bestParams.map(v => (v*100).toFixed(0)).join(':')}`);
    await new Promise(r => setTimeout(r, 200));
  }

  const best = tuner.getBestWeights();
  console.log(`\n✅ 最优权重: 树:${(best.tree*100).toFixed(0)}% 向量:${(best.vector*100).toFixed(0)}% 图:${(best.graph*100).toFixed(0)}%`);
}

// ─── 入口 ──────────────────────────────────

async function main() {
  const rawArgs = process.argv.slice(2);
  const { opts, positional } = parseArgs(rawArgs);
  const cmd = positional[0];

  if (!cmd || cmd === 'help') {
    usage();
    process.exit(cmd ? 0 : 1);
  }

  try {
    switch (cmd) {
      case 'status':
        await cmdStatus(opts);
        break;
      case 'vectorize':
        await cmdVectorize(positional[1], opts);
        break;
      case 'graph':
        await cmdGraph(positional[1], opts);
        break;
      case 'query':
        await cmdQuery(positional[1], positional[2], opts);
        break;
      case 'bench':
        await cmdBench(positional[1], opts);
        break;
      case 'tune':
        await cmdTune(positional[1], opts);
        break;
      case 'cache':
        {
          const engine = createEngine(opts);
          await engine.loadIndex(positional[1]);
          const r = await engine.query(opts.query || 'test');
          console.log(`${r.fromCache ? '⚡ 命中' : '🔍 未命中'}`);
        }
        break;
      default:
        console.error(`未知命令: ${cmd}`);
        usage();
        process.exit(1);
    }
  } catch (err) {
    console.error(`❌ 错误: ${err.message}`);
    process.exit(1);
  }
}

main();
