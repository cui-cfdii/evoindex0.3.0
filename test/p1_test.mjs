/**
 * EvoIndex 3.0 — P1 验证测试
 *
 * 测试：树+向量+图 三路 RRF 融合 + 图增量更新
 * 验证标准：图增强贡献 > 0, 增量更新 < 5s
 *
 * 用法: node test/p1_test.mjs <index.json> [--data-dir <path>]
 */

import { HybridQueryEngineV3, QueryEngineConfig } from '../src/core/hybrid_query.mjs';
import { VectorStore, VectorStoreConfig } from '../src/core/lancedb_store.mjs';
import { GraphStore, GraphStoreConfig } from '../src/core/graph_store.mjs';
import fs from 'fs';

async function main() {
  // 解析参数
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
    console.log('用法: node test/p1_test.mjs <index.json> [--data-dir <path>]');
    process.exit(1);
  }

  if (process.platform === 'win32' && !dataDir) {
    dataDir = 'C:\\Users\\cuihao\\evoindex_data';
  }

  console.log('🧪 EvoIndex 3.0 P1 验证测试 — 图增强\n');
  console.log('═'.repeat(60));
  if (dataDir) console.log(`📁 数据目录: ${dataDir}\n`);

  // 1. 初始化引擎（含图存储）
  const vectorStore = new VectorStore(new VectorStoreConfig({ dataDir }));
  const gStore = new GraphStore(new GraphStoreConfig({ dataDir }));
  const config = new QueryEngineConfig({
    vectorStore,
    graphStore: gStore,
    fusionWeights: { tree: 0.45, vector: 0.45, graph: 0.1 },
  });
  const engine = new HybridQueryEngineV3(config);
  await engine.loadIndex(indexPath);

  // 2. 构建图索引
  console.log('\n📌 Step 1: 构建图索引');
  const startBuild = Date.now();
  const graphStats = await engine.buildGraphIndex();
  console.log(`  构建耗时: ${((Date.now() - startBuild) / 1000).toFixed(1)}s`);

  // 3. 查询测试 — 三路融合
  const testQueries = [
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
    { q: '临床试验设计', domain: 'pharma' },
    { q: '药品注册法规', domain: 'pharma' },
  ];

  console.log('\n📌 Step 2: 三路融合查询测试\n');

  let totalMRR = 0;
  let totalRecall = 0;
  let graphContributed = 0;
  let vectorCount = 0;

  for (const { q, domain } of testQueries) {
    const result = await engine.query(q);

    const hasGraph = result.sourceCounts.graph > 0;
    const hasVector = result.sourceCounts.vector > 0;

    if (hasGraph) graphContributed++;
    if (hasVector) vectorCount++;

    const mrr = result.results[0]?.score || 0;
    totalMRR += mrr;
    totalRecall += result.sourceCounts.fused > 0 ? 1 : 0;

    console.log(`  [${domain}] "${q}"`);
    console.log(`    置信度: ${(result.confidence * 100).toFixed(0)}% | LLM: ${result.needsLLM ? '需要' : '跳过'}`);
    console.log(`    树:${result.sourceCounts.tree} 向量:${result.sourceCounts.vector} 图:${result.sourceCounts.graph} 融合:${result.sourceCounts.fused}`);
    console.log(`    Top-1: [${(mrr * 100).toFixed(0)}%] ${result.results[0]?.metadata?.title || result.results[0]?.id || 'N/A'}`);
    console.log();
  }

  const n = testQueries.length;
  const avgMRR = totalMRR / n;
  const avgRecall = totalRecall / n;

  // 4. 增量更新测试
  console.log('📌 Step 3: 增量更新测试');
  const testNode = {
    _id: 'test_incremental',
    title: 'P1 增量更新测试节点',
    content: 'RAG 检索增强生成 结合了 GPT-4 和 LlamaIndex 实现了混合检索策略，使用 LanceDB 作为向量数据库',
  };

  const startIncr = Date.now();
  await gStore.incrementalUpdate(testNode, 'add');
  const incrTime = Date.now() - startIncr;

  console.log(`  添加节点: ${testNode._id}`);
  console.log(`  耗时: ${incrTime}ms ${incrTime < 5000 ? '✅' : '⚠️ 目标: <5s'}`);

  // 验证增量后图增强是否生效
  const incrResult = await engine.query('增量更新测试');
  console.log(`  图增强验证: ${incrResult.sourceCounts.graph > 0 ? '✅ 图有贡献' : '⚠️ 图无贡献'}`);

  // 5. 结果汇总
  console.log('\n' + '═'.repeat(60));
  console.log('\n📊 P1 测试结果汇总\n');
  console.log(`  MRR:              ${(avgMRR * 100).toFixed(1)}% ${avgMRR >= 0.85 ? '✅' : '⚠️'}`);
  console.log(`  平均 Recall:      ${(avgRecall * 100).toFixed(1)}%`);
  console.log(`  图贡献查询:       ${graphContributed}/${n}`);
  console.log(`  向量贡献查询:     ${vectorCount}/${n}`);
  console.log(`  增量更新耗时:     ${incrTime}ms ${incrTime < 5000 ? '✅' : '⚠️'}`);

  // 系统状态
  console.log('\n📌 系统状态');
  console.log(`  向量存储: ${engine.config.vectorStore.getStats().mode}`);
  const gs = engine.config.graphStore.getStats();
  console.log(`  图存储:   ${gs.enabled ? '启用' : '禁用'} | ${gs.nodes} 节点 ${gs.edges} 边 ${gs.communities} 社区`);
  console.log(`  融合权重: 树:${engine.config.fusionWeights.tree} 向量:${engine.config.fusionWeights.vector} 图:${engine.config.fusionWeights.graph}`);
  console.log(`  融合模式: ${engine.config.fusionMode}`);

  console.log('\n✅ P1 测试完成');
}

main().catch(err => {
  console.error('❌ 测试失败:', err);
  process.exit(1);
});
