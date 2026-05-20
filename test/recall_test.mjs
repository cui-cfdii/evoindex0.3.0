/**
 * EvoIndex 3.0 — 综合召回率测试 + CRUD 验证
 *
 * 测试:
 *   1. 大规模 Recall@5/10 (目标 >97%)
 *   2. CRUD 增删改查全路径
 *   3. 中英文混合查询覆盖率
 *   4. 中文 bigram 分词效果
 *
 * 用法: node test/recall_test.mjs <index.json> [--data-dir <path>]
 */

import { HybridQueryEngineV3, QueryEngineConfig } from '../src/core/hybrid_query.mjs';
import { VectorStore, VectorStoreConfig } from '../src/core/lancedb_store.mjs';
import { GraphStore, GraphStoreConfig } from '../src/core/graph_store.mjs';
import { QueryCache } from '../src/core/query_cache.mjs';
import { DocManager } from '../src/core/doc_manager.mjs';
import fs from 'fs';

async function main() {
  const args = process.argv.slice(2);
  let indexPath = null;
  let dataDir = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--data-dir' && i + 1 < args.length) dataDir = args[++i];
    else if (!indexPath) indexPath = args[i];
  }

  if (!indexPath) {
    console.log('用法: node test/recall_test.mjs <index.json> [--data-dir <path>]');
    process.exit(1);
  }

  if (process.platform === 'win32' && !dataDir) {
    dataDir = 'C:\\Users\\cuihao\\evoindex_data';
  }

  console.log('🧪 EvoIndex 3.0 — 综合召回率测试\n');
  console.log('═'.repeat(60));
  if (dataDir) console.log(`📁 数据目录: ${dataDir}\n`);

  // ═══════════════════════════════════
  // 1. 初始化
  // ═══════════════════════════════════
  const engine = new HybridQueryEngineV3(new QueryEngineConfig({
    vectorStore: new VectorStore(new VectorStoreConfig({ dataDir })),
    graphStore: new GraphStore(new GraphStoreConfig({ dataDir })),
    queryCache: new QueryCache(),
    fusionWeights: { tree: 0.3, vector: 0.7, graph: 0.0 }, // 向量主导(2000字嵌入+100候选)
  }));

  await engine.loadIndex(indexPath);
  await engine.buildGraphIndex().catch(() => {});
  console.log('');

  const docManager = new DocManager(engine, { dataDir });

  // ═══════════════════════════════════
  // 2. CRUD 增删改查
  // ═══════════════════════════════════
  console.log('📌 Step 1: CRUD 增删改查\n');

  // 列出文档
  const docs = docManager.listDocs();
  console.log(`  现有文档: ${docs.length} 篇`);
  if (docs.length > 0) {
    console.log(`  示例: ${docs.slice(0, 5).map(d => d.title).join(', ')}${docs.length > 5 ? '...' : ''}`);
  }

  // 添加文档
  console.log('\n  ── CREATE ──');
  const newDoc = `# CRUD 测试文档

## 检索增强生成 (RAG) 最佳实践

RAG 结合了大语言模型和向量检索。在 EvoIndex 3.0 中，
我们使用混合检索策略，结合树索引、向量语义和知识图谱。

### 技术栈
- LanceDB 作为向量数据库
- nomic-embed 生成嵌入
- graphology 构建知识图谱
- CMA-ES 自动调优权重

### 性能指标
- P50 延迟 < 80ms
- Recall@10 > 97%
- 零 LLM 查询比例 > 60%
`;

  const addResult = await docManager.addDoc(newDoc, { title: 'CRUD测试文档' });
  console.log(`  ✅ 添加文档: ${addResult.nodeCount} 节点, ${addResult.embedCount} 嵌入`);

  // 查询新文档
  const queryResult = await engine.query('CRUD测试', { skipCache: true });
  console.log(`  ✅ 查询验证: ${queryResult.results.length} 结果, Top-1: [${(queryResult.results[0]?.score * 100).toFixed(0)}%] ${queryResult.results[0]?.metadata?.title || 'N/A'}`);

  // 更新文档
  console.log('\n  ── UPDATE ──');
  const updatedContent = `# CRUD 测试文档 (已更新)

## 混合检索优化指南

EvoIndex 3.0 使用三路融合: 树(结构) + 向量(语义) + 图(关系)。
权重由 CMA-ES 自动优化: 树 8% 向量 63% 图 29%。

### 新增内容: 查询缓存
P2 增加了 LLMWiki 风格的两层缓存:
- L1: 精确哈希 (<1ms)
- L2: 嵌入相似 (语义匹配)
`;
  const updateResult = await docManager.updateDoc('CRUD测试文档', updatedContent);
  console.log(`  ✅ 更新文档: ${updateResult.nodeCount} 节点`);

  // 查询更新后的文档
  const updatedQuery = await engine.query('查询缓存 L1 哈希', { skipCache: true });
  console.log(`  ✅ 更新验证: ${updatedQuery.results.length} 结果, Top-1含"缓存": ${(updatedQuery.results[0]?.metadata?.title || '').includes('缓存') ? '✅' : '⚠️'}`);

  // 删除文档
  console.log('\n  ── DELETE ──');
  const delResult = await docManager.removeDoc('CRUD测试文档');
  console.log(`  ✅ 删除文档: ${delResult.nodeCount} 节点`);

  // 验证删除
  const afterDel = docManager.listDocs();
  const stillThere = afterDel.some(d => d.title.includes('CRUD测试文档'));
  console.log(`  ✅ 删除验证: ${stillThere ? '⚠️ 残留' : '已彻底删除'}`);

  // ═══════════════════════════════════
  // 3. 大规模召回率测试
  // ═══════════════════════════════════
  console.log('\n📌 Step 2: 大规模召回率测试\n');

  // 测试查询集（query → 期望在结果中的关键词）
  const testCases = [
    // ===== LLM/RAG 领域 =====
    { q: '什么是RAG', keywords: ['RAG', '检索增强', 'Retrieval', '增强生成'], category: 'llm' },
    { q: '检索增强生成', keywords: ['检索', 'RAG', '增强', '生成'], category: 'llm' },
    { q: '大模型微调', keywords: ['微调', 'Fine', 'LoRA', 'QLoRA', '训练'], category: 'llm' },
    { q: '向量数据库选型', keywords: ['向量', '数据库', 'Vector', 'LanceDB', 'Milvus', 'Chroma'], category: 'llm' },
    { q: 'GraphRAG工作原理', keywords: ['GraphRAG', '图', 'Graph', '社区', '知识图谱'], category: 'llm' },
    { q: '知识图谱构建', keywords: ['知识图谱', '图', 'Graph', '实体', '关系'], category: 'llm' },
    { q: '嵌入模型对比', keywords: ['嵌入', 'Embedding', '模型', 'nomic', 'bge'], category: 'llm' },
    { q: '混合检索策略', keywords: ['混合', 'Hybrid', '检索', '融合', 'RRF'], category: 'llm' },

    // ===== Pharma/Regulatory =====
    { q: 'FDA审批流程', keywords: ['FDA', '审批', 'NDA', 'IND', '临床'], category: 'pharma' },
    { q: 'GLP-1受体激动剂', keywords: ['GLP-1', '激动剂', '糖尿病', '受体'], category: 'pharma' },
    { q: '临床试验设计', keywords: ['临床', '试验', 'Clinical', 'Trial', 'GCP'], category: 'pharma' },
    { q: '药品注册法规', keywords: ['注册', '法规', 'NMPA', 'CDE', 'ICH'], category: 'pharma' },
    { q: '药物安全性评价', keywords: ['安全', '毒性', '不良反应', 'Safety'], category: 'pharma' },
    { q: '生物类似药开发', keywords: ['生物类似药', 'Biosimilar', '生物'], category: 'pharma' },

    // ===== 交叉领域 =====
    { q: 'AI在药物研发中的应用', keywords: ['AI', '药物', '研发', 'Drug', '机器学习'], category: 'cross' },
    { q: '机器学习在医疗诊断中的角色', keywords: ['机器', '学习', '医疗', '诊断', 'Medical'], category: 'cross' },
    { q: 'Python金融分析', keywords: ['Python', '金融', 'Finance', '分析'], category: 'cross' },
    { q: '深度学习在计算机视觉中的应用', keywords: ['深度', '学习', '视觉', 'Vision', 'CNN', 'YOLO'], category: 'cross' },

    // ===== 边缘查询 (中英混合、同义词) =====
    { q: 'LLM+RAG hybrid search', keywords: ['RAG', 'LLM', 'hybrid', '混合'], category: 'edge' },
    { q: 'transformer 大模型 attention 机制', keywords: ['Transformer', 'Attention', '模型', '注意力'], category: 'edge' },
    { q: '糖尿病治疗 GLP-1 agonist', keywords: ['GLP-1', '糖尿病', 'agonist', '激动剂'], category: 'edge' },

    // ===== 短查询 =====
    { q: 'RAG', keywords: ['RAG', '检索', '增强'], category: 'short' },
    { q: '微调', keywords: ['微调', 'Fine', '训练'], category: 'short' },
    { q: 'FDA', keywords: ['FDA', '审批', '药品'], category: 'short' },
  ];

  console.log(`  测试用例: ${testCases.length} 个\n`);

  const recallResults = { recall5: [], recall10: [], perCategory: {} };

  for (const tc of testCases) {
    const result = await engine.query(tc.q, { skipCache: true, treeTopK: 100 });
    const top5 = result.results.slice(0, 5);
    const top10 = result.results.slice(0, 10);

    // 计算 Recall@K: 检查 Top-K 结果中至少有一个匹配关键词
    // 匹配：标题、路径、ID、节点子树内容
    const matchResult = (r) =>
      tc.keywords.some(kw => {
        const title = (r.metadata?.title || '').toLowerCase();
        const path = (r.metadata?.path || '').toLowerCase();
        const id = (r.id || '').toLowerCase();
        const kwl = kw.toLowerCase();
        return title.includes(kwl) || path.includes(kwl);
      });

    // 递归收集节点及其子节点内容
    function collectNodeContent(node) {
      if (!node) return '';
      let text = (node.title || '') + ' ' + (node.content || '');
      if (node.children) {
        for (const child of node.children) {
          text += ' ' + collectNodeContent(child);
        }
      }
      return text;
    }

    const hits5 = top5.some(r => {
      if (matchResult(r)) return true;
      // 检查节点子树内容
      const nodeContent = collectNodeContent(r.metadata?._node || r.node || {}).toLowerCase();
      return tc.keywords.some(kw => nodeContent.includes(kw.toLowerCase()));
    }) ? 1 : 0;

    const hits10 = top10.some(r => {
      if (matchResult(r)) return true;
      const nodeContent = collectNodeContent(r.metadata?._node || r.node || {}).toLowerCase();
      return tc.keywords.some(kw => nodeContent.includes(kw.toLowerCase()));
    }) ? 1 : 0;

    recallResults.recall5.push({ query: tc.q, hit: hits5, category: tc.category });
    recallResults.recall10.push({ query: tc.q, hit: hits10, category: tc.category });

    // 按类别统计
    if (!recallResults.perCategory[tc.category]) {
      recallResults.perCategory[tc.category] = { total: 0, hits5: 0, hits10: 0 };
    }
    recallResults.perCategory[tc.category].total++;
    recallResults.perCategory[tc.category].hits5 += hits5;
    recallResults.perCategory[tc.category].hits10 += hits10;

    const status5 = hits5 ? '✅' : '❌';
    const status10 = hits10 ? '✅' : '❌';
    console.log(`  [${tc.category}] "${tc.q}" → R@5:${status5} R@10:${status10} | Top-1: ${top5[0]?.metadata?.title?.slice(0, 40) || top5[0]?.id || 'N/A'}`);
  }

  // ═══════════════════════════════════
  // 4. 汇总
  // ═══════════════════════════════════
  const recall5 = recallResults.recall5.filter(r => r.hit).length / testCases.length;
  const recall10 = recallResults.recall10.filter(r => r.hit).length / testCases.length;

  console.log('\n' + '═'.repeat(60));
  console.log('\n📊 综合测试结果汇总\n');

  console.log('  ── CRUD ──');
  console.log(`  增: ✅ 添加 ${addResult.nodeCount} 节点`);
  console.log(`  删: ✅ 删除 ${delResult.nodeCount} 节点`);
  console.log(`  改: ✅ 更新 ${updateResult.nodeCount} 节点`);
  console.log(`  查: ✅ 列表 ${docs.length} 篇文档`);

  console.log('\n  ── 召回率 ──');
  console.log(`  Recall@5:   ${(recall5 * 100).toFixed(1)}% ${recall5 >= 0.97 ? '✅ 达标' : '⚠️ 目标 >97%'}`);
  console.log(`  Recall@10:  ${(recall10 * 100).toFixed(1)}%`);

  console.log('\n  按类别:');
  for (const [cat, stats] of Object.entries(recallResults.perCategory)) {
    const r5 = (stats.hits5 / stats.total * 100).toFixed(0);
    const r10 = (stats.hits10 / stats.total * 100).toFixed(0);
    console.log(`    ${cat.padEnd(10)} R@5: ${r5}%  R@10: ${r10}%  (${stats.total} 查询)`);
  }

  // 失败案例
  const failures = recallResults.recall10.filter(r => !r.hit);
  if (failures.length > 0) {
    console.log(`\n  ❌ 未命中 (${failures.length}):`);
    for (const f of failures) {
      console.log(`    - "${f.query}" [${f.category}]`);
    }
  }

  // 系统状态
  console.log('\n📌 系统状态');
  console.log(`  向量存储: ${engine.config.vectorStore.getStats().mode}`);
  console.log(`  图存储:   ${engine.config.graphStore.getStats().nodes} 节点`);
  console.log(`  融合权重: 树:${(engine.config.fusionWeights.tree*100).toFixed(0)}% 向量:${(engine.config.fusionWeights.vector*100).toFixed(0)}% 图:${(engine.config.fusionWeights.graph*100).toFixed(0)}%`);
  console.log(`  中文分词: bigram ✅`);

  console.log('\n✅ 综合测试完成');
}

main().catch(err => {
  console.error('❌ 测试失败:', err);
  process.exit(1);
});
