/**
 * 简单测试脚本
 * 验证 PageIndex-CN v2.0 基本功能
 */

import { EntityExtractor } from '../src/core/entity_extractor.mjs';
import { CommunityDetector } from '../src/core/community_detector.mjs';
import { createEntityRelationGraph, detectCommunities } from '../src/utils/graph_utils.mjs';
import { LLMClient } from '../src/utils/llm_client.mjs';

async function testEntityExtractor() {
  console.log('\n📝 测试 1: 实体关系提取器\n');

  const llm = new LLMClient();
  const extractor = new EntityExtractor(llm);

  const testText = `
张三是谷歌的软件工程师，专门从事人工智能研究。
他开发了一个基于深度学习的自然语言处理系统。
这个系统在2023年获得了最佳论文奖。
李四是张三的同事，他在微软工作。
`;

  console.log('测试文本:', testText.trim());

  try {
    const result = await extractor.extract(testText);

    console.log('\n✅ 提取成功！');
    console.log(`\n实体 (${result.entities.length} 个):`);
    result.entities.forEach((e, i) => {
      console.log(`  ${i + 1}. ${e.name} (${e.type}): ${e.description}`);
    });

    console.log(`\n关系 (${result.relationships.length} 个):`);
    result.relationships.forEach((r, i) => {
      console.log(
        `  ${i + 1}. ${r.source} ${r.type} ${r.target}: ${r.description}`
      );
    });
  } catch (error) {
    console.error('❌ 提取失败:', error.message);
  }
}

async function testCommunityDetection() {
  console.log('\n📝 测试 2: 社区检测\n');

  const entities = [
    { name: '张三', type: '人名', description: '软件工程师' },
    { name: '李四', type: '人名', description: '软件工程师' },
    { name: '谷歌', type: '组织', description: '科技公司' },
    { name: '微软', type: '组织', description: '科技公司' },
    { name: '人工智能', type: '技术', description: 'AI 技术' },
    { name: '深度学习', type: '技术', description: '机器学习方法' },
    { name: '自然语言处理', type: '技术', description: 'NLP 技术' },
  ];

  const relationships = [
    {
      source: '张三',
      target: '谷歌',
      type: '属于',
      description: '在谷歌工作',
    },
    {
      source: '李四',
      target: '微软',
      type: '属于',
      description: '在微软工作',
    },
    {
      source: '张三',
      target: '人工智能',
      type: '研究',
      description: '研究 AI',
    },
    {
      source: '张三',
      target: '深度学习',
      type: '使用',
      description: '使用深度学习',
    },
    {
      source: '张三',
      target: '自然语言处理',
      type: '开发',
      description: '开发 NLP 系统',
    },
    {
      source: '李四',
      target: '张三',
      type: '同事',
      description: '同事关系',
    },
  ];

  console.log(`实体 (${entities.length} 个):`, entities.map(e => e.name));
  console.log(`关系 (${relationships.length} 个):`);

  // 构建图
  const graph = createEntityRelationGraph(entities, relationships);

  console.log(`图节点数: ${graph.order}`);
  console.log(`图边数: ${graph.size}`);

  // 检测社区
  const communities = detectCommunities(graph);

  console.log(`\n检测到 ${communities.length} 个社区:`);

  communities.forEach((community, i) => {
    console.log(`\n社区 ${i + 1} (${community.id}):`);
    console.log(`  节点: ${community.nodes.join(', ')}`);
  });

  // 使用社区检测器
  const detector = new CommunityDetector();
  const hierarchical = detector.detectFromEntities(entities, relationships);

  console.log('\n层次化社区:');
  console.log(JSON.stringify(hierarchical, null, 2));
}

async function testLLMClient() {
  console.log('\n📝 测试 3: LLM 客户端\n');

  const llm = new LLMClient();

  // 健康检查
  const isHealthy = await llm.healthCheck();
  console.log('LLM 服务状态:', isHealthy ? '✅ 正常' : '❌ 不可用');

  if (!isHealthy) {
    console.log('\n⚠️  请确保 LM Studio 运行在 http://localhost:1234');
    return;
  }

  // 简单测试
  try {
    const response = await llm.chat('你好！请简单介绍一下你自己。', {
      maxTokens: 100,
    });

    console.log('\nLLM 响应:', response);
  } catch (error) {
    console.error('❌ LLM 调用失败:', error.message);
  }
}

async function runAllTests() {
  console.log('='.repeat(60));
  console.log('PageIndex-CN v2.0 - 功能测试');
  console.log('='.repeat(60));

  // 测试 1: LLM 客户端
  await testLLMClient();

  // 测试 2: 实体关系提取器（需要 LLM）
  try {
    await testEntityExtractor();
  } catch (error) {
    console.error('⚠️  跳过实体提取测试（LLM 不可用）');
  }

  // 测试 3: 社区检测（不需要 LLM）
  await testCommunityDetection();

  console.log('\n' + '='.repeat(60));
  console.log('测试完成！');
  console.log('='.repeat(60));
}

// 运行测试
runAllTests();