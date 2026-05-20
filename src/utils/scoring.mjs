/**
 * 评分算法
 * 混合评分：标题匹配 + 实体关系 + 社区相关
 * 支持中英文混合查询（中文用bigram，英文用空格分词）
 */

/**
 * 智能分词：中文用 bigram，英文用空格
 */
function tokenize(text) {
  if (!text) return [];
  // 检测是否主要为中文
  const chineseChars = (text.match(/[\u4e00-\u9fff]/g) || []).length;
  const totalChars = text.replace(/\s/g, '').length;

  if (totalChars === 0) return [];

  // 中文为主 → bigram
  if (chineseChars / totalChars > 0.3) {
    const cleaned = text.replace(/[^\u4e00-\u9fff]/g, '');
    const bigrams = [];
    for (let i = 0; i < cleaned.length - 1; i++) {
      bigrams.push(cleaned.slice(i, i + 2));
    }
    // 也加上单字（处理短查询）
    for (const ch of cleaned) {
      bigrams.push(ch);
    }
    return [...new Set(bigrams)];
  }

  // 英文 → 空格分词 + 小写
  return text.toLowerCase().split(/[\s,，。！？、；：""''（）\(\)\[\]{}]+/).filter(t => t.length > 0);
}

/**
 * 标题匹配评分 (保留原始基于空格分词的版本，避免中文 bigram 误匹配)
 */
export function titleMatchScore(title, query) {
  if (!title || !query) return 0;
  const titleLower = title.toLowerCase();
  const queryLower = query.toLowerCase();
  // 中文用字符包含匹配，英文用空格分词
  const hasChinese = /[\u4e00-\u9fff]/.test(query);
  const keywords = hasChinese
    ? [queryLower.replace(/\s/g, '')] // 中文: 整个查询作为一个 token
    : queryLower.split(/\s+/);

  let score = 0, matchedCount = 0;
  for (const keyword of keywords) {
    if (titleLower.includes(keyword)) {
      matchedCount++;
      score += titleLower === keyword ? 1.0 : 0.5;
    }
  }
  if (keywords.length > 0) score /= keywords.length;
  return Math.min(score, 1.0);
}

/**
 * 实体匹配评分
 */
export function entityMatchScore(nodeEntities, query) {
  if (!nodeEntities || nodeEntities.length === 0) return 0;
  const queryLower = query.toLowerCase();
  const keywords = queryLower.split(/\s+/);
  let totalScore = 0;
  for (const entity of nodeEntities) {
    const entityLower = entity.name.toLowerCase();
    for (const keyword of keywords) {
      if (entityLower === keyword) totalScore += 1.0;
      else if (entityLower.includes(keyword) || keyword.includes(entityLower)) totalScore += 0.5;
      else if (entity.description && entity.description.toLowerCase().includes(keyword)) totalScore += 0.3;
    }
  }
  const maxScore = nodeEntities.length * keywords.length;
  if (maxScore > 0) totalScore /= maxScore;
  return Math.min(totalScore, 1.0);
}

/**
 * 社区相关性评分
 */
export function communityRelevanceScore(
  communityId,
  query,
  communitySummaries
) {
  if (!communitySummaries || communitySummaries.length === 0) {
    return 0;
  }

  // 找到对应的社区摘要
  const community = communitySummaries.find(c => c.id === communityId);
  if (!community) {
    return 0;
  }

  const queryLower = query.toLowerCase();
  const keywords = queryLower.split(/\s+/);

  let score = 0;

  // 摘要匹配
  if (community.summary) {
    const summaryLower = community.summary.toLowerCase();

    for (const keyword of keywords) {
      if (summaryLower.includes(keyword)) {
        score += 0.5;
      }
    }

    // 归一化
    if (keywords.length > 0) {
      score = score / keywords.length;
    }
  }

  return Math.min(score, 1.0);
}

/**
 * 混合评分
 * 权重：标题 40% + 实体 30% + 社区 30%
 */
export function hybridScore(node, query, context, options = {}) {
  const weights = {
    title: options.titleWeight || 0.4,
    entity: options.entityWeight || 0.3,
    community: options.communityWeight || 0.3,
    ...options.weights,
  };

  const titleScore = titleMatchScore(node.title, query);
  const entityScore = entityMatchScore(node.entities || [], query);
  const communityScore = communityRelevanceScore(
    node.communityId,
    query,
    context.communitySummaries || []
  );

  return (
    weights.title * titleScore +
    weights.entity * entityScore +
    weights.community * communityScore
  );
}

/**
 * 内容匹配评分（补充）
 */
export function contentMatchScore(content, query) {
  if (!content || !query) return 0;
  const contentLower = content.toLowerCase();
  const queryLower = query.toLowerCase();
  const keywords = queryLower.split(/\s+/);
  let score = 0, matchedCount = 0;
  for (const keyword of keywords) {
    const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const matches = (contentLower.match(new RegExp(escaped, 'g')) || []).length;
    if (matches > 0) { matchedCount++; score += Math.min(matches * 0.2, 0.8); }
  }
  if (keywords.length > 0) score /= keywords.length;
  return Math.min(score, 1.0);
}