/**
 * EvoIndex 3.0 — 查询意图路由器 (Layer 0)
 *
 * 根据查询关键词路由到对应领域，约束搜索范围。
 * 解决 97% pharma 数据噪声压倒 llm 信号的问题。
 *
 * 路由规则 (优先级递减):
 *   1. 精确关键词匹配 → domain
 *   2. 字符级匹配 (中文) → domain
 *   3. 默认 → all
 */

// 领域关键词映射
const DOMAIN_KEYWORDS = {
  llm: [
    'rag', '检索增强', 'retrieval', 'llm', '大模型', '大语言',
    '微调', 'fine', 'lora', 'qlora', 'rlhf', '训练', '模型',
    '向量', 'vector', 'embed', '嵌入', 'embedding',
    'graphrag', '知识图谱', 'knowledge graph', 'graph',
    'transformer', 'attention', '注意力', 'bert', 'gpt',
    '混合检索', 'hybrid', 'rrf', '融合', 'fuse',
    'lancedb', 'chroma', 'milvus', '向量数据库', 'vector database',
    'langchain', 'llamaindex', 'prompt', '提示词',
    '深度学习', 'deep learning', '神经网络', 'neural',
    'token', 'tokenizer', '分词', 'decoder', 'encoder',
    '语义', 'semantic', 'nlp', '自然语言',
  ],
  pharma: [
    'fda', 'ema', 'nmpa', 'cde', '药品', '药物', 'drug',
    '临床', 'clinical', 'trial', '试验', 'gcp', 'glp', 'gmp',
    '审批', '注册', '法规', 'regulatory', 'nda', 'ind', 'anda',
    'glp-1', '激动剂', 'agonist', '糖尿病', 'diabetes',
    '安全性', 'safety', '不良反应', 'adverse', '毒性', 'toxicity',
    '生物类似药', 'biosimilar', '生物等效', 'bioequivalence',
    '疫苗', 'vaccine', '基因治疗', 'gene therapy',
    '溶出度', 'dissolution', '仿制药', 'generic',
    'ich', '持有人', '上市许可', 'ma',
  ],
  medtech: [
    '诊断', 'diagnosis', '检测', 'detection', '影像', 'imaging',
    'ct', 'mri', '超声', 'ultrasound', 'x光', 'x-ray',
    '肺结节', 'nodule', '病理', 'pathology',
  ],
  fintech: [
    '金融', 'finance', '风控', 'risk', '量化', 'quant',
    'python', 'stock', '股票', '交易', 'trading',
  ],
};

/**
 * 路由查询到领域
 * @param {string} query - 用户查询
 * @returns {string} 领域名称 ('all' 表示不限制)
 */
export function routeQuery(query) {
  const q = query.toLowerCase();
  const scores = {};

  for (const [domain, keywords] of Object.entries(DOMAIN_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (q.includes(kw)) {
        score += kw.length >= 3 ? 2 : 1; // 长关键词权重高
      }
    }
    if (score > 0) scores[domain] = score;
  }

  // 最高分领域
  let best = 'all', bestScore = 0;
  for (const [domain, score] of Object.entries(scores)) {
    if (score > bestScore) { best = domain; bestScore = score; }
  }

  // 如果多个领域得分接近，返回 all
  if (bestScore > 0) {
    const secondBest = Object.entries(scores)
      .filter(([d]) => d !== best)
      .reduce((max, [, s]) => Math.max(max, s), 0);
    if (secondBest >= bestScore * 0.7) return 'all'; // 模糊→全搜
  }

  return best;
}

/**
 * 判断文档节点属于哪个领域
 * @param {object} node - 树节点
 * @returns {string}
 */
export function classifyNodeDomain(node) {
  const text = `${node.title || ''} ${node.content || ''}`.toLowerCase();

  // 检查 pharma 特征（文件名前缀）
  const title = (node.title || '').toLowerCase();
  if (title.match(/\d{8}_(cde|nmpa|fda|ema|ich|most|cdr|nhc)/)) return 'pharma';
  if (title.match(/(指导原则|技术指南|管理办法|通知|通告)/)) return 'pharma';

  // 检查 llm 特征
  const llmHits = DOMAIN_KEYWORDS.llm.filter(kw => text.includes(kw)).length;
  const pharmaHits = DOMAIN_KEYWORDS.pharma.filter(kw => text.includes(kw)).length;

  if (llmHits > pharmaHits) return 'llm';
  if (pharmaHits > llmHits) return 'pharma';
  return 'all';
}

/**
 * 文档节点领域过滤
 * @param {object} node - 树节点
 * @param {string} targetDomain - 目标领域 (或 'all')
 * @returns {boolean}
 */
export function nodeMatchesDomain(node, targetDomain) {
  if (targetDomain === 'all') return true;
  return classifyNodeDomain(node) === targetDomain;
}
