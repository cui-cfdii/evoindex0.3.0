# EvoIndex 3.0 自建记忆方案（LLMWiki 模式）

## 问题

MEM0 依赖外部 API Key → 违背纯本地原则。2.0 的本地记忆只有 `string.includes()` 关键词匹配，精确度极低。

## 方案：用 EvoIndex 自己的基础设施建记忆

核心逻辑：**EvoIndex 3.0 已经有 LanceDB + nomic-embed + qwen3.5-9b，直接用这套东西做记忆，零新增依赖。**

```
记忆写入：
  查询+结果+反馈 → nomic-embed 向量化 → LanceDB memories 表
                  → 每 N 条触发 LLM 摘要 → 摘要也向量化存入
                  → JSONL 兜底持久化

记忆检索：
  新查询 → nomic-embed → LanceDB ANN 语义搜索 → 找到相似历史
         → Top-K 匹配 + 匹配度阈值过滤
         → 返回：相关历史 + 用户偏好 + 已验证检索策略

执行计划缓存：
  相似查询 → 命中缓存 → 直接复用 树权重/向量权重/图展开深度
  → 跳过意图路由 → 端到端延迟 <5ms
```

## 与 2.0 的对比

| 维度 | 2.0 本地记忆 | 3.0 LLMWiki 记忆 |
|------|------------|-----------------|
| 检索方式 | `string.includes()` | nomic-embed 语义向量 |
| 召回精度 | 低（"苹果手机"找不到"iPhone"） | 高（余弦相似度） |
| LLM 参与 | 零 | 记忆摘要 + 关系抽取 |
| 存储 | JSONL 文件 | LanceDB + JSONL 兜底 |
| 外部依赖 | 无 | 无（复用已有基础设施） |
| 执行计划缓存 | 无 | ✅ 相似查询直达 |
| 新增依赖 | — | 0（全部复用 P0 的 LanceDB + nomic-embed） |

## 三层记忆模型

```
Layer M1: 热记忆 (<5ms)
  ├── SQLite FTS5 精确匹配（查询原文完全命中 → 直接返回缓存）
  └── 适用：用户重复问完全相同的问题

Layer M2: 温记忆 (<50ms)
  ├── LanceDB 语义向量检索（查询嵌入 → ANN 找相似历史）
  ├── 适用：用户问类似但不完全相同的问题
  └── 返回：相关历史 + 偏好领域 + 已验证权重

Layer M3: 冷记忆 (离线, LLM)
  ├── 每 50 条新交互触发 LLM 记忆整理
  ├── LLM 任务：抽取关键事实 → 生成记忆摘要 → 更新用户画像
  └── 适用：长期知识沉淀，不阻塞在线查询
```

## LLM 的记忆角色（轻量、离线）

```
触发条件：memories 表新增 ≥ 50 条 → 自动触发

LLM Prompt（~200 tokens）：
"""
你是 EvoIndex 的记忆整理器。分析以下最近交互：

{最近 50 条查询+结果+反馈}

输出 JSON：
{
  "key_facts": ["用户关心的核心事实1", ...],    // 最多 5 条
  "user_profile": {"preferred_domains": [...], "query_style": "..."},
  "frequent_patterns": ["常见查询模式"],
  "contradictions": ["用户观点变化/矛盾"],
  "summary_embedding": "简要摘要(用于语义搜索)"
}
"""

结果：摘要存入 LanceDB memories_summary 表，下次相似查询可匹配到摘要
```

## 执行计划缓存（核心价值）

这是 LLMWiki 模式对 EvoIndex 3.0 最大的增量价值：

```
查询流程：
  1. 新查询 → 嵌入 → LanceDB 搜索 memories 表
  2. 命中相似查询（余弦 > 0.85） → 提取该查询的已验证检索参数
     {
       tree_weight: 0.4,
       vector_weight: 0.6,
       graph_expand_depth: 2,
       top_k: 5,
       intent: "knowledge_query",
       domain: "medical_ai"
     }
  3. 直接套用参数 → 跳过意图路由 → 直达 Layer 1 检索
  4. 延迟：<5ms（一次向量查找 + 一次表查询）
```

## 实现路径

### Phase M0（0.5 天，P2 阶段）
1. 新建 `src/core/llm_memory.mjs`
2. 复用 P0 的 LanceDB 连接 + nomic-embed
3. 实现 `storeQuery()` → 向量化 → LanceDB insert
4. 实现 `searchSimilar()` → 嵌入 → ANN → 返回 Top-K

### Phase M1（1 天，P2 阶段）
5. 实现执行计划缓存 `getCachedPlan()`
6. 实现 LLM 记忆摘要触发（每 50 条）
7. 在 `src/index.mjs` 的 query 流程中插入记忆层
8. 测试：重复查询延迟 <5ms

### Phase M2（远期）
9. 实现用户画像的长期演化（跨会话持久化）
10. 实现跨文档矛盾检测（两篇文档说法冲突 → 标记）

## 零外部依赖验证

| 组件 | 来源 |
|------|------|
| 向量库 | LanceDB（P0 已引入） |
| 嵌入模型 | nomic-embed（P0 已引入） |
| LLM | qwen3.5-9b（P1 已使用） |
| 持久化 | JSONL（2.0 已有） |
| 热匹配 | SQLite FTS5（可轻量引入，也可用内存 Map） |

**新增 npm 依赖：0。全部复用 EvoIndex 3.0 已有基础设施。**
