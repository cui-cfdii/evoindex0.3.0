# OpenViking 深度调研 & EvoIndex 移植可行性报告

> 调研日期: 2026-05-19 | 项目: github.com/volcengine/OpenViking | 出品: 火山引擎

---

## 一、OpenViking 概览

| 维度 | 详情 |
|------|------|
| **定位** | Context Database for AI Agents（上下文数据库） |
| **技术栈** | Python 3.10+ + Rust (RAGFS) + npm CLI |
| **生态** | OpenClaw/OpenCode/ClaudeCode/LangChain 插件 |
| **协议** | AGPLv3 (核心) / Apache 2.0 (CLI, examples) |

## 二、五大核心技术

### 1. 文件系统管理范式
虚拟文件系统 `viking://resources/`、`viking://user/`、`viking://agent/`，每个上下文有唯一 URI

### 2. L0/L1/L2 分层上下文加载 ⭐⭐⭐
- L0 Abstract (~100 tokens) → 快速相关性
- L1 Overview (~2k tokens) → 规划决策
- L2 Details (无限) → 深度阅读
- **效果: Token -91%**

### 3. 目录递归检索 ⭐⭐⭐
意图分析 → 向量定位高分目录 → 目录内二次检索 → 递归钻取子目录 → 聚合
- **核心理念**: "先锁定高分目录，再细化内容探索"

### 4. 检索轨迹可视化
每次检索的目录浏览轨迹完整保留，可观察根因

### 5. 自动会话管理
用户偏好更新 + Agent 经验积累，越用越聪明

## 三、Benchmark (LoCoMo10, 1540 条对话)

| 方案 | 完成率 | Token | vs 原始 |
|------|--------|-------|---------|
| 原始 OpenClaw | 35.65% | 24.6M | — |
| + OpenViking | **52.08%** | 4.26M | +43%/Token -91% |
| + LanceDB | 44.55% | 51.6M | — |

## 四、EvoIndex 移植优先级

### P0 - 立即实施
1. **目录递归检索** → hybrid_query.mjs 改造，召回率 +15~25%
2. **L0/L1 分层加载** → 树节点增强，Token -80%

### P1 - 近期规划
3. **检索轨迹可视化** → 调试接口，问题定位效率 +10x
4. **URI 体系 (evo://)** → doc_manager.mjs 升级

### P2 - 中期规划
5. **自动会话记忆** → session_memory.mjs 新模块

### P3 - 长期规划
6. **Rust 加速层** → napi-rs 重写树搜索，性能 +10x

## 五、从 70.8% → 97% 路径

```
当前 70.8%
  ├─ bge-m3 中文嵌入       +5~10%
  ├─ 目录递归检索 (OV)     +15~25%  ← 最大杠杆
  ├─ 子树内容匹配          +3~5%
  └─ CMA-ES 领域感知       +2~5%
  ▼ 预计 95~97%
```
