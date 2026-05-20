# Text-to-SQL Benchmarks for Enterprise Realities: Under Massive...

**来源**: https://openreview.net/forum?id=gXkIkSN2Ha  
**领域**: ai_research  
**抓取时间**: 2026-03-14T03:46:34.291Z  
**相关性评分**: 0.50022036

---

## 摘要

In fact, current Text-to-SQL methods do not consider knowledge retrieval at all. Many table-retrieval or schema-linking methods are LLM-driven, but the scale of tables and knowledge in our benchmark far exceeds the context length that current LLM methods can handle. Existing approaches cannot support retrieval at this scale.

### Keyword Search Tool Baseline

We evaluate BM25 as the keyword search tool and compare keyword-only, embedding-only, and hybrid retrieval. The results are:

1.   Retrieving Top-10 Tables

| Search Method | Precision | Recall | F1 Score | Perfect Recall |
 ---  --- 
| Embedding-based search tool | 16.4 | 86.0 | 27.0 | 76.5 |
| Keyword search tool (BM25) | 10.5 | 56.4 | 17.4 | 41.7 |
| Hybrid Search | 16.6 | 87.2 | 27.4 | 77.8 | [...] Nevertheless, our benchmark successfully introduces three enterprise-specific and unavoidable challenges, complementing Spider 2.0, BEAVER, and others. Together, these benchmarks form a more complete and diversified Text-to-SQL evaluation ecosystem.

−＝≡

#### Official Comment by Authors

Copy URL of note chEWx46HMd

Official Comment by Authors 03 Dec 2025, 10:12 Everyone

Comment:

W3) Scope of Evaluation

### Baselines with Retrieval Capabilities

After carefully reviewing the papers you provided, we note that:

---

## 正文

In fact, current Text-to-SQL methods do not consider knowledge retrieval at all. Many table-retrieval or schema-linking methods are LLM-driven, but the scale of tables and knowledge in our benchmark far exceeds the context length that current LLM methods can handle. Existing approaches cannot support retrieval at this scale.

### Keyword Search Tool Baseline

We evaluate BM25 as the keyword search tool and compare keyword-only, embedding-only, and hybrid retrieval. The results are:

1.   Retrieving Top-10 Tables

| Search Method | Precision | Recall | F1 Score | Perfect Recall |
 ---  --- 
| Embedding-based search tool | 16.4 | 86.0 | 27.0 | 76.5 |
| Keyword search tool (BM25) | 10.5 | 56.4 | 17.4 | 41.7 |
| Hybrid Search | 16.6 | 87.2 | 27.4 | 77.8 | [...] Nevertheless, our benchmark successfully introduces three enterprise-specific and unavoidable challenges, complementing Spider 2.0, BEAVER, and others. Together, these benchmarks form a more complete and diversified Text-to-SQL evaluation ecosystem.

−＝≡

#### Official Comment by Authors

Copy URL of note chEWx46HMd

Official Comment by Authors 03 Dec 2025, 10:12 Everyone

Comment:

W3) Scope of Evaluation

### Baselines with Retrieval Capabilities

After carefully reviewing the papers you provided, we note that:

---

*本文由 PageIndex-CN 自进化系统自动抓取*
