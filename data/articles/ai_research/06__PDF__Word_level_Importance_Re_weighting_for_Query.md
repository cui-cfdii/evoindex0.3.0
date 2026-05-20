# [PDF] Word-level Importance Re-weighting for Query Expansion

**来源**: https://aclanthology.org/2025.findings-acl.434.pdf  
**领域**: ai_research  
**抓取时间**: 2026-03-14T03:46:34.292Z  
**相关性评分**: 0.45101276

---

## 摘要

In ACL.
Mingyang Song and Mao Zheng. 2024. A survey of query optimization in large language models. arXiv preprint arXiv:2412.17558.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-hishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. In NeurIPS.
Haoyu Wang, Ruirui Li, Haoming Jiang, Jinjin Tian, Zhengyang Wang, Chen Luo, Xianfeng Tang, Mon-ica Xiao Cheng, Tuo Zhao, and Jing Gao. 2024.
Blendfilter: Advancing retrieval-augmented large lan-guage models via query generation blending and knowledge filtering. In EMNLP.
Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query expansion with large language models. In EMNLP. [...] 3.1 Formulation of Query Expansion The BM25 framework can be re-formulated to illustrate the impact of word-level importance re-weighting in query expansion as: 𝑆( ˜ 𝑄, Chunk) = ∑︁ ∀(𝑡,𝐼𝑡)∈˜ 𝑄 𝐼𝑡· BM25(𝑡, Chunk), (1) where ˜ 𝑄denotes an expanded query derived from the original query 𝑄using a query expansion method. Specifically, ˜ 𝑄is represented as a set of tuples (𝑡, 𝐼𝑡), where each tuple contains a unique word 𝑡from the expanded query and its correspond-ing importance score 𝐼𝑡. Here, BM25(𝑡, Chunk) represents the BM25 score of word 𝑡with respect to a given chunk. Based on this formulation, we iden-tify two key factors that influence retrieval results in query expansion: (1) the set of unique words and (2) the importance score of each word. [...] 4.4 Task 2: Question and Answering Since HyDE, Q2D, and MuGI (Gao et al., 2023a; Wang et al., 2023; Zhang et al., 2024) have focused primarily on IR evaluation without reporting their QA performance in RAG, it remains uncertain whether gains in IR performance directly lead to better QA results. Therefore, evaluating both IR and QA performance is crucial. Table 3 shows the QA performance of three query expansion methods, along with the canonical BM25 as a reference. Note that we omit the Acc scores for the FiQA dataset, as all values are 0 due to its long-form QA nature.

---

## 正文

In ACL.
Mingyang Song and Mao Zheng. 2024. A survey of query optimization in large language models. arXiv preprint arXiv:2412.17558.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-hishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. In NeurIPS.
Haoyu Wang, Ruirui Li, Haoming Jiang, Jinjin Tian, Zhengyang Wang, Chen Luo, Xianfeng Tang, Mon-ica Xiao Cheng, Tuo Zhao, and Jing Gao. 2024.
Blendfilter: Advancing retrieval-augmented large lan-guage models via query generation blending and knowledge filtering. In EMNLP.
Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query expansion with large language models. In EMNLP. [...] 3.1 Formulation of Query Expansion The BM25 framework can be re-formulated to illustrate the impact of word-level importance re-weighting in query expansion as: 𝑆( ˜ 𝑄, Chunk) = ∑︁ ∀(𝑡,𝐼𝑡)∈˜ 𝑄 𝐼𝑡· BM25(𝑡, Chunk), (1) where ˜ 𝑄denotes an expanded query derived from the original query 𝑄using a query expansion method. Specifically, ˜ 𝑄is represented as a set of tuples (𝑡, 𝐼𝑡), where each tuple contains a unique word 𝑡from the expanded query and its correspond-ing importance score 𝐼𝑡. Here, BM25(𝑡, Chunk) represents the BM25 score of word 𝑡with respect to a given chunk. Based on this formulation, we iden-tify two key factors that influence retrieval results in query expansion: (1) the set of unique words and (2) the importance score of each word. [...] 4.4 Task 2: Question and Answering Since HyDE, Q2D, and MuGI (Gao et al., 2023a; Wang et al., 2023; Zhang et al., 2024) have focused primarily on IR evaluation without reporting their QA performance in RAG, it remains uncertain whether gains in IR performance directly lead to better QA results. Therefore, evaluating both IR and QA performance is crucial. Table 3 shows the QA performance of three query expansion methods, along with the canonical BM25 as a reference. Note that we omit the Acc scores for the FiQA dataset, as all values are 0 due to its long-form QA nature.

---

*本文由 PageIndex-CN 自进化系统自动抓取*
