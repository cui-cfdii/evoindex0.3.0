# [PDF] RFG Framework: Retrieval-Feedback-Grounded Multi-Query ...

**来源**: https://www.scitepress.org/Papers/2025/138369/138369.pdf  
**领域**: ai_research  
**抓取时间**: 2026-03-14T03:46:34.288Z  
**相关性评分**: 0.55796635

---

## 摘要

• Supervised: We evaluate several high-performance models trained on benchmark corpora, such as Contriever-ft (fine-tuned on MS MARCO) (Izacard et al., 2022), DPR (fine-tuned on Natural Questions) (Karpukhin et al., 2020), and two of the most powerful recent embedding models, which consistently rank as top perform-ers on the MTEB (Massive Text Embedding Benchmark) leaderboard (Muennighoff et al., KDIR 2025 - 17th International Conference on Knowledge Discovery and Information Retrieval 512 2023)1: BGE-large (Xiao et al., 2024) and GTE-large (Li et al., 2023).
Furthermore, our query expansion approach was compared against two LLM-based baseline methods: • HyDE: Evaluated on all the aforementioned em-bedding models and datasets. [...] A comprehensive study by Weller et al. (Weller et al., 2024) revealed a strong negative correlation KDIR 2025 - 17th International Conference on Knowledge Discovery and Information Retrieval 510 between a retrieval model’s base results and the gains obtained from expansion, indicating that exist-ing techniques benefit weaker models, but harm more robust ones. In this sense, high-performing models are negatively affected because the generated text in-troduces noise that dilutes the original relevance sig-nal.
Our present investigation is situated at the in-tersection of these research lines.
Our proposed RFG framework explores LLMs for query expansion. [...] 5.1 Overall Outcome Comparison Table 1 presents the main results of our evaluation, comparing the effectiveness of the different query ex-pansion methods on the four selected datasets. The re-ported metric is nDCG@10. The evaluated methods are: 1) No Expansion, which serves as our baseline; 2) Query2doc; 3) HyDE; and 4) RFG, our proposed method. Each method was evaluated on the full set of embedding models. To facilitate visual analysis, the best results for each model and dataset are marked in bold in Table 1. The background colors of the cells indicate the performance change concerning the ”No 2The code for this research is available at:  3Mistral AI pricing information, accessed July 24, 2025, available at:  RFG Framework: Retrieval-Feedback-Grounded Multi-Query Expansion 513 Table 1:

---

## 正文

• Supervised: We evaluate several high-performance models trained on benchmark corpora, such as Contriever-ft (fine-tuned on MS MARCO) (Izacard et al., 2022), DPR (fine-tuned on Natural Questions) (Karpukhin et al., 2020), and two of the most powerful recent embedding models, which consistently rank as top perform-ers on the MTEB (Massive Text Embedding Benchmark) leaderboard (Muennighoff et al., KDIR 2025 - 17th International Conference on Knowledge Discovery and Information Retrieval 512 2023)1: BGE-large (Xiao et al., 2024) and GTE-large (Li et al., 2023).
Furthermore, our query expansion approach was compared against two LLM-based baseline methods: • HyDE: Evaluated on all the aforementioned em-bedding models and datasets. [...] A comprehensive study by Weller et al. (Weller et al., 2024) revealed a strong negative correlation KDIR 2025 - 17th International Conference on Knowledge Discovery and Information Retrieval 510 between a retrieval model’s base results and the gains obtained from expansion, indicating that exist-ing techniques benefit weaker models, but harm more robust ones. In this sense, high-performing models are negatively affected because the generated text in-troduces noise that dilutes the original relevance sig-nal.
Our present investigation is situated at the in-tersection of these research lines.
Our proposed RFG framework explores LLMs for query expansion. [...] 5.1 Overall Outcome Comparison Table 1 presents the main results of our evaluation, comparing the effectiveness of the different query ex-pansion methods on the four selected datasets. The re-ported metric is nDCG@10. The evaluated methods are: 1) No Expansion, which serves as our baseline; 2) Query2doc; 3) HyDE; and 4) RFG, our proposed method. Each method was evaluated on the full set of embedding models. To facilitate visual analysis, the best results for each model and dataset are marked in bold in Table 1. The background colors of the cells indicate the performance change concerning the ”No 2The code for this research is available at:  3Mistral AI pricing information, accessed July 24, 2025, available at:  RFG Framework: Retrieval-Feedback-Grounded Multi-Query Expansion 513 Table 1:

---

*本文由 PageIndex-CN 自进化系统自动抓取*
