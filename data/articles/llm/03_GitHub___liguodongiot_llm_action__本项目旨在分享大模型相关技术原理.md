# GitHub - liguodongiot/llm-action: 本项目旨在分享大模型相关技术原理 ...

**来源**: https://github.com/liguodongiot/llm-action  
**领域**: llm  
**抓取时间**: 2026-03-13T23:34:37.234Z  
**相关性评分**: 0.99978846

---

## 摘要

### LLM推理优化技术

 LLM推理优化技术-概述
 大模型推理优化技术-KV Cache
 大模型推理服务调度优化技术-Continuous batching
 大模型低显存推理优化-Offload技术
 大模型推理优化技术-KV Cache量化
 大模型推理优化技术-张量并行
 大模型推理服务调度优化技术-Chunked Prefill
 大模型推理优化技术-KV Cache优化方法综述
 大模型吞吐优化技术-多LoRA推理服务
 大模型推理服务调度优化技术-公平性调度
 大模型访存优化技术-FlashAttention
 大模型显存优化技术-PagedAttention
 大模型解码优化-Speculative Decoding及其变体
 大模型推理优化-结构化文本生成
 Flash Decoding
 FlashDecoding++

## LLM压缩

近年来，随着Transformer、MOE架构的提出，使得深度学习模型轻松突破上万亿规模参数，从而导致模型变得越来越大，因此，我们需要一些大模型压缩技术来降低模型部署的成本，并提升模型的推理性能。 模型压缩主要分为如下几类：

 模型剪枝（Pruning）
 知识蒸馏（Knowledge Distillation）
 模型量化（Quantization）
 低秩分解（Low-Rank Factorization）

### LLM量化

本系列将针对一些常见大模型量化方案（GPTQ、LLM.int8()、SmoothQuant、AWQ等）进行讲述。 [...] | LLM | 预训练/SFT/RLHF... | 参数 | 教程 | 代码 |
 ---  --- 
| Alpaca | full fine-turning | 7B | 从0到1复现斯坦福羊驼（Stanford Alpaca 7B） | 配套代码 |
| Alpaca(LLaMA) | LoRA | 7B~65B | 1.足够惊艳，使用Alpaca-Lora基于LLaMA(7B)二十分钟完成微调，效果比肩斯坦福羊驼 2. 使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理 | 配套代码 |
| BELLE(LLaMA/Bloom) | full fine-turning | 7B | 1.基于LLaMA-7B/Bloomz-7B1-mt复现开源中文对话大模型BELLE及GPTQ量化   2. BELLE(LLaMA-7B/Bloomz-7B1-mt)大模型使用GPTQ量化后推理性能测试 | N/A |
| ChatGLM | LoRA | 6B | 从0到1基于ChatGLM-6B使用LoRA进行参数高效微调 | 配套代码 |
| ChatGLM | full fine-turning/P-Tuning v2 | 6B | 使用DeepSpeed/P-Tuning v2对ChatGLM-6B进行微调 | 配套代码 |
| Vicuna(LLaMA) | full fine-turning | 7B | 大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼 | N/A |
| OPT | RLHF | 0.1B~66B | 1.一键式 RLHF 训练 DeepSpeed Chat（一）：理论篇   2. 一键式 RLHF 训练 DeepSpeed Chat（二）：实践篇 | 配套代码 | [...] | MiniGPT-4(LLaMA) | full fine-turning | 7B | 大杀器，多模态大模型MiniGPT-4入坑指南 | N/A |
| Chinese-LLaMA-Alpaca(LLaMA) | LoRA（预训练+微调） | 7B | 中文LLaMA&Alpaca大语言模型词表扩充+预训练+指令精调 | 配套代码 |
| LLaMA | QLoRA | 7B/65B | 高效微调技术QLoRA实战，基于LLaMA-65B微调仅需48G显存，真香 | 配套代码 |
| LLaMA | GaLore | 60M/7B | 突破内存瓶颈，使用 GaLore 一张4090消费级显卡也能预训练LLaMA-7B | 配套代码 |

---

## 正文

### LLM推理优化技术

 LLM推理优化技术-概述
 大模型推理优化技术-KV Cache
 大模型推理服务调度优化技术-Continuous batching
 大模型低显存推理优化-Offload技术
 大模型推理优化技术-KV Cache量化
 大模型推理优化技术-张量并行
 大模型推理服务调度优化技术-Chunked Prefill
 大模型推理优化技术-KV Cache优化方法综述
 大模型吞吐优化技术-多LoRA推理服务
 大模型推理服务调度优化技术-公平性调度
 大模型访存优化技术-FlashAttention
 大模型显存优化技术-PagedAttention
 大模型解码优化-Speculative Decoding及其变体
 大模型推理优化-结构化文本生成
 Flash Decoding
 FlashDecoding++

## LLM压缩

近年来，随着Transformer、MOE架构的提出，使得深度学习模型轻松突破上万亿规模参数，从而导致模型变得越来越大，因此，我们需要一些大模型压缩技术来降低模型部署的成本，并提升模型的推理性能。 模型压缩主要分为如下几类：

 模型剪枝（Pruning）
 知识蒸馏（Knowledge Distillation）
 模型量化（Quantization）
 低秩分解（Low-Rank Factorization）

### LLM量化

本系列将针对一些常见大模型量化方案（GPTQ、LLM.int8()、SmoothQuant、AWQ等）进行讲述。 [...] | LLM | 预训练/SFT/RLHF... | 参数 | 教程 | 代码 |
 ---  --- 
| Alpaca | full fine-turning | 7B | 从0到1复现斯坦福羊驼（Stanford Alpaca 7B） | 配套代码 |
| Alpaca(LLaMA) | LoRA | 7B~65B | 1.足够惊艳，使用Alpaca-Lora基于LLaMA(7B)二十分钟完成微调，效果比肩斯坦福羊驼 2. 使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理 | 配套代码 |
| BELLE(LLaMA/Bloom) | full fine-turning | 7B | 1.基于LLaMA-7B/Bloomz-7B1-mt复现开源中文对话大模型BELLE及GPTQ量化   2. BELLE(LLaMA-7B/Bloomz-7B1-mt)大模型使用GPTQ量化后推理性能测试 | N/A |
| ChatGLM | LoRA | 6B | 从0到1基于ChatGLM-6B使用LoRA进行参数高效微调 | 配套代码 |
| ChatGLM | full fine-turning/P-Tuning v2 | 6B | 使用DeepSpeed/P-Tuning v2对ChatGLM-6B进行微调 | 配套代码 |
| Vicuna(LLaMA) | full fine-turning | 7B | 大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼 | N/A |
| OPT | RLHF | 0.1B~66B | 1.一键式 RLHF 训练 DeepSpeed Chat（一）：理论篇   2. 一键式 RLHF 训练 DeepSpeed Chat（二）：实践篇 | 配套代码 | [...] | MiniGPT-4(LLaMA) | full fine-turning | 7B | 大杀器，多模态大模型MiniGPT-4入坑指南 | N/A |
| Chinese-LLaMA-Alpaca(LLaMA) | LoRA（预训练+微调） | 7B | 中文LLaMA&Alpaca大语言模型词表扩充+预训练+指令精调 | 配套代码 |
| LLaMA | QLoRA | 7B/65B | 高效微调技术QLoRA实战，基于LLaMA-65B微调仅需48G显存，真香 | 配套代码 |
| LLaMA | GaLore | 60M/7B | 突破内存瓶颈，使用 GaLore 一张4090消费级显卡也能预训练LLaMA-7B | 配套代码 |

---

*本文由 PageIndex-CN 自进化系统自动抓取*
