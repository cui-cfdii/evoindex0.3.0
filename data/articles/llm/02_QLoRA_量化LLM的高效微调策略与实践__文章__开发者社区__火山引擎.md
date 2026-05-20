# QLoRA：量化LLM的高效微调策略与实践- 文章- 开发者社区- 火山引擎

**来源**: https://developer.volcengine.com/articles/7386867895290036263  
**领域**: llm  
**抓取时间**: 2026-03-13T23:34:37.233Z  
**相关性评分**: 0.9998822

---

## 摘要

`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
model\_name = "TinyPixel/Llama-2-7B-bf16-sharded"
bnb\_config = BitsAndBytesConfig(
load\_in\_4bit=True,
bnb\_4bit\_quant\_type="nf4",
bnb\_4bit\_compute\_dtype=torch.float16,
)
model = AutoModelForCausalLM.from\_pretrained(
model\_name,
quantization\_config=bnb\_config,
trust\_remote\_code=True
)
model.config.use\_cache = False`

加载预训练模型的分词器，并将其配置为在序列结尾添加填充标记，以便在使用模型进行推断时进行批处理。

`tokenizer = AutoTokenizer.from\_pretrained(model\_name, trust\_remote\_code=True)
tokenizer.pad\_token = tokenizer.eos\_token`

创建一个 PEFT 配置对象，以便在训练和评估模型时使用。 [...] AI 大模型体验中心
AI 大模型体验中心
动手实验室
动手实验室
Agent 评测集
Agent 评测集
AI 案例广场
AI 案例广场

AI 大模型体验中心
AI 大模型体验中心
动手实验室
动手实验室
Agent 评测集
Agent 评测集
AI 案例广场
AI 案例广场

# QLoRA：量化LLM的高效微调策略与实践

技术狂潮AI

picture.image点击上方蓝字关注我们

picture.image

一、前言

在大型语言模型（LLM）领域，微调是提高性能和调整行为的关键过程。然而，由于内存需求巨大，对于大型模型进行微调可能非常昂贵。最近，华盛顿大学发表了一项关于解决这一问题的创新方案——QLoRA（Quantized Low-Rank Adapter）。

QLoRA
是一种新的微调大型语言模型（LLM）的方法，它能够在节省内存的同时保持速度。其工作原理是首先将LLM进行4位量化，从而显著减少模型的内存占用。接着，使用低阶适配器（LoRA）方法对量化的LLM进行微调。LoRA使得改进后的模型能够保留原始LLM的大部分准确性，同时具有更小的体积和更快的速度。

以上是对QLoRA的简要介绍，下面将进一步探讨其原理和应用。

二、QLoRA 介绍

QLoRA
是一种高效的微调方法，通过将梯度反向传播到低阶适配器(LoRA)中，以显著减少内存使用量。
它可以在单个48GB GPU上微调 650 亿个参数的模型，并且能够保持完整的16位微调任务性能
。

同时还推出了一个名为 Guanaco 的新模型家族，它在Vicuna基准上表现出色，达到了ChatGPT性能水平的 99.3%。令人惊喜的是，只需要在单个GPU上进行24小时的微调，就能够取得如此优异的结果。这些创新使得在资源有限的情况下，能够以更高效的方式进行模型微调，并取得了非常令人满意的成果。 [...] ### 

### 

6.1、量化

量化参数由 `BitsandbytesConfig` 控制，如下所示：

`BitsandbytesConfig`
`load_in_4bit`
`bnb_4bit_compute_dtype`
`bnb_4bit_use_double_quant`
`bnb_4bit_quant_type`
`fp4`
`nf4`
`nf4`
`model = AutoModelForCausalLM.from\_pretrained(
model\_name\_or\_path='/name/or/path/to/your/model',
load\_in\_4bit=True,
device\_map='auto',
max\_memory=max\_memory,
torch\_dtype=torch.bfloat16,
quantization\_config=BitsAndBytesConfig(
load\_in\_4bit=True,
bnb\_4bit\_compute\_dtype=torch.bfloat16,
bnb\_4bit\_use\_double\_quant=True,
bnb\_4bit\_quant\_type='nf4'
),
)`

### 

### 

6.2、分页优化器

为了处理 GPU 偶尔耗尽内存的情况，QLoRA 使用了利用 NVIDIA 统一内存功能的分页优化器，该功能在 CPU 和 GPU 之间执行自动页到页传输，其功能与 CPU RAM 和 GPU 之间的常规内存分页非常相似。磁盘。此功能用于为优化器状态分配分页内存，然后在 GPU 内存不足时将其移至 CPU RAM，并在需要时转移回 GPU 内存。

我们可以使用以下参数访问分页优化器。

---

## 正文

`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
model\_name = "TinyPixel/Llama-2-7B-bf16-sharded"
bnb\_config = BitsAndBytesConfig(
load\_in\_4bit=True,
bnb\_4bit\_quant\_type="nf4",
bnb\_4bit\_compute\_dtype=torch.float16,
)
model = AutoModelForCausalLM.from\_pretrained(
model\_name,
quantization\_config=bnb\_config,
trust\_remote\_code=True
)
model.config.use\_cache = False`

加载预训练模型的分词器，并将其配置为在序列结尾添加填充标记，以便在使用模型进行推断时进行批处理。

`tokenizer = AutoTokenizer.from\_pretrained(model\_name, trust\_remote\_code=True)
tokenizer.pad\_token = tokenizer.eos\_token`

创建一个 PEFT 配置对象，以便在训练和评估模型时使用。 [...] AI 大模型体验中心
AI 大模型体验中心
动手实验室
动手实验室
Agent 评测集
Agent 评测集
AI 案例广场
AI 案例广场

AI 大模型体验中心
AI 大模型体验中心
动手实验室
动手实验室
Agent 评测集
Agent 评测集
AI 案例广场
AI 案例广场

# QLoRA：量化LLM的高效微调策略与实践

技术狂潮AI

picture.image点击上方蓝字关注我们

picture.image

一、前言

在大型语言模型（LLM）领域，微调是提高性能和调整行为的关键过程。然而，由于内存需求巨大，对于大型模型进行微调可能非常昂贵。最近，华盛顿大学发表了一项关于解决这一问题的创新方案——QLoRA（Quantized Low-Rank Adapter）。

QLoRA
是一种新的微调大型语言模型（LLM）的方法，它能够在节省内存的同时保持速度。其工作原理是首先将LLM进行4位量化，从而显著减少模型的内存占用。接着，使用低阶适配器（LoRA）方法对量化的LLM进行微调。LoRA使得改进后的模型能够保留原始LLM的大部分准确性，同时具有更小的体积和更快的速度。

以上是对QLoRA的简要介绍，下面将进一步探讨其原理和应用。

二、QLoRA 介绍

QLoRA
是一种高效的微调方法，通过将梯度反向传播到低阶适配器(LoRA)中，以显著减少内存使用量。
它可以在单个48GB GPU上微调 650 亿个参数的模型，并且能够保持完整的16位微调任务性能
。

同时还推出了一个名为 Guanaco 的新模型家族，它在Vicuna基准上表现出色，达到了ChatGPT性能水平的 99.3%。令人惊喜的是，只需要在单个GPU上进行24小时的微调，就能够取得如此优异的结果。这些创新使得在资源有限的情况下，能够以更高效的方式进行模型微调，并取得了非常令人满意的成果。 [...] ### 

### 

6.1、量化

量化参数由 `BitsandbytesConfig` 控制，如下所示：

`BitsandbytesConfig`
`load_in_4bit`
`bnb_4bit_compute_dtype`
`bnb_4bit_use_double_quant`
`bnb_4bit_quant_type`
`fp4`
`nf4`
`nf4`
`model = AutoModelForCausalLM.from\_pretrained(
model\_name\_or\_path='/name/or/path/to/your/model',
load\_in\_4bit=True,
device\_map='auto',
max\_memory=max\_memory,
torch\_dtype=torch.bfloat16,
quantization\_config=BitsAndBytesConfig(
load\_in\_4bit=True,
bnb\_4bit\_compute\_dtype=torch.bfloat16,
bnb\_4bit\_use\_double\_quant=True,
bnb\_4bit\_quant\_type='nf4'
),
)`

### 

### 

6.2、分页优化器

为了处理 GPU 偶尔耗尽内存的情况，QLoRA 使用了利用 NVIDIA 统一内存功能的分页优化器，该功能在 CPU 和 GPU 之间执行自动页到页传输，其功能与 CPU RAM 和 GPU 之间的常规内存分页非常相似。磁盘。此功能用于为优化器状态分配分页内存，然后在 GPU 内存不足时将其移至 CPU RAM，并在需要时转移回 GPU 内存。

我们可以使用以下参数访问分页优化器。

---

*本文由 PageIndex-CN 自进化系统自动抓取*
