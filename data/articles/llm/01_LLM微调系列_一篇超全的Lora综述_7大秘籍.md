# LLM微调系列：一篇超全的Lora综述，7大秘籍

**来源**: https://www.53ai.com/news/finetuning/2024122409124.html  
**领域**: llm  
**抓取时间**: 2026-03-13T23:34:37.228Z  
**相关性评分**: 0.9999008

---

## 摘要

参数效率（Parameter Efficiency）：LoRA通过仅更新模型参数的一个小子集来实现参数效率，这减少了微调时所需的内存和计算需求，同时没有增加推理延迟。
 内存使用减少（Reduced Memory Usage）：LoRA显著降低了微调大型语言模型（LLMs）时的内存使用量。这包括模型权重存储所需的内存、前向传播期间中间激活占用的内存、反向传播期间存储梯度所需的内存，以及优化器状态所需的内存。
 实际效率比较（Empirical Efficiency Comparison）：在特定硬件配置下，使用批量大小为1的LLaMA2-7B模型进行全参数微调和LoRA微调的全面比较。研究表明，全参数微调需要大约60GB的内存，超过了单个NVIDIA RTX4090 GPU的容量；而LoRA微调仅需要大约23GB的内存。
 内存使用的细分（Breakdown of Memory Usage）：LoRA减少了优化内存和梯度内存的显著用量，分别减少了大约25GB和14GB。虽然LoRA引入了一些额外的“增量参数”，导致激活内存和权重内存略有增加（总计约2GB），但考虑到整体内存的减少，这种增加是可以忽略不计的。
 前向传播加速（Forward Propagation Acceleration）：减少内存使用还带来了前向传播的加速。LoRA比全参数微调快1.9倍。
 其他应用案例（Beyond Fine-tuning）：除了微调之外，LoRA还可以应用于其他学习范式，例如预训练和持续训练。在预训练中，LoRA可以用于训练高秩网络；在持续训练中，LoRA可以解决灾难性遗忘问题。 [...] ## 4 提升效率（Efficiency Improving ）

随着下游任务越来越多，lora插件的数量也会随之增加，要进一步提高其效率成为了一个关键问题。因此，可以从以下三个方面来改进：

 参数缩减（Parameter Reduction）：通过参数冻结（Parameter Freezing）、参数剪枝（Parameter Pruning）和参数共享（Parameter Sharing）来减少LoRA的可训练参数数量，降低内存成本。
 参数量化（Parameter Quantization）：通过减少参数的位宽（例如，从32位浮点数量化到4位整数），来降低LoRA的内存和计算成本。这包括后训练量化（Post-Training Quantization, PTQ）和量化感知训练（Quantization-Aware Training, QAT）方法。
 并行LoRA计算框架（Parallel LoRA Computing Frameworks）：通过在单个GPU或GPU集群上并行微调或推理多个LoRA插件，节省计算资源并提高LoRA的效率。这包括并行微调（Parallel Fine-tuning）和并行推理（Parallel Inference）框架。

### 4.1 参数缩减（Parameter Reduction）

通过缩减参数数量，减少了模型在微调期间的内存占用。减少了需要更新的参数数量，从而加快了训练和推理的速度。

#### 4.1.1 参数冻结（Parameter Freezing）

通过在微调过程中冻结一部分LoRA参数，只更新其余的参数。 [...] 通过在微调过程中冻结一部分LoRA参数，只更新其余的参数。

 LoRA-SP：随机选择一半的LoRA参数进行冻结，只更新剩余的参数。
 LoRA-FA：冻结下投影权重，只更新每个LoRA层的上投影权重。
 AFLoRA：构建一个低秩可训练路径，并在训练LoRA时逐步冻结参数。
 DropBP：通过在反向传播过程中随机丢弃一些LoRA梯度计算来加速训练过程

#### 4.1.2 参数剪枝（Parameter Pruning）

在训练和推理过程中，通过评估参数的重要性来移除不重要的LoRA参数。

 LoRA-drop：使用每层LoRA的输出来评估参数的重要性，并剪枝那些不重要的参数。
 LoRA-prune：基于LoRA的梯度信息，联合剪枝LoRA矩阵和大型语言模型（LLM）的参数，以优化模型结构。
 LoRA-shear：通过剪枝特定的参数来调整模型的剪枝粒度，实现更细粒度的优化。

#### 4.1.3 参数共享（Parameter Sharing）：

通过在不同的层或模块之间共享参数来减少参数总数。

 VeRA：VeRA（Vector-based Random Matrix Adaptation）提出在所有层之间共享一对冻结的随机矩阵，并通过“缩放向量”进行逐层适应。
 VB-LoRA：VB-LoRA（Vector Bank-based LoRA）提出了一种“分割和共享”范式，通过秩一分解将LoRA的低秩分解进行分割，并基于混合模型实现全局共享。

### 4.2 参数量化（Parameter Quantization）

大型语言模型（LLMs）通常需要大量的计算资源，部署到资源受限的环境中时，参数量化通过减少参数的精度，使得模型可以用更少的比特来表示，从而减少内存占用和计算复杂性。

---

## 正文

参数效率（Parameter Efficiency）：LoRA通过仅更新模型参数的一个小子集来实现参数效率，这减少了微调时所需的内存和计算需求，同时没有增加推理延迟。
 内存使用减少（Reduced Memory Usage）：LoRA显著降低了微调大型语言模型（LLMs）时的内存使用量。这包括模型权重存储所需的内存、前向传播期间中间激活占用的内存、反向传播期间存储梯度所需的内存，以及优化器状态所需的内存。
 实际效率比较（Empirical Efficiency Comparison）：在特定硬件配置下，使用批量大小为1的LLaMA2-7B模型进行全参数微调和LoRA微调的全面比较。研究表明，全参数微调需要大约60GB的内存，超过了单个NVIDIA RTX4090 GPU的容量；而LoRA微调仅需要大约23GB的内存。
 内存使用的细分（Breakdown of Memory Usage）：LoRA减少了优化内存和梯度内存的显著用量，分别减少了大约25GB和14GB。虽然LoRA引入了一些额外的“增量参数”，导致激活内存和权重内存略有增加（总计约2GB），但考虑到整体内存的减少，这种增加是可以忽略不计的。
 前向传播加速（Forward Propagation Acceleration）：减少内存使用还带来了前向传播的加速。LoRA比全参数微调快1.9倍。
 其他应用案例（Beyond Fine-tuning）：除了微调之外，LoRA还可以应用于其他学习范式，例如预训练和持续训练。在预训练中，LoRA可以用于训练高秩网络；在持续训练中，LoRA可以解决灾难性遗忘问题。 [...] ## 4 提升效率（Efficiency Improving ）

随着下游任务越来越多，lora插件的数量也会随之增加，要进一步提高其效率成为了一个关键问题。因此，可以从以下三个方面来改进：

 参数缩减（Parameter Reduction）：通过参数冻结（Parameter Freezing）、参数剪枝（Parameter Pruning）和参数共享（Parameter Sharing）来减少LoRA的可训练参数数量，降低内存成本。
 参数量化（Parameter Quantization）：通过减少参数的位宽（例如，从32位浮点数量化到4位整数），来降低LoRA的内存和计算成本。这包括后训练量化（Post-Training Quantization, PTQ）和量化感知训练（Quantization-Aware Training, QAT）方法。
 并行LoRA计算框架（Parallel LoRA Computing Frameworks）：通过在单个GPU或GPU集群上并行微调或推理多个LoRA插件，节省计算资源并提高LoRA的效率。这包括并行微调（Parallel Fine-tuning）和并行推理（Parallel Inference）框架。

### 4.1 参数缩减（Parameter Reduction）

通过缩减参数数量，减少了模型在微调期间的内存占用。减少了需要更新的参数数量，从而加快了训练和推理的速度。

#### 4.1.1 参数冻结（Parameter Freezing）

通过在微调过程中冻结一部分LoRA参数，只更新其余的参数。 [...] 通过在微调过程中冻结一部分LoRA参数，只更新其余的参数。

 LoRA-SP：随机选择一半的LoRA参数进行冻结，只更新剩余的参数。
 LoRA-FA：冻结下投影权重，只更新每个LoRA层的上投影权重。
 AFLoRA：构建一个低秩可训练路径，并在训练LoRA时逐步冻结参数。
 DropBP：通过在反向传播过程中随机丢弃一些LoRA梯度计算来加速训练过程

#### 4.1.2 参数剪枝（Parameter Pruning）

在训练和推理过程中，通过评估参数的重要性来移除不重要的LoRA参数。

 LoRA-drop：使用每层LoRA的输出来评估参数的重要性，并剪枝那些不重要的参数。
 LoRA-prune：基于LoRA的梯度信息，联合剪枝LoRA矩阵和大型语言模型（LLM）的参数，以优化模型结构。
 LoRA-shear：通过剪枝特定的参数来调整模型的剪枝粒度，实现更细粒度的优化。

#### 4.1.3 参数共享（Parameter Sharing）：

通过在不同的层或模块之间共享参数来减少参数总数。

 VeRA：VeRA（Vector-based Random Matrix Adaptation）提出在所有层之间共享一对冻结的随机矩阵，并通过“缩放向量”进行逐层适应。
 VB-LoRA：VB-LoRA（Vector Bank-based LoRA）提出了一种“分割和共享”范式，通过秩一分解将LoRA的低秩分解进行分割，并基于混合模型实现全局共享。

### 4.2 参数量化（Parameter Quantization）

大型语言模型（LLMs）通常需要大量的计算资源，部署到资源受限的环境中时，参数量化通过减少参数的精度，使得模型可以用更少的比特来表示，从而减少内存占用和计算复杂性。

---

*本文由 PageIndex-CN 自进化系统自动抓取*
