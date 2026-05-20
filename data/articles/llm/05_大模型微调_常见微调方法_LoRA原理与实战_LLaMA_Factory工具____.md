# 大模型微调（常见微调方法、LoRA原理与实战、LLaMA-Factory工具 ...

**来源**: https://adg.csdn.net/696f3892437a6b403369af7c.html  
**领域**: llm  
**抓取时间**: 2026-03-13T23:34:37.234Z  
**相关性评分**: 0.9996111

---

## 摘要

存储参数的量化值，同时存储一个全局或局部的缩放因子，用于反量化时恢复精度。

公式：  
 X Int8 = round ( 127 absmax ( X FP32 ) ⋅ X FP32 ) = round ( c FP32 ⋅ X FP32 ) \mathbf{X}^{\text{Int8}} = \text{round}\Big(\frac{127}{\text{absmax}(\mathbf{X}^{\text{FP32}})} \cdot \mathbf{X}^{\text{FP32}}\Big) = \text{round}\big(c^{\text{FP32}} \cdot \mathbf{X}^{\text{FP32}}\big) XInt8=round(absmax(XFP32)127​⋅XFP32)=round(cFP32⋅XFP32)  
解释：

absmax：取整个权重矩阵的绝对值最大值  
 absmax ( X ) = max ⁡ ( ∣ X i j ∣ ) \text{absmax}(\mathbf{X}) = \max(|X\_{ij}|) absmax(X)=max(∣Xij​∣)  
这个值起到 缩放因子 的作用

缩放到 [-127,127]

回浮点值时  
 X FP32 ≈ X Int8 / c FP32 \mathbf{X}^{\text{FP32}} \approx \mathbf{X}^{\text{Int8}} / c^{\text{FP32}} XFP32≈XInt8/cFP32  
相当于查表，表是线性的、等间隔的。

【例子】

假设原始 FP32 数据是 `[2.5, -4.0, 3.2]`

`[2.5, -4.0, 3.2]`
`[79, -127, 102]` [...] 在计算时，这些  i n t 4 int4 int4 值会根据  b i n bin bin 映射回浮点数，得到的近似权重：  
 W ~ = [ 0.12 , − 1.0 , 0.33 , 1.0 , − 0.44 ] \tilde{W} = [0.12, -1.0, 0.33, 1.0, -0.44] W~=[0.12,−1.0,0.33,1.0,−0.44]  
和原始  F P 32 FP32 FP32 版本相比，有少量误差，但总体形状保持一致。

## 四、QLoRA

QLoRA 的全称是 Quantized Low-Rank Adaptation（量化低秩适应）。

它是一个革命性的微调方法，核心思想是：将预训练的大模型以极低的精度（4比特）量化到内存中，从而大幅减少内存占用，在此期间再通过一组少量的、可训练的“低秩适配器”（LoRA）来对模型进行微调。

简单来说，它让你可以用一块消费级GPU（如RTX 3090/4090） 来微调原本需要数张A100才能运行的超大模型。

#### 1. QLoRA三大核心技术

##### 1. 1比特 NormalFloat (NF4) 量化

这是QLoRA的内存压缩核心。

##### 1.2 双量化（Double Quantization）

对量化常数进行二次量化，进一步节省内存。

`scale`
`zero_point`

##### 1.3 分页优化器（Paged Optimizers）

利用CPU内存来分担GPU显存的压力。

#### 2. 核心原理

将参数范围压缩到一个有限范围，比  − 127 -127 −127 到  127 127 127（INT8 Qlora 8）或  − 7 -7 −7 到  7 7 7（INT4 Qlora4）。 [...] 优点：减轻极端值影响 → 大多数权重映射更精确

##### 2.2 NF4量化

NormalFloat4

NF4 的核心是假设权重近似正态分布（μ≈0）。

统计权重的标准差 σ

设置范围

构建 NF4 离散表

#### 3. 量化实现

量化的核心思想是：将一个范围的浮点数值映射到一个更小、更离散的整数集合上。

##### 3.1 均匀量化映射

对于一个浮点数向量 `r` (FP32)，其量化值 `q` (INT4) 可以通过以下公式计算：

`r`
`q`

计算 步长scale  
 scale = m a x − m i n 2 b − 1 \text{scale} = \frac{max - min}{2^b - 1} scale=2b−1max−min​

这里  b = 4 b = 4 b=4

m a x = 2 , m i n = − 2 max = 2, min = -2 max=2,min=−2  
 scale = 2 − ( − 2 ) 16 − 1 = 4 15 ≈ 0.2667 \text{scale} = \frac{2 - (-2)}{16 - 1} = \frac{4}{15} \approx 0.2667 scale=16−12−(−2)​=154​≈0.2667

将  F P 32 FP32 FP32 值映射到 bin 序号（0–15）  
 bin = round ( x − m i n s c a l e ) \text{bin} = \text{round}\left(\frac{x - min}{scale}\right) bin=round(scalex−min​)

---

## 正文

存储参数的量化值，同时存储一个全局或局部的缩放因子，用于反量化时恢复精度。

公式：  
 X Int8 = round ( 127 absmax ( X FP32 ) ⋅ X FP32 ) = round ( c FP32 ⋅ X FP32 ) \mathbf{X}^{\text{Int8}} = \text{round}\Big(\frac{127}{\text{absmax}(\mathbf{X}^{\text{FP32}})} \cdot \mathbf{X}^{\text{FP32}}\Big) = \text{round}\big(c^{\text{FP32}} \cdot \mathbf{X}^{\text{FP32}}\big) XInt8=round(absmax(XFP32)127​⋅XFP32)=round(cFP32⋅XFP32)  
解释：

absmax：取整个权重矩阵的绝对值最大值  
 absmax ( X ) = max ⁡ ( ∣ X i j ∣ ) \text{absmax}(\mathbf{X}) = \max(|X\_{ij}|) absmax(X)=max(∣Xij​∣)  
这个值起到 缩放因子 的作用

缩放到 [-127,127]

回浮点值时  
 X FP32 ≈ X Int8 / c FP32 \mathbf{X}^{\text{FP32}} \approx \mathbf{X}^{\text{Int8}} / c^{\text{FP32}} XFP32≈XInt8/cFP32  
相当于查表，表是线性的、等间隔的。

【例子】

假设原始 FP32 数据是 `[2.5, -4.0, 3.2]`

`[2.5, -4.0, 3.2]`
`[79, -127, 102]` [...] 在计算时，这些  i n t 4 int4 int4 值会根据  b i n bin bin 映射回浮点数，得到的近似权重：  
 W ~ = [ 0.12 , − 1.0 , 0.33 , 1.0 , − 0.44 ] \tilde{W} = [0.12, -1.0, 0.33, 1.0, -0.44] W~=[0.12,−1.0,0.33,1.0,−0.44]  
和原始  F P 32 FP32 FP32 版本相比，有少量误差，但总体形状保持一致。

## 四、QLoRA

QLoRA 的全称是 Quantized Low-Rank Adaptation（量化低秩适应）。

它是一个革命性的微调方法，核心思想是：将预训练的大模型以极低的精度（4比特）量化到内存中，从而大幅减少内存占用，在此期间再通过一组少量的、可训练的“低秩适配器”（LoRA）来对模型进行微调。

简单来说，它让你可以用一块消费级GPU（如RTX 3090/4090） 来微调原本需要数张A100才能运行的超大模型。

#### 1. QLoRA三大核心技术

##### 1. 1比特 NormalFloat (NF4) 量化

这是QLoRA的内存压缩核心。

##### 1.2 双量化（Double Quantization）

对量化常数进行二次量化，进一步节省内存。

`scale`
`zero_point`

##### 1.3 分页优化器（Paged Optimizers）

利用CPU内存来分担GPU显存的压力。

#### 2. 核心原理

将参数范围压缩到一个有限范围，比  − 127 -127 −127 到  127 127 127（INT8 Qlora 8）或  − 7 -7 −7 到  7 7 7（INT4 Qlora4）。 [...] 优点：减轻极端值影响 → 大多数权重映射更精确

##### 2.2 NF4量化

NormalFloat4

NF4 的核心是假设权重近似正态分布（μ≈0）。

统计权重的标准差 σ

设置范围

构建 NF4 离散表

#### 3. 量化实现

量化的核心思想是：将一个范围的浮点数值映射到一个更小、更离散的整数集合上。

##### 3.1 均匀量化映射

对于一个浮点数向量 `r` (FP32)，其量化值 `q` (INT4) 可以通过以下公式计算：

`r`
`q`

计算 步长scale  
 scale = m a x − m i n 2 b − 1 \text{scale} = \frac{max - min}{2^b - 1} scale=2b−1max−min​

这里  b = 4 b = 4 b=4

m a x = 2 , m i n = − 2 max = 2, min = -2 max=2,min=−2  
 scale = 2 − ( − 2 ) 16 − 1 = 4 15 ≈ 0.2667 \text{scale} = \frac{2 - (-2)}{16 - 1} = \frac{4}{15} \approx 0.2667 scale=16−12−(−2)​=154​≈0.2667

将  F P 32 FP32 FP32 值映射到 bin 序号（0–15）  
 bin = round ( x − m i n s c a l e ) \text{bin} = \text{round}\left(\frac{x - min}{scale}\right) bin=round(scalex−min​)

---

*本文由 PageIndex-CN 自进化系统自动抓取*
