# GaitPart

* * *

# 0 摘要

## 1\. 背景与痛点：为什么要提出新方法？

- **步态识别的潜力：** 步态识别是一种通过行人的走路姿态进行身份识别的生物特征技术。它的优势在于可以在**远距离**（不需要像人脸识别那样靠近摄像头）进行识别，因此应用前景非常广阔。
- **现有的问题（Whole-body vs. Parts）：** 目前大多数主流方法是将**整个人体**看作一个整体单元来提取时空特征。
    
    - **缺陷：** 作者观察到，人体在行走时，**不同部位（如头、躯干、腿）的视觉外观和运动模式差异巨大**。例如，手臂是摆动的，躯干相对静止，腿部则交替跨步。如果把全身“一视同仁”地混在一起分析，会忽略这些局部细节的差异。

## 2\. 核心理念：分块建模 (Part-based Modeling)

作者的假设非常直观：**人体的每个部分都需要拥有独立的时空表达。**

基于最新的研究成果（局部特征对识别有益），作者提出了GaitPart 模型。这个模型的核心思想是不再“囫囵吞枣”地看全身，而是将人体切分为若干个部分，对每个部分分别进行精细化的分析。

## 3\. 关键创新点：GaitPart 强在哪？

GaitPart 通过两个主要的技术模块来提升性能，分别对应空间（Spatial）和时间（Temporal）两个维度：

### A. 空间维度：Focal Convolution Layer (焦点卷积层)

- **作用：** 增强**局部（Part-level）空间特征**的细粒度学习。
- **解释：** 普通的卷积可能关注全局，容易把特征“平滑”掉。焦点卷积层专门针对划分好的身体部位，提取更精细、更有针对性的外观特征。

### B. 时间维度：Micro-motion Capture Module (MCM, 微动捕捉模块)

- **作用：** 这是一个并行的结构，分别对应人体的不同部位。
- **核心逻辑（短时 vs. 长时）：**
    
    - 以往的方法往往关注**长距离**的时间特征（比如看这人走了几十步）。作者认为步态是周期性的（走一步和走十步模式是一样的），长距离特征会有大量冗余。
    - MCM 专注于**短距离（Short-range）的时间特征**，也就是捕捉行走过程中瞬间的、微小的动作变化（Micro-motion）。这种方式更高效，且更能抓取到最具辨识度的运动细节。

## 总结 Summary

简单来说，这段话讲述了作者如何通过“化整为零”的策略改进了步态识别。

> 传统方法就像是看一个人走路的“大概轮廓”和“长视频”；
> 
> GaitPart 则是把人切分成不同部位（头、身、腿），用“放大镜”看局部的细节（Focal Conv），并专注于捕捉瞬间的微小动作（MCM），从而实现了更精准的身份识别。

* * *

# 1 Introduction

## **1.基于深度学习的步态识别现有方法**

文献回顾（Related Work）。主要介绍了三种代表性的深度学习思路，分别列举了基于**3D-CNN**、混合模型（Auto-Encoder + LSTM）**和**集合（Set）的三种不同技术路线：

### A. 基于 3D-CNN 的方法 (Thomas et al.)

- **核心技术：** **3D-CNN（三维卷积神经网络）**。
- “Thomas et al.[25] applied 3D-CNN to extract the spatio-temporal information, trying to find a general descriptor for human gait.” ([Fan 等, 2020, p. 14225](zotero://select/library/items/2AKFTYPU)) ([pdf](zotero://open-pdf/library/items/5WHYGL7L?page=1))
- **解释：** 普通的卷积（2D-CNN）只能处理单张图片的**空间**信息（长和宽）。
    
    - 3D-CNN 则引入了**时间**维度，像切面包块一样同时处理视频的“长、宽、时间”。
    - **目的：** 直接从视频块中提取**时空联合特征**（Spatio-temporal），试图找到一个通用的描述符来代表一个人的步态。

### B. 基于序列建模的方法 (GaitNet)

- **核心技术：** **Auto-Encoder（自编码器） + LSTM（长短期记忆网络）**。
- “GaitNet[30] proposed an Auto-Encoder framework to extract the gait-related features from raw RGB images and then used LSTMs to model the temporal changes of gait sequence.” ([Fan 等, 2020, p. 14225](zotero://select/library/items/2AKFTYPU)) ([pdf](zotero://open-pdf/library/items/5WHYGL7L?page=1))
- **解释：** 这是一个“分两步走”的策略：
    
    1. **空间提取：** 使用自编码器从原始的 **RGB 图像**（注意：这里用的是彩色图，而非剪影）中提取与步态相关的视觉特征。
    2. **时间建模：** 将提取出的特征按时间顺序输入到 **LSTM** 中。LSTM 是一种专门处理序列数据的网络，它能“记住”动作的前后变化逻辑。

### C. 基于集合的方法 (GaitSet)

- **核心技术：** **Set-based Learning（基于集合的学习）**。
- “GaitSet[5] assumed that the appearance of a silhouette has contained its position information and thus regarded gait as a set to extract temporal information.” ([Fan 等, 2020, p. 14225](zotero://select/library/items/2AKFTYPU)) ([pdf](zotero://open-pdf/library/items/5WHYGL7L?page=1))
- **地位：** GaitSet 是步态识别领域非常经典且极具颠覆性的论文。
- **核心逻辑：**
    
    - **传统思维：** 认为走路是一个严格的序列（第一步 -> 第二步 -> 第三步），时间顺序很重要。
    - **GaitSet 的思维：** 作者认为，单张剪影（Silhouette）的外观其实已经隐含了它的位置和状态信息。因此，**不需要严格的时间顺序**。
    - 它把一个视频看作**一堆图片的集合（Set）**，而不是一个序列。这样做的好处是计算更灵活，对帧数和顺序不敏感。

## 2\. 动机：为什么要推翻旧方法？(Motivation)

作者通过观察（Observation）和引用证据，指出了现有方法的两个主要缺陷，确立了 GaitPart 的设计哲学：

### A. 空间上：从“整体”到“局部” (Spatial)

- **旧方法缺陷：** 把整个人体当作一个单位处理。
- **作者洞察：** 人体不同部位（头、躯干、腿）在行走时的形状和运动模式完全不同（如图 1(a) 所示）。

[image]

- **结论：** “分而治之”是更好的策略。每个部位都需要独立的表达方式，因为局部的细粒度特征对识别身份更有帮助。

### B. 时间上：从“长时依赖”到“微动模式” (Temporal)

- **旧方法缺陷：**
    
    1. 要么**完全不建模时间**（丢失了动作信息）。
    2. 要么**过度建模长距离依赖**（Long-range dependencies），比如使用深层 3D-CNN 或 RNN 去分析整个长视频序列。
- **作者洞察：** 步态是**周期性**的（左脚迈完右脚迈，周而复始）。过长的序列包含大量重复冗余的信息，反而降低了识别的灵活性。
- **结论：** **“短小精悍”是核心。局部的、短距离的时空特征（即微动模式 Micro-motion Patterns**）才是最具辨识度的特征。

## 3\. 架构：GaitPart 是如何工作的？(Framework)

基于上述假设，作者提出了 GaitPart 框架（对应图 1(b)）。其处理流程如下：

[image]

```
图 1：(a)：人类步态的不同部分在行走过程中具有明显不同的形状和运动模式。(b)：步态部分概述，由帧级部分特征提取器（FPFE）和微运动捕捉模块（MCM）组成。
```

1. **输入：** 一系列的步态剪影（Gait Silhouettes）。
2. **第一步：FPFE (帧级局部特征提取器)**
    
    - 这是一个特殊的堆叠 CNN。
    - 它先处理每一帧图像，然后进行**预定义的水平切割（Horizontal Partition）**。
    - *形象理解：* 就像把每一帧画面切成几条，分别对应头、肩、腰、腿等。
3. **第二步：并行处理**
    
    - 切分后的每一部分（Part）都有自己独立的通道。
    - 这些通道是**相互独立**的（Part-independent），互不干扰。
4. **第三步：MCM (微动捕捉模块)**
    
    - 每个部位对应的序列进入各自的 MCM。
    - MCM 负责捕捉该部位特有的“微动模式”。
5. **输出：** 最后简单地将所有 MCM 的输出拼接（Concatenate）起来，形成最终的步态特征表达。

## 4\. 三大核心贡献 (Contributions)

作者总结了这篇论文的三个具体的创新点：

### 创新点 1：Focal Convolution (FConv, 焦点卷积) —— 针对空间

- **位置：** 在 FPFE 模块中应用。
- **核心思想：** 这是一种简单但有效的卷积应用方式。它让卷积核（Kernel）专注于输入帧中**特定部位的局部细节**。
- **目的：** 增强对局部空间特征的“细粒度学习”（Fine-grained learning）。不仅仅是看个大概，而是要看清局部的纹理和形状。

### 创新点 2：Micro-motion Capture Module (MCM) —— 针对时间

- **理论依据：** 作者认为，对于周期性步态，**短距离**的时空特征（微动）是最具辨识度的，而长距离依赖是冗余且低效的。
- **技术实现：** 提出了一个基于注意力机制（Attention-based）的 MCM。它既能捕捉局部的微小动作，也能兼顾对整个步态序列的全局理解。

### 创新点 3：SOTA 效果验证

- **实验：** 在两个最大的步态数据集（**CASIA-B** 和 **OU-MVLP**）上进行了广泛测试。
- **结果：** GaitPart 的性能大幅超越了之前的最先进方法（outperforms by a large margin）。同时，通过消融实验（Ablation experiments）证明了上述 FPFE 和 MCM 组件各自的有效性。

## 总结 (Key Takeaway)

这段话的核心逻辑链条是：

> 既然人体各部位运动不同 -> 我们要分块处理 (Part-based)；
> 
> 既然步态是周期性的 -> 我们不需要看太长，只需看短时的微动 (Short-range/Micro-motion)。

* * *

# 2 **Related Work**

为了方便理解，将这段内容拆解为 **空间特征提取 (Spatial)** 和 **时间建模 (Temporal)** 两个主要维度，以及作者针对这两点提出的**针对性改进**。

## 1\. 空间特征提取 (Spatial Feature Extraction)

作者主要讨论了如何处理单帧图像的空间特征，并对比了步态识别与行人重识别（Re-ID）的差异。

### A. 现有方法的局限

- **“一刀切”的问题：** 以前的大多数方法（无论是基于 2D-CNN 还是 3D-CNN）都是对 **整个特征图 (Whole Feature Map)** 进行统一的卷积操作。
- **作者的批判：** 这种做法忽略了一个显而易见的事实——人体的不同部位（头、躯干、腿）在行走任务中具有显著的差异。把它们混在一起处理是不够精细的。

> *注：传统的 GEI 方法，它将所有动作压缩成一张图，模糊了具体的部位运动细节。*

### B. 借鉴与区分：Part-based Model (基于分块的模型)

作者提到了在 **Person Re-ID (行人重识别)** 领域常用的“水平切片”方法，并指出了它与步态识别的区别

<table><tbody><tr><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">领域</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">假设前提 (Assumption)</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">处理方式</span></strong></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">Person Re-ID</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">认为不同部位可能共享相同的属性（如衣服颜色、纹理），因此可以共享一部分参数。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">Part-shared (部分共享)</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">Gait Recognition</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">认为不同部位的</span><strong><span style="background-color: rgba(0, 0, 0, 0)">运动模式</span></strong><span style="background-color: rgba(0, 0, 0, 0)">和</span><strong><span style="background-color: rgba(0, 0, 0, 0)">外观形状</span></strong><span style="background-color: rgba(0, 0, 0, 0)">截然不同（头是不动的，腿是摆动的）。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">Part-dependent (部分独立)</span></strong></p></td></tr></tbody></table>

### C. 作者的创新：Focal Convolution (FConv)

基于上述分析，作者提出了 **FConv (焦点卷积)**：

- **机制：** 先将输入特征图切分成若干个水平条带（Parts），然后对**每个条带单独进行卷积**。
- **目的：** 当层数加深时，顶层神经元的**感受野 (Receptive Field)** 会被限制在该条带内部。
- **效果：** 强制网络专注于捕捉**该部位内部的局部细节**，而不是被其他部位的信息干扰。

* * *

## 2\. 时间建模 (Temporal Modeling)

这一部分讨论如何处理视频序列中的时间信息。作者将现有方法分为三类，并指出了它们的不足，最后引出了自己的“短时建模”理念。

### A. 三大现有流派对比

<table><tbody><tr><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">流派</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">代表方法</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">优点</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">缺点 (作者观点)</span></strong></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">3D-CNN</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">[25] 等</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">直接提取时空特征。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">难以训练，性能提升有限。</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">LSTM (RNN)</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">GaitNet [30]</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">能模拟时间序列变化。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">“画蛇添足”</span></strong><span style="background-color: rgba(0, 0, 0, 0)">：步态是周期性的，不需要像语言那样严格的长序列约束。</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">Set-based (集合)</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">GaitSet [5]</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">将视频看作图片集合，通过池化聚合特征。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">简洁但粗糙</span></strong><span style="background-color: rgba(0, 0, 0, 0)">：虽然效果好，但没有显式地去建模时间的动态变化 (Micro-motion)。</span></p></td></tr></tbody></table>

### B. 作者的洞察：长时依赖 vs. 短时微动

- **观察：** 步态是**周期性 (Periodic)** 的。走完一个周期（左脚迈出到右脚跟上），动作就会重复。
- **推论：**
    
    - **长距离依赖 (Long-range dependencies)** 是冗余的。看一个人走 100 步和走 2 步，获取的身份信息其实差不多，反而增加了计算负担。
    - **短距离特征 (Local short-range)** 才是关键。即在一个周期内的微小动作变化（Micro-motion）。

### C. 作者的创新：Micro-motion Capture Module (MCM)

基于此，作者提出了 **MCM (微动捕捉模块)**。

- **核心策略：** 放弃捕捉冗长的全局序列关系，转而专注于**局部的、短时的微动特征**建模。这将在Sec 3.3详细讨论。

## Summary

这段 Related Work 的核心逻辑链条如下：

1. **空间上：** 既然人体各部位运动不同 -> **我们要分块 (Part-based)** -> 但不能像 Re-ID 那样共享参数，要独立处理 -> 提出 **FConv**。
2. **时间上：** 既然步态是周期性的 -> 长时间序列是啰嗦的 -> **我们要关注短时微动** -> 提出 **MCM**。

* * *

# 3.1 Pipeline

[image]

```
图2：GaitPart的框架。Block1、2和3由FConvs和池化层组成。HP为水平池化，MCM为微运动捕捉模块。其中，MCM_j是组成局部特征矩阵中第j行的所有向量，生成时空特征v_j，以便最终识别。
```

## 1\. 图像输入与空间特征提取 (Input & FPFE)

- **输入：** 模型接收一个包含$t$帧的步态剪影序列（Sequence of gait silhouettes）作为输入。
- **处理模块：** **帧级局部特征提取器 (FPFE)**。这是一个专门设计的卷积网络（如图中 Block1, Block2, Block3 所示），它逐帧处理输入图像。
- **输出：** 对于序列中的每一帧 $f_i$ ，FPFE 会生成对应的空间特征图 $F_i$ 。这产生了一个特征图序列 $S_F$ 。

## 2\. 水平池化与切分 (Horizontal Pooling - HP)

为了获得人体局部的精细特征，模型对特征图进行了特殊的处理：

- **水平切分：** **HP 模块**将每个特征图 $F_i$在水平方向上切分为$n$个部分（parts）。
- **池化操作：** 对于每一个切分出来的部分（第$j$部分），HP 模块结合使用了**全局平均池化 (Global Average Pooling)** 和 **全局最大池化 (Global Max Pooling)**。
    
    - 公式：$p_{j,i} = \text{Avgpool2d}(F_{j,i}) + \text{Maxpool2d}(F_{j,i})$ 。
- **结果：** 这一步将三维的特征图降维成了特征列向量 $p_{j,i}$ 。

## 3\. 构建 (Part Representation Matrix)

- 经过 HP 处理后，整个步态序列被转化为一个 **局部特征矩阵 (PR-Matrix)**，记为 $P$ 。
- **结构含义：** 这是一个$n \times t$的 矩阵（如图 2 中间部分的网格所示）：
    
    - **列 (Column)：** 代表某一时刻（第 $i$帧 ）全身各部位的特征。
    - **行 (Row)：** 代表人体**某一部位（第** $j$**部 分）随时间变化**的特征序列，记为 $P_{j,\cdot}$ 。这一行向量直观地反映了该特定身体部位的步态变化。

## 4\. 时序特征聚合 (Temporal Feature Aggregator - TFA)

最后一步是对时间维度进行建模，提取最终的识别特征：

- **并行处理：** 既然每一行代表一个身体部位，GaitPart使用了$n$个并行的**微动捕捉模块 (MCM)** 来分别处理这些行向量。
    
    - 例如，$MCM_1$专门处理头部序列，$MCM_n$专门处理腿部序列。
- **独立性：** 值得注意的是，这 $n$个 MCM 的参数是**相互独立**的。这意味着模型是“部位独立”（Part-dependent）的，每个部位都有自己专属的参数来捕捉其独特的微动模式。
- **输出：** 每个 MCM 输出一个特征向量 $v_j$ 。最后，这些向量通过独立的全连接层（FC）映射到度量空间，用于计算 Triplet Loss 进行身份识别。

## 总结

简单来说，这个 Pipeline 的核心逻辑是：**先逐帧提取图像特征，然后把人切成** $n$**条 ，最后让**$n$**个独立的处理器分别去分析这** $n$**个条带随时间的细微变化。**

* * *

# 3.2 **帧级局部特征提取器**FPFE

[image]

```
图3.(a)：深层网络中顶层神经元感受野的扩展。上：常规情况。底部：使用FConv。 (b)：FConv的示意图，特征图按其尺寸显示，例如C×H×W。
```

**FPFE (帧级局部特征提取器)** 及其核心组件 **FConv (焦点卷积)**。这是 GaitPart 模型中负责提取空间特征的关键部分。

## 1\. 核心创新：Focal Convolution (FConv, 焦点卷积)

作者提出 FConv 是为了解决传统卷积“看太宽”的问题。

### **A. 运作机制 (Operation)**

可以把 FConv 想象成一个“分而治之”的处理过程。

[image]

1. **切分 (Split)：**输入的特征图被**水平切分**成 $p$个 部分（Strips）。例如图中 $p=4$ ，图像就被切成了 4 个横条。
2. **独立卷积 (Separate Convolution)：**对每一个切分出来的横条，**分别**进行常规的卷积操作。
3. **拼接 (Concatenate)：**处理完后，再把这 $p$个结果在水平方向上重新拼回去，形成输出特征图。

> **注意：** 当 $p=1$时 ，FConv 就等同于普通的卷积层。

### **B. 设计动机 (Motivation)**

为什么要这么麻烦地切开再卷？核心目的是为了**限制感受野 (Receptive Field)**：

[image]

- **传统卷积 (上方图示)：**随着网络层数加深，顶层神经元的感受野会变得很大，甚至覆盖全身。这意味着处理“腿部”特征的神经元可能会受到“头部”信息的干扰。
- **FConv (下方图示)：**通过切分，强制卷积核只在各自的横条内滑动。这样，即使网络很深，顶层神经元的关注点依然被**限制**在对应的局部区域内（Local details）。
- **好处：** 这能增强对**细粒度特征 (Fine-grained features)** 的学习，让腿部的特征提取器只专注于腿部细节，不受上半身干扰。

## 2\. 整体结构：Frame-level Part Feature Extractor (FPFE)

理解了 FConv 后，FPFE 的结构就很好理解了。它是一个专门设计的卷积网络，用于处理输入的每一帧剪影。

- **位置：** 在整个 GaitPart 流程中，FPFE 位于最前端。它包含 Block1, Block2, Block3。

[image]

- **组成：** FPFE 由 **3 个 Block** 组成。
- **内部构造：** 每个 Block 内部包含 **2 个 FConv 层**。

[image]

```
表1. 帧级局部特征提取器的具体结构。InC、OutC、Kernel和Pad分别表示FConv的输入通道、输出通道、内核大小和填充。特别地，p表示FConv中预定义部分的数量。
```

## 总结

FPFE 的设计哲学非常明确：通过堆叠 FConv，模型在提取空间特征时，人为地设立了“隔离带”。这确保了当信息从浅层传到深层时，头部、躯干和腿部的特征是相对独立演化的，从而提取出更纯净、更具辨识度的局部空间特征。

# 3.3 **时序特征聚合器TFA**

**GaitPart** 模型中的**时序特征聚合器 (Temporal Feature Aggregator, TFA)** 及其核心组件**微动捕捉模块 (Micro-motion Capture Module, MCM)**。

[image]

这是模型处理“时间”维度的关键部分。拆解为三个层级：**总体架构**、**第一步：构建微动 (MTB)** 和 **第二步：时序池化 (TP)**

## 1\. 总体架构：MCM 是做什么的？

[image]

```
图4.微动捕捉模块（MCM，包括MTB和TP模块）的详细结构。 MTB模块在序列维度上滑动，将每个相邻的2r+1列向量压缩成微动特征向量。然后，TP模块使用一个简单最大值函数来收集帧和通道维度中最具辨别力的微运动特征以进行最终识别。
```

- **角色：** 这一步发生在“切片”之后。既然人体被切成了 $n$个水平条带，我们就有 $n$ 个并行的通道。TFA 就是由 $n$个并行的 **MCM** 组成的。
- **任务：** 每个 MCM 负责处理对应身体部位的特征序列（即 PR-Matrix 中的一行 $S_p$ ）。
- **流程：** 一个 MCM 内部包含两个子模块，按顺序执行：
    
    1. **MTB (Micro-motion Template Builder)：**负责捕捉短时的微小动作。
    2. **TP (Temporal Pooling)：**负责将整个序列聚合成一个最终特征向量。

[image]

```
图5. 实际微动作捕捉模块MCM的抽象结构，包含TP和两个具有不同窗口大小（3和5）的并行MTB模块。
```

> *注：如图 5 所示，实际应用中，一个 MCM 包含了两个并行的 MTB 分支（MTB1 和 MTB2）以及一个 TP 模块。*

## 2\. 第一步：微动模板构建器 (MTB)

这是 MCM 的核心创新点。作者认为，步态识别不需要看太长，**短时 (Short-range)** 的邻域特征才是最关键的。

### **A. 核心逻辑：滑动窗口 (Sliding Window)**

MTB 就像一个滑动的窗口探测器。

- **输入：** 某一时刻的特征 $p_i$ 及其前后的邻居帧（共 $2r+1$帧 ）。
- **操作 (TempFunc):**在这个小窗口内，通过统计函数把这些帧“压缩”成一个微动特征向量 $m_i$ 。
- **具体算法：** 作者借鉴了 GEI 的思路，使用了**1D Global Average Pooling** 和 **1D Global Max Pooling** 的相加。
    
    - $S_m = \text{Avgpool1d}(S_p) + \text{Maxpool1d}(S_p)$

### **B. 增强机制：通道注意力 (Channel-wise Attention)**

为了进一步提取更有辨识度的特征，MTB 引入了注意力机制来对特征进行“重加权” (Re-weighting)。

[image]

```
表2. MTB1和MTB2的确切结构。InC、OutC、Kernel和Pad分别表示FConv的输入通道、输出通道、内核大小和填充。特别地，C和s分别表示输入特征图的通道和压缩比。请注意，“|”两边的值分别代表MTB1和MTB2的设置。
```

- 使用一个小型的卷积网络 **Conv1dNet**（结构见 Table 2）来计算权重。
- **公式：** $S_m^{re} = S_m \cdot \text{Sigmoid}(S_{logits})$ 。
- **目的：** 自动判断哪些通道的微动特征更重要，予以放大；哪些是噪声，予以抑制。

### **C. 多尺度设计 (Multi-scale)**

如图5和图 4 所示，作者在实践中使用了 **两个并行的 MTB**：

- **MTB1：**窗口大小为 3 (Kernel=3)。
- **MTB2：**窗口大小为 5 (Kernel=5)。
- **目的：** 融合不同尺度的时序信息，捕捉不同速度或跨度的微动特征。

## 3\. 第二步：时序池化 (TP)

MTB 输出的是一串微动特征序列，TP 的任务是把这串序列变成**一个**向量 $v$ ，作为最终的身份指纹。

### **A. 核心原则：步态时序聚合原则**

作者提出了一个非常重要的理论依据：**Ground Principle of Gait Temporal Aggregation**。

- **含义：** 步态是**周期性**的。在一个完整的周期（Cycle）内，信息量已经饱和。
- **推论：** 只要视频长度超过一个周期，再增加视频长度，不应该改变提取出的特征结果。

### **B. 为什么选 Max Pooling？**

作者对比了两种聚合方式：

1. **Mean (平均值):**
    
    - **缺点：** 平均值会受到视频长度 $t$的影响。如果视频长度不是周期的整数倍，平均值会波动。这对现实中长度不定的视频很不友好。
2. **Max (最大值):**
    
    - **优点：** 只要视频包含至少一个完整周期，序列中的最大值（Max）就是固定的。
    - **结论：** Max Pooling 完美符合上述原则（Eq.11 = Eq.12），因此被选为最终方案。

## Summary

**MCM 的工作流程**可以概括为：

1. 用不同大小的**滑动窗口**（3 和 5）扫过整个视频序列。
2. 在窗口内提取**微动特征**并利用**注意力机制**强化关键信息。
3. 最后，用 **Max Pooling** 挑出整个过程中最显著的特征，以此作为该身体部位的最终步态表达。

* * *

# 3.4 **实施细节 (Implementation Details)**

这部分内容不再涉及新的理论创新，而是侧重于“怎么把模型搭建出来”、“怎么训练”以及“怎么测试”。分为**网络配置**、**训练策略**和**测试方法** 三个部分。

## 1\. 网络配置与超参数 (Network Hyper-parameters)

这部分主要补充了 FPFE（帧级特征提取器）的具体设置，特别是关于 FConv 中 $p$ 值（分块数量）的选择。

- **基础组件：** FPFE 由 FConv 层、Max Pooling 层和 Leaky ReLU 激活函数组成。
- **核心参数** $p$**( Parts Number)：**
    
    [image]
    
    - **设置规律：** 随着网络层数变深，$p$ 的值逐渐增大。
    - **具体配置：** 参照 **Table 1**，Block 1 中 $p=1$ （相当于普通卷积），Block 2 中 $p=4$ ，Block 3 中 $p=8$ 。
    - **原因：** 当 $p$越大，对感受野的限制就越强。在浅层（Deep shallow）我们允许感受野大一点，但在深层（Deep layers），为了提取精细的局部特征，强制限制感受野只关注局部。

## 2\. 训练策略：损失函数与数据采样 (Loss and Sampler)

这部分解决了两个实际问题：用什么标准来优化网络？以及长短不一的视频怎么喂给模型？

### **A. 损失函数 Loss Function**

- **方法：** 采用了 **Separate Batch All (BA+) Triplet Loss**（分离的难样本三元组损失）。
- **分离 (Separate) 的含义：** 模型输出了 $n$个特征向量（分别对应头、身、腿等）。计算损失时，是**对应的部位和对应的部位比**（比如：用样本 A 的“头部”去对比样本 B 的“头部”），而不是混在一起比。
- **Batch Size：**设定为 $(p_r, k)$ ，其中 $p_r$是人数，$k$是每个人抽取的样本数。

### **B. 数据采样 (Sampler)**

步态视频长度不一，但训练神经网络通常需要固定的输入维度。作者采取了以下策略：

- **训练阶段 (Train Phase)：** 必须凑齐固定的 **30 帧**。
    
    1. **正常视频：** 先截取一段 30-40 帧的片段，然后从中随机抽取并排序 **30 帧**作为输入。
    2. **过短视频 (<15 帧)：** 直接丢弃 (Discard)。
    3. **中等视频 (15-30 帧)：** 重复采样 (Repeatedly sampled) 直到凑够数。
- **测试阶段 (Test Phase)：** 不需要固定长度，**原始视频 (Raw video)** 直接输入模型进行全量计算。

## 3\. 测试方法 (Testing)

- **距离度量：** 在比对两个样本（Gallery vs. Probe）时，计算它们输出的特征向量之间的**欧几里得距离 (Euclidean distance)**。
- **最终得分：** 将所有部位（$n$个 向量）的距离取**平均值 (Average)**，作为判定两个人是否为同一人的最终依据。

## 总结

这一节不仅提供了复现代码所需的具体参数，还给出了处理非定长视频数据的实用工程技巧（采样策略）。

* * *

# 4 实验设置

## 4.1 **数据集介绍**和**训练配置**

### 1\. 数据集介绍 (Datasets)

作者选择了两个最具代表性的公开步态数据集：**CASIA-B**（侧重多状态变化）和 **OU-MVLP**（侧重海量数据规模）。

#### **A. CASIA-B (复杂状态数据集)**

这是一个广泛使用的中等规模数据集，最大的特点是**包含多种行走状态**（如背包、穿大衣），非常考验模型的鲁棒性。

- **规模：** 124 个对象（Subjects）。
- **视角 (Views)：** 每个人有 11 个视角（0° 到 180°，每 18° 一个视角）。
- **行走状态 (Conditions)：** 每个视角下有 10 个序列，分为三类：
    
    - **NM (Normal, 正常行走)：**6 个序列 (NM#1-6)。
    - **BG (Bag, 背包)：**2 个序列 (BG#1-2)。
    - **CL (Clothing, 穿外套)：**2 个序列 (CL#1-2)。
- **实验协议 (Protocol)：**遵循标准协议：
    
    - **训练集：** 前 74 人。
    - **测试集：** 后 50 人。
    - **测试方式：**
        
        - **Gallery (注册库/底库)：**NM#1-4（即用这 4 次正常行走的视频作为已知底库）。
        - **Probe (探针/待测样本)：**分为三个子集进行测试——NM#5-6（测正常）、BG#1-2（测背包）、CL#1-2（测换衣）。
    - *注：CL（换衣）通常是最难识别的场景。*

#### **B. OU-MVLP (超大规模数据集)**

这是目前世界上最大的公开步态数据集，主要考验模型在大数据下的泛化能力。

- **规模：** 10,307 个对象（是 CASIA-B 的近百倍）。
- **划分：** 5,153 人用于训练，5,154 人用于测试。
- **视角：** 14 个视角（覆盖 0°-90° 和 180°-270°）。
- **序列：** 每个视角只有 2 个序列（#01, #02）。
- **测试方式：** 序列 #01 做底库 (Gallery)，序列 #02 做待测 (Probe)。

### 2\. 训练细节 (Training Details)

为了保证实验公平且可复现，作者详细列出了参数设置。值得注意的是，针对两个不同规模的数据集，模型配置略有调整。

#### **A. 通用配置 (Common Configuration)**

- **输入尺寸：** 所有的步态剪影被调整为 **64 × 44**。
- **优化器：** Adam。
- **学习率 (LR)：**1e-4。
- **三元组损失 Margin：**0.2。

#### **B. 针对 CASIA-B 的配置**

- **Batch Size：**(8, 16)。*一个 Batch 包含 8个人，每个人取 16 个样本。*
- **迭代次数：**12 万次 (120K)。

#### **C. 针对 OU-MVLP 的特殊调整**

由于这个数据集数据量巨大（是 CASIA-B 的 20 倍），作者加深了网络结构：

- **网络加深：** 在 FPFE 中**增加了一个额外的 Block**（包含两个 FConv 层），输出通道数设为 256。
- **参数** $p$**的 调整：** 随着网络加深，$p$值 （分块数）设置为 **1, 1, 3, 3**。
- **训练增强：**
    
    - Batch Size 增大为 (32, 16)。
    - 迭代次数增加到 25 万次 (250K)。
    - 学习率策略：在 15 万次时降至 1e-5。

### 总结

这一段展示了严谨的实验设计：

1. **CASIA-B** 用来证明模型能不能抗干扰（背包、换衣服）。
2. **OU-MVLP** 用来证明模型能不能“在大场面下”工作（海量人群）。
3. **OU-MVLP 的网络调整** 暗示了 GaitPart 架构具有可扩展性，可以通过加深层数来适应更大的数据规模。

* * *

## 4.2 对比实验

**GaitPart** 与当时最先进（State-of-the-art, SOTA）的其他步态识别方法进行的对比实验结果。作者通过这两个实验证明了 GaitPart 不仅在准确率上领先，而且在模型效率上也具有显著优势。从 **CASIA-B（综合性能对比）** 和 **OU-MVLP（泛化能力对比）** 两个维度解读。

### 1\. CASIA-B 上的对比实验 (综合性能)

在这一部分，作者对比了三种代表性的方法：**CNN-LB** (基于GEI)、**GaitSet** (基于集合) 和 **GaitNet** (基于LSTM)。

[image]

#### **A. 视频流 vs. 静态图 (Video-based vs. GEI-based)**

- **对比对象：** CNN-LB是一种基于 **GEI (步态能量图)** 的方法，即将视频压缩成一张图来处理。而表格中其他方法都是基于视频序列的。
- **结果：** 所有基于视频的方法（GaitSet, GaitNet, GaitPart）都显著超越了CNN-LB。
- **结论：** 这证明了直接处理**原始步态序列 (Raw gait sequence)** 能比单一的能量图提取出更细粒度、更具辨识度的信息。

#### **B. 效率与性能的双赢 (vs. GaitSet)**

- **对比对象：** GaitSet 是该领域的标杆模型。
- **结果：** GaitPart 在拥有类似骨干网络的情况下，性能明显优于 GaitSet。
- **关键优势：** 作者特别指出，**GaitPart 的参数量仅为 GaitSet 的一半左右**。
- **结论：** 这通过实验证明了 **FConv (焦点卷积)** 和 **MCM (微动捕捉)** 模块的设计比单纯的集合操作更高效、更精准。

#### **C. 建模方式的胜利 (vs. GaitNet)**

- **对比对象：** GaitNet 使用了“自编码器 + 多层 LSTM”的结构。
- **结果：** GaitPart 在各种行走状态下（尤其是复杂的背包和换衣场景）都取得了更好的表现。
- **结论：** 这印证了作者之前的假设：针对步态这种周期性运动，GaitPart 的**局部短时建模 (MCM)** 比 LSTM 的**长时序列建模**更有效。

> Table 3:特别关注表格最下方的 CL #1-2 (换衣场景)。这是最难的任务。
> 
> - CNN-LB 只有 54.0%，GaitNet 只有 58.9%，GaitSet 提升到 70.4%，**GaitPart 达到了 78.7%**，优势非常明显。

### 2\. OU-MVLP 上的对比实验 (泛化能力)

为了验证模型在超大规模数据上的表现，作者使用了包含一万多人的 OU-MVLP 数据集。

[image]

#### **A. 刷新 SOTA**

- **结果：** 如 Table 4 所示，GaitPart 在各个视角下的平均Rank-1准确率达到了**88.7%**，超越了 GaitSet (87.1%) 和 GEINet (35.8%)，达到了新的 SOTA 水平。

#### **B. 关于 "88.7%" 的重要说明**

作者特别解释了一个数据上的细节：**现象：** 准确率最高也到不了 100%。**原因：** 这是因为测试集中有一些受试者**缺失了对应的样本**（即有的 Probe 找不到对应的 Gallery），这部分被算作错误。**校正后：** 如果排除掉这些本来就没有数据的无效样本，GaitPart 的实际平均准确率应该高达 **95.1%**。

### 总结 (Key Takeaways)

这段实验分析传递了三个核心信息：

1. **更准：** 在困难场景（如换衣 CL）下，GaitPart 的识别率大幅领先。
2. **更轻：** 相比于强大的对手 GaitSet，GaitPart 用一半的参数量做到了更好的效果。
3. **更稳：** 在全球最大的数据集上，GaitPart 依然保持了最先进的性能，证明它不是“过拟合”特定小数据集的模型。

* * *

## **4.3 消融实验 Ablation Study**

所谓的消融实验，就是把模型“拆开”来测，通过控制变量法，分别验证模型中每一个组件（FConv 和 MCM）到底有没有用，以及参数该怎么设才最好。所有的实验都是在 **CASIA-B** 数据集上进行的。实验结果分为 **空间维度 (FConv)** 和 **时间维度 (MCM)** 两个部分。

### 1\. 验证 FConv 的有效性 (空间维度)

这一组实验主要探究：**我们到底应该把图像切成几块 (**$p$**值设为多少)？**

[image]

**Table 5** (Group A)，作者对比了四种不同的设置方案：

- **方案 A-a (1-1-1)：**所有的 Block 中 $p=1$ 。这意味着**完全不使用 FConv**（相当于普通卷积）。
- **方案 A-c (1-4-8)：**随着层数加深，逐渐增加切分数量（1 -> 4 -> 8）。这是最终选定的方案。
- **方案 A-d (2-4-8)：**在第一层就开始切分 ($p=2$ )。

#### **实验结论：**

1. **FConv 确实有效：** 只要使用了 FConv 的方案（A-b, c, d），整体效果都比完全不用的 A-a 要好。这证明了限制感受野、关注局部细节的思路是对的。
2. **浅层不要切 (Shallow layers need global view)：** 对比 A-c 和 A-d，发现在第一层（Block1）如果使用 FConv ($p=2$ )，效果反而下降了。
    
    - **原因分析：** 浅层网络主要负责提取边缘和轮廓。如果过早切分，会破坏相邻部位之间的边缘连续性信息。
3. **深层切分对困难场景更有用：** 随着 $p$ 值增大（方案 c vs. b），在 **BG (背包)** 和 **CL (换衣)** 这种复杂场景下的提升非常明显。这意味着局部细节特征对于抗干扰非常关键。

### 2\. 验证 MCM 的有效性 (时间维度)

这一组实验主要探究：**微动捕捉模块 (MCM) 内部的结构该怎么设计？**

**Table 6** (Group B)，作者对比了 MTB 的组合、注意力机制以及池化方式：

[image]

```
表6.消融研究，B组。控制条件：使用和不使用MTB1或MTB2，使用和不使用MTB中的注意机制以及TP的不同实例。结果是11个视图的平均rank1准确度，不包括相同视图的情况。
```

#### **A. 窗口大小与多尺度 (MTB1 vs. MTB2)**

- **对比：** B-b (只用 MTB1, 窗口3) vs. B-c (只用 MTB2, 窗口5) vs. **B-a (两者都用)**。
- **结论：** **双管齐下 (B-a)** 的效果最好。
- **原因：** 这证明了多尺度设计（Multi-scale）是有效的。同时捕捉“快一点的微动”和“慢一点的微动”，能获得更丰富的特征。

#### **B. 注意力机制 (Attention)**

- **对比：** B-a (有 Attention) vs. B-d (无 Attention)。
- **结论：** 去掉注意力机制后，准确率全面下降。
- **原因：** 证明了引入通道注意力机制确实能帮助模型自动筛选出最具代表性的微动特征。

#### **C. 池化方式 (TP：Max vs. Mean)**

这是一个非常关键的对比，验证了之前的理论假设。

- **对比：** B-a (Max Pooling) vs. B-e (Mean Pooling)。
- **结果：** 使用 Mean Pooling (B-e) 的效果**最差**，甚至不如不用 FConv 的版本。
- **原因：** 这验证了 **"Ground Principle"**。步态是周期性的，取最大值 (Max) 能保证特征在周期内的稳定性；而取平均值 (Mean) 会因为视频长度不是周期的整数倍而引入噪声。

### Summary

通过这一系列严谨的消融实验，作者确定了 **GaitPart 的最终形态 (Baseline)**：

1. **FPFE 设置：** 采用 **1-4-8** 的渐进式切分策略（浅层不切，深层细切）。
2. **MCM 设置：** 并行使用 **MTB1 和 MTB2**（多尺度），开启**注意力机制**，并使用 **Max Pooling** 进行时序聚合。

这也解释了为什么 GaitPart 能在前面的 SOTA 对比中取得那么好的成绩：每一个组件的设计细节都经过了实验数据的反复推敲和验证。

* * *

## **4.4 时空分析 Spatio-temporal Study**

这个实验非常有意思，作者试图回答一个在步态识别领域争论已久的问题：**到底是我们长的样子（静态外观）重要，还是我们走路的动作（时序动态）重要？**为了搞清楚 GaitPart 到底学到了什么，作者设计了一组“打乱顺序”的对照实验。

### 1\. 实验背景与目的

- **背景：** 之前的很多方法（如 GaitSet）认为时间顺序不重要，只要有一堆剪影就能识别。但 GaitPart 引入了 MCM 专门捕捉微动，理论上应该很依赖时间顺序。
- **目的：** 把输入的视频帧**打乱顺序 (Shuffle)**，看看模型的准确率会不会暴跌。
    
    - 如果暴跌 $\rightarrow$ 说明**时间信息 (Temporal)** 极其重要。
    - 如果没跌多少 $\rightarrow$说明 **外观信息 (Appearance)** 才是主力。

### 2\. 实验设置 (Group C)

**Table 7** ，作者设置了三种情况：

1. **C-a (Train Shuffle)：**训练时把帧打乱，测试时正常。
2. **C-b (Baseline)：**训练和测试都正常（不打乱）。这是标准对照组。
3. **C-c (Test Shuffle)：**训练时正常，测试时突然把帧打乱。

[image]

```
表7.时空研究，C组。控制条件：在训练/测试阶段对输入序列进行排序/打乱。结果是11个视图的平均rank1准确度，不包括相同视图的情况。
```

### 3\. 结果与洞察

#### **A. 静态外观是“基本盘”**

- **现象：** 即使打乱了顺序（C-a 和 C-c），模型的准确率虽然有所下降，但**并没有崩盘**。
    
    - 例如在 NM（正常行走）条件下，C-b 是 96.2%，C-c 即使乱序也有 92.5%。
- **结论：** 这说明**静态外观特征 (Static appearance features)** 确实在步态识别中扮演了至关重要的角色。即便没有动作连贯性，光靠身形轮廓也能识别出大部分人。

#### **B. 时间信息是“杀手锏”**

- **现象：** 请特别注意 **CL (换衣)** 这一列的数据。
    
    - 正常顺序 (C-b)：**78.7%；**测试乱序 (C-c)：**65.1%**
    - **跌幅巨大：** 在换衣场景下，准确率掉了 13% 以上。
- **结论：**
    
    - 当人穿了大衣（CL），身形轮廓（外观）发生了巨大改变，这时候“看外形”就不准了。
    - 此时，**动作模式（时间信息）** 就成了救命稻草。GaitPart 通过 MCM 捕捉到的微动特征，在外观受干扰时提供了关键的鲁棒性。

### 总结

这个实验完美地为论文画上了句号：GaitPart 之所以强，是因为它“两条腿走路”：既利用了外观特征保住了基本准确率，又利用 MCM 提取的时间特征解决了换衣、背包等复杂场景下的识别难题。

* * *

# 5 **总结 Conclusion**

## 1\. 核心洞察 (The Novel Insight)

作者再次强调了这篇论文的**出发点**：

- **观点：** 人体的不同部位（头、身、腿）在行走时，无论是**视觉外观**还是**运动模式**，都有着本质的区别。
- **推论：** 因此，不能“一锅端”，**每个部位都需要拥有自己专属的时空建模方式**。

## 2\. 提出的解决方案 (The Proposed Solution：GaitPart)

基于上述洞察，作者提出了 **GaitPart** 模型，它由两个核心组件构成，分别解决空间和时间问题：

- **空间上：FPFE (帧级局部特征提取器)**
    
    - **核心技术：** **FConv (焦点卷积)**。
    - **目的：** 增强**局部特征的细粒度学习** (enhance the fine-grained learning of part-level features)。也就是让模型能更精细地看清每个部位的细节。
- **时间上：TFA (时序特征聚合器)**
    
    - **核心技术：** 若干个并行且参数独立 (dependent) 的 **MCM (微动捕捉模块)**。
    - **目的：** 提取**局部的、短距离的时空表达** (extract the local short-range spatio-temporal expressions)。也就是不再关注冗长的整个周期，而是专注于捕捉瞬间的微小动作。

> *注：这里的 "dependent" 呼应了前文提到的 "Part-dependent"，意思是指这些模块是*“因部位而异”*的，即头部有头部的参数，腿部有腿部的参数，互不混用。*

## 3\. 最终成就 (Final Achievement)

- **验证：** 在两个最权威的公共数据集 **CASIA-B** 和 **OU-MVLP** 上进行了实验。
- **结论：** 实验结果充分证明了 **GaitPart 整体模型** 以及其 **所有组件** (FConv, MCM) 的优越性。

**核心回顾：**

1. **痛点：** 传统方法忽略了人体部位差异和局部微动。
2. **方法：** 提出了 **GaitPart**。
    
    - 空间上：用 **FConv** 切分人体，细看局部。
    - 时间上：用 **MCM** 捕捉短时微动。
3. **结果：** 在 CASIA-B 和 OU-MVLP 上取得了 SOTA，特别是换衣场景提升巨大。

- **它的核心贡献**在于打破了以往将人体视为整体进行长序列分析的惯性思维。
- **它的成功秘诀**在于“**分块 (Part-based)**”和“**微动 (Micro-motion)**”：
    
    1. 用 **FConv** 像手术刀一样把人体切开，精细化处理空间特征。
    2. 用 **MCM** 像放大镜一样聚焦时间轴，捕捉最具辨识度的短时微动。

这种设计使得模型在处理**换衣、背包**等复杂场景时表现出了惊人的鲁棒性，同时保持了较低的参数量和极高的准确率。

* * *

* * *

# 代码解析

这段的核心思想：**将特征图在空间上水平切分（Horizontal Pooling），然后对每个切分后的部位（Part）独立进行时间维度的微动捕捉（MCM/TFA）。**以下是核心概念与论文的对应关系：

## A. 核心模块：`TemporalFeatureAggregator` (TFA)[3.3 时序特征聚合器TFA - GaitPart](zotero://note/u/JSDKKYSD/?section=3.3%20%E6%97%B6%E5%BA%8F%E7%89%B9%E5%BE%81%E8%81%9A%E5%90%88%E5%99%A8TFA)

这是论文中 **MCM (Micro-motion Capture Module)** 的具体实现。

1. **Part-Dependent (部位独立性)**：
    
    - **代码体现**：`self.conv1d3x1 = clones(conv3x1, parts_num)`。
    - **原理**：使用 `clones` 函数复制了16个（设parts_num=16）结构相同但权重不共享的小网络。
    - **目的**：头部、躯干、腿部的运动模式不同，独立的参数能让模型分别学习各部位特有的微动模式。
2. **Short-range Spatio-temporal Features (短时特征)**：
    
    - **代码体现**：`BasicConv1d(..., kernel_size=3)` 和 `AvgPool1d(3)`。
    - **原理**：相比于 LSTM 或 全局 3D-CNN，这里只在很小的时间窗口（3帧或5帧）内进行卷积和池化。
    - **目的**：捕捉行走时的瞬间动作变化（微动）。
3. **Attention Mechanism (注意力机制)**：
    
    - **代码体现**：`scores3x1 = torch.sigmoid(logits3x1)` 然后 `feature * scores`。
    - **原理**：使用 1D 卷积计算出一个 0~1 之间的权重 mask，乘在特征上。
    - **目的**：增强关键的微动特征，抑制噪声。
4. **MTB1 & MTB2 (多尺度设计)**：
    
    - **代码体现**：分别定义了 kernel=3 和 kernel=5 (padding=2) 的两组操作，最后相加 `feature3x1 + feature3x3`。
    - **目的**：融合不同时间跨度的特征，提高鲁棒性。

## B. 数据流 (Data Flow) [3.1 Pipeline - GaitPart](zotero://note/u/JSDKKYSD/?section=3.1%20Pipeline)

1. **Backbone**：`[n, c, s, h, w]` -> 每一帧单独提取图像特征。
2. **HPP (Horizontal Pooling)**：`[n, c, s, h, w] -> [n, c, s, p]` -> 每一帧被水平切成 $p$ 个条带，降维成向量。
3. **TFA (MCM)**：`[n, c, s, p] -> [n, c, p]` -> 在 $s$ (时间) 维度上进行微动提取和Max Pooling，时间维度消失。
4. **Head (SeparateFCs)**：`[n, c, p] -> [n, c_out, p]` -> 将特征映射到用于计算 Triplet Loss 的特征空间。

## C. 工具类说明

- `SetBlockWrapper`：OpenGait 的工具，用于把 5D 数据 `[n, c, s, h, w]` 里的 `n` 和 `s` 合并成 `n*s`，喂给只接受 4D 输入的普通 CNN (ResNet 等)，处理完后再变回 5D。
- `HorizontalPoolingPyramid`：对应论文中的“水平切分”，通常包含 Global Max Pooling + Global Avg Pooling。
- `SeparateFCs`：对应论文最后的 "Separate FC layers"，也是 Part-dependent 的，每个部位特征有自己专属的全连接层。

* * *

## 1\. Micro-motion Capture Module (MCM)：是如何提取短时序信息的？这与 GaitSet 的粗暴池化有何不同？

MCM 的设计初衷是捕捉“微动”（Micro-motion），即短时间内的动作变化，而非长时间的统计信息。

### **MCM 的提取机制**

从提供的代码 `TemporalFeatureAggregator` 类可以看出其工作流程：

1. **滑动窗口 (Sliding Window)**：使用 `kernel_size=3` 或 `5` 的 1D 卷积/池化在时间轴 $s$ 上滑动。这对应了论文中提到的提取 $i$ 帧及其邻域帧 $(i-r, ..., i+r)$ 的信息。
2. **微动模版 (Template Function)**：`self.avg_pool3x1(x) + self.max_pool3x1(x)`。它在小窗口内结合了平均值和最大值，捕捉局部的动态变化。
3. **注意力重加权 (Attention Re-weighting)**： `scores3x1 = torch.sigmoid(logits3x1)`。它通过一个小型的卷积网络计算出权重，强调那些最具辨识度的微动瞬间，抑制静止或噪声帧。

### **与 GaitSet 的区别**

- **GaitSet (粗暴池化 / Set-based)**：
    
    - **机制**：将步态序列视为无序的**集合 (Set)**。它直接对**整个视频序列**进行全局最大池化 (Global Max Pooling)，把 $t$ 帧直接压缩成 1 帧。
    - **缺陷**：虽然简洁，但它**完全丢失了时间顺序**和帧与帧之间的精细动态变化（即丢失了“迈步过程”中的微动信息）。
- **GaitPart (MCM)**：
    
    - **优势**：MCM 关注的是**局部短时依赖 (Local short-range dependencies)**。它不把整个视频压扁，而是看“这一帧和前后两帧”的关系，从而保留了动作发生的动态细节。

## 2\. 如何将特征图在高度（Height）方向切分为多个 Parts 的？为什么 OpenGait 中很多模型会有 p (parts) 这个维度？

### **代码中的切分实现**

在 `GaitPart` 类的 `forward` 函数中，切分操作是由 `HorizontalPoolingPyramid` (HPP) 模块完成的：

```
# 此时out的维度是[n, c, s, h, w](Batch, Channel, Time, Height, Width)
out = self.Backbone(sils)
# 关键步骤：HPP模块将(h, w)空间维度转换为p(parts)维度
# 输出out的维度变为[n, c, s, p]
out = self.HPP(out)
```

虽然代码没有给出 `HorizontalPoolingPyramid` 的内部实现逻辑是：

1. **水平切片**：将高度 $H$ 平均切分成 $p$ 份。
2. **池化降维**：对每一份（Part）进行 Global Max + Avg Pooling，将该部分的 $H_{part} \times W$ 压缩为一个数值。
3. **结果**：原来的图像空间 $(H, W)$ 就变成了一个向量 $(p)$，代表人体从头到脚的 $p$ 个水平条带的特征。

### **为什么要有** `p` (parts) 这个维度？

这源于步态识别的一个核心假设：**Part-based Assumption (分块假设)**。

1. **部位差异性**：论文反复强调，人体不同部位（头、躯干、腿）在行走时的**运动模式**和**视觉外观**截然不同。例如，头是相对静止的，手是摆动的，腿是交替跨越的。
2. **独立建模 (Part-dependent)**：引入 `p` 维度后，模型可以针对每个部位使用**独立不共享的参数**。
    
    - 代码证明：在 `TemporalFeatureAggregator` 中，使用了 `clones(conv3x1, parts_num)`。这意味着如果有 16 个 part，就有 16 个独立的卷积网络，第 1 个网络专门学“头怎么动”，第 16 个网络专门学“脚怎么动”，互不干扰。

总结来说，`p` 维度是步态识别中为了实现**精细化局部建模**而特有的维度设计。