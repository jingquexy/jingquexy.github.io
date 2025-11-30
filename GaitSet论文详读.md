# GaitSet论文详读

* * *

## 2 两大主流技术路线

> “The existing works have tried to tackle the problem from two aspects. They either regard gait as a single image or regard it as a video sequence. Methods in the first category ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=1)) compress all gait silhouettes into one image, i.e., a gait template, for gait recognition [1], [2], [3], [4], [5], [6], [7]. Although various existing gait templates [5], [6], [7] encode information as abundantly as possible, the compression process omits significant features such as temporal information and fine-grained spatial information. To address this issue, the methods in the second category extract features directly from the original gait silhouette sequences [8], [9], [10]. These methods preserve more temporal information but would suffer a significant degradation when an input contains discontinuous frames or has a frame rate different from the training dataset.” ([Chao 等, 2021, p. 3467](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=1))

基于您提供的这段文献综述，我将为您深入展开步态识别（Gait Recognition）中这两大主流技术路线的原理、优劣势，特别是重点解析为什么“基于图像的方法”会丢失**细粒度空间信息**，以及“基于视频的方法”面临的具体挑战。

这段话实际上描述了步态识别技术从**手工特征/模板时代**向**深度学习/序列建模时代**演进的过程。

### 1\. 第一类：基于图像的方法 (Gait as Image / Template-based)

这种方法的核心思想是“**降维**”。它试图将一个三维的动作（长、宽、时间）压缩成一个二维的图像。

#### **代表技术：步态能量图 (Gait Energy Image, GEI)**

这是最经典的例子。做法很简单：把一个人走路的所有帧（剪影），全部“叠加”在一起取平均值。

[image]

```
从左上到右下是 CASIA-B 步态数据集中受试者完成的一个阶段的剪影。
```

#### **为什么说它“丢失了细粒度空间信息”？**

请想象一下，把一部电影的所有画面叠在一起变成一张图：

- **边缘模糊化：** 在走路过程中，手和脚是在不断摆动的。当我们把几十帧图像叠加，手脚的运动轨迹会变成一片模糊的灰色区域（Ghosting effect）。原本清晰的手指轮廓、鞋子的形状、膝盖弯曲的精确角度（这些就是**细粒度空间信息**），都被平均化处理抹平了。
- **瞬间特征消失：** 在某一帧中，人的衣服可能有一个特定的褶皱，或者身体有一个微小的倾斜。这些瞬间的高频空间细节，在“平均”操作下完全消失。
- **信息混叠：** 手臂在躯干前摆动时，手臂的像素和躯干的像素叠加在一起，导致很难分清哪里是手、哪里是身体。

#### **优缺点总结**

- **优点：** 简单高效，把视频变成了图片，可以使用标准的2D-CNN（如ResNet）进行处理，计算量小。
- **缺点：** 极度依赖“周期性”，一旦发生了**遮挡**，或者每一帧的**细微空间特征**对识别很重要（比如区分两个体型极其相似的人），这种模糊的模板就失效了。

### 2\. 第二类：基于视频序列的方法 (Gait as Video Sequence)

这种方法的核心思想是“**保真**”。它认为步态是一个时空连续体，不应该为了省事而压缩时间维度。

#### **处理方式**

直接将原始的剪影序列（比如 30 帧的 $H \times W$二 值图像）作为输入。

- **3D-CNN：** 使用三维卷积核，同时在空间（长宽）和时间（帧）上滑动，提取特征。
- **LSTM / RNN：** 先提取每一帧的空间特征，再通过循环神经网络记忆时间上的变化。
- **Set-based (如 GaitSet)：** 把序列看作一堆图的“集合”，虽然不强调顺序，但保留了每一帧的独立特征。

#### **对“细粒度空间信息”的保留**

在这种方法中，每一帧图像都保持独立。神经网络可以“看清”第5帧里腿迈出的确切高度，也能“看清”第10帧里手臂摆动的边缘。这种**帧级别的像素级细节**得以保留，模型可以学习到更微小的体态差异。

#### **文献提到的挑战解析**

1. **“容易受到外部因素的影响”：**
    
    - 因为保留了所有细节，模型也同时保留了“噪音”。
    - 比如：如果一个人背包或穿大衣，基于视频的方法会清晰地看到大衣下摆的摆动。模型可能会错误地把“大衣的摆动”当作这个人的步态特征，而不是学习他“腿的运动”。相比之下，GEI因为取了平均，偶尔出现的衣服摆动反而被模糊掉了，有时反而更鲁棒。
2. **“难训练” (Hard to Train)：**
    
    - **参数量爆炸：** 3D-CNN 的参数量远大于 2D-CNN，计算成本呈指数级上升。
    - **梯度问题：** 处理长序列（如RNN）容易出现梯度消失或梯度爆炸，导致模型收敛困难。
    - **数据需求：** 为了训练这种复杂的深层网络，需要海量的带标注步态数据，而现有的步态数据集（如CASIA-B, OUMVLP）相比人脸数据集来说，规模依然较小。

### 3\. 总结与直观对比

为了帮您理清这两个概念，我们可以用看书来做比喻：

<table><tbody><tr><td data-colwidth="181" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">特性</span></strong></p></td><td data-colwidth="483" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">基于图像 (步态模板)</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">基于视频 (序列信息)</span></strong></p></td></tr><tr><td data-colwidth="181" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">操作</span></strong></p></td><td data-colwidth="483" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">把整本书的内容</span><strong><span style="background-color: rgba(0, 0, 0, 0)">叠印在一张纸上</span></strong><span style="background-color: rgba(0, 0, 0, 0)">。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">逐页阅读</span></strong><span style="background-color: rgba(0, 0, 0, 0)">整本书。</span></p></td></tr><tr><td data-colwidth="181" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">结果</span></strong></p></td><td data-colwidth="483" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">看到的是密密麻麻的文字重影（GEI）。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">看到每一页清晰的文字和故事情节。</span></p></td></tr><tr><td data-colwidth="181" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">细粒度空间信息</span></strong></p></td><td data-colwidth="483" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">丢失。</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 无法看清具体的字形和标点。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">保留。</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 可以看清每一个字的笔画。</span></p></td></tr><tr><td data-colwidth="181" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">时间信息</span></strong></p></td><td data-colwidth="483" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">丢失。</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 不知道哪句话是先说的。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">保留。</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 知道故事的起承转合。</span></p></td></tr><tr><td data-colwidth="181" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">代价</span></strong></p></td><td data-colwidth="483" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">省力，一眼看完。</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">费力，需要大脑（算力）高速运转，且容易被某一页的污渍（外部干扰）分心</span></p></td></tr></tbody></table>

* * *

## **2.1.1 第一类：基于模板的方法，Template-based Approaches**

### 1\. 第一步：模板生成 (Template Generation) —— “造图”

这是数据预处理阶段，目的是**压缩信息**。

- **原始输入：** 一段视频（很多帧画面）。
- **操作流程：**
    
    1. **背景移除 (Background Removal)：** 把人从复杂的背景里扣出来，只留下黑白剪影。
    2. **对齐 (Alignment)：** 把每一帧里的人都挪到画面正中间，大小缩放一致。
    3. **像素级操作 (Pixel level operators)：** 这是关键。通常是将几十帧图像叠加在一起取平均值。
- **最终产出：** 一张单张图像，最著名的就是 **GEI (Gait Energy Image，步态能量图)**。
    
    - *注：CGI (Chrono-Gait Image) 是另一种把时间信息编码成彩色的图，但原理类似。*

> **直观理解：** 这一步就像把一套连环画叠在一起，透写成一张画。好处是数据量变小了（从几十张图变成一张图），坏处就是之前提到的“丢失了细粒度信息”。

### 2\. 第二步：模板匹配 (Template Matching) —— “认图”

有了这张“压缩图”（模板）后，如何用它来识别这是谁？作者列举了从传统机器学习到深度学习的演进过程。

#### A. 特征提取 (Extract Representation)

不能直接比对像素，要提取数学特征。

- **传统方法：** 使用 **CCA** (典型相关分析) 或 **LDA** (线性判别分析)。这些数学方法的目的是在数据中找到“最能区分不同人”的那些特征维度。
- **深度学习方法：** 用卷积神经网络（CNN）去读这张GEI图，提取特征向量。

#### B. 相似度度量 (Measure Similarity)

- **核心逻辑：** 算距离。计算“测试样本的特征”和“数据库里张三的特征”之间的**欧氏距离**。距离越近，是同一人的概率越大。
- **解决“视角”问题 (View Transformation)：**
    
    - 难点：同一个人的步态，从侧面看（90度）和从正面看（0度），生成的 GEI 完全不一样。
    - 解决方法：文中提到的 **VTM** 和 **ViDP** 都是为了解决这个问题。它们试图通过数学变换，把“侧面的图”转换成“正面的图”，或者把所有角度的图都映射到一个“视角不变的潜在空间” (View-invariant latent space)中进行比对。

### 3\. 这段话在论文中的作用

作者写这段话的目的是为了**铺垫（立靶子）**。

通过详细描述这个流程，作者暗示了这种主流方法的局限性：

1. **信息有损：** 在“模板生成”这一步，为了压缩成一张图，不仅丢了时间信息，还让细粒度的空间信息（如具体的肢体边缘）变得模糊。
2. **依赖视角变换：** 后续的 VTM/ViDP 等步骤非常复杂，专门为了修补“视角不同导致GEI不同”这个缺陷。

逻辑连接：

正因为这一套流程（先压成图 -> 再费力去解视角问题）太繁琐且丢失信息，所以作者才在后文提出了 GaitSet：直接把原图当成一个集合扔进去，既保留了信息，又不需要复杂的视角转换模型。

* * *

## **2.1.2 第二类：基于序列的方法，Sequence-based Approaches**

如果不“拍扁”成一张图（GEI），而是直接处理视频序列，前人都有哪些高招？作者在这里快速回顾了五种主流的技术路线。我们可以把它们看作是**为了从视频中榨取更多信息而尝试的不同“工具”**。

以下是详细的拆解与技术点拨：

### 1\. 3D CNN流派 (The 3D CNN-based approaches)

- **原文引用：**“extract temporal-spatial information using 3D convolutions;” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
- **核心逻辑：** 暴力美学。既然数据是长（$H$）x 宽（$W$）x 时间（$T$）的立方体，那就直接用三维的卷积核（3D Kernel）在上面滑动。
- **优势：** **时空不分家**。它能同时捕捉到“空间上的形状”和“时间上的运动”。比如，它能直接理解“腿部（空间）正在向前迈（时间）”这个整体动作。
- **代价：** 计算量极其巨大，参数多，极其难训练。

### 2\. 骨骼流派 (Skeleton-based)

- **原文引用：** “utilized human skeletons to learn gait features robust to the change of clothing” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
- **核心逻辑：** 透视眼。不看剪影（轮廓），而是先用算法（如OpenPose）把人的关节关键点（骨架）提取出来，只分析骨架的运动。
- **为什么这么做？** 为了**抗干扰**。
    
    - 如果你穿了一件厚大衣，你的剪影（轮廓）会变得很宽，看起来像个胖子。
    - 但你的**骨骼节点**（膝盖、脚踝）的位置是不变的。
    - 因此，作者称其对“服装变化（Change of clothing）”具有鲁棒性。

### 3\. 序列融合流派 (LSTM/Attention)

- **原文引用：** “fused sequential frames by LSTM attention units” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
- **核心逻辑：** 像读文章一样读视频。
    
    - **LSTM (长短期记忆网络)：** 这种网络擅长记住“上文”。它会一帧一帧地看，把前面的动作信息存在“记忆”里，用来理解后面的动作。
    - **Attention (注意力机制)：** 并不是每一帧都重要（比如人站着不动的帧就没用）。注意力机制让模型学会“只盯着最有信息量的几帧看”。

### 4\. 图神经网络流派 (Graph-based / STGAN)

- **原文引用：**“uncover the graph relationships between gait frames” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
- **核心逻辑：** 建立关系网。
    
    - 它不把视频看作简单的线性序列（1->2->3），而是把帧看作图中的节点。
    - 通过**STGAN（时空图注意力网络）**，模型去挖掘帧与帧之间更复杂的潜在关系（比如第1帧的左脚和第10帧的右脚之间可能存在某种对称关系）。这是一个比较高级且抽象的建模方式。

### 5\. 局部切片流派 (Part-based model) —— **重点关注**

这是目前步态识别中非常流行且有效的一个方向（GaitSet 后来也结合了这个思想）。

- **原文引用：**“capture spatial-temporal features of each part” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
- **核心逻辑：** **切开看**。
    
    - 人体的运动是不均匀的。头部的运动（主要是平移）和腿部的运动（大幅度摆动）完全不同。
    - 如果把全身混在一起提取特征，腿部的剧烈运动可能会掩盖头部的微小特征。
    - **做法：** 把特征图在水平方向切成几条（Strips），比如头部一条、躯干一条、腿一条，分别提取特征，最后再拼起来。
- **进阶优化（Refinement）：**
    
    - **问题：** 切片太多会导致特征冗余（相邻的切片可能长得很像）。
    - **解决：** 文中提到的 *[b]* 使用了“两阶段训练策略”来压缩特征，去掉废话，只留精华（Compact features）。

### 这段话的“潜台词”

作者把这些方法列出来，其实是在展示一个“痛点列表”，为自己的方法（GaitSet）铺路：

1. **3D CNN** 太重了，难训练。
2. **LSTM** 必须依赖顺序，如果丢帧或乱序就崩了。
3. **骨骼法** 依赖姿态估计的准确度，如果骨架提错了，后面全错。
4. **Part-based** 虽然好，但容易有冗余。

**GaitSet 的高明之处在于：** 它不需要 LSTM 的严格顺序，不需要3D CNN的庞大算力，也不需要骨骼提取的预处理，直接把它们当成一堆图（Set），甚至可以结合 Part-based 的切片思想（GaitSet 代码中确实用了 Horizontal Pyramid Mapping），从而集百家之长。

* * *

## **2.2 GaitSet的“灵感来源”和“理论基础”**

作者并没有凭空创造“把数据看作集合（Set）”这个想法，而是借鉴了 **3D 点云处理（Point Cloud）** 领域的经典网络 **PointNet**。

为了让你读懂这段话，我们需要跨界理解一下“点云”和“步态”的惊人相似之处。

### 1\. 灵感源头：PointNet 与点云 (Point Cloud)

- **原文引用：** “The initial goal for using unordered sets was to address point cloud tasks in the computer vision domain based on PointNet.” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
- **背景知识：**
    
    - **图像（Image）：** 是规则的网格（Grid）。像素点是整齐排列的，必须知道谁在 $(0,0)$ ，谁在 $(0,1)$ 。
    - **点云（Point Cloud）：** 是激光雷达扫出来的 3D 数据。它只是一堆 $(x, y, z)$坐标 的列表。
    - **核心特性：** **无序性**。一个杯子的 3D 点云，无论你在列表中先写杯底的点，还是先写杯口的点，它在空间中组成的形状都是那个杯子。

**PointNet 的伟大之处**就在于：它设计了一种神经网络，**不在乎你输入的点的顺序**，只要这堆点都在，它就能认出这是个杯子。

### 2\. 为什么要用 Set？为了避免“量化噪声”

- **原文引用：** “avoid the noise introduced by quantization” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
- **这是什么意思？**
    
    - **在 3D 领域：** 以前为了处理点云，人们会把空间切成一个个小方块（体素，Voxel，类似 Minecraft）。这叫“量化”。但这会带来**精度损失**（细粒度信息丢失），因为两个靠得很近的点可能会被合并进同一个方块里。
    - **PointNet 的做法：** 直接处理原始的点（Raw Points），不切方块，所以保留了**最高精度的空间信息**。
- **迁移到步态识别（Gait）：**
    
    - 作者认为，把步态视频强制压缩成 GEI（一张图），或者强制对齐时间轴，其实也是一种人为的“量化”或“规整化”，这会引入噪声或丢失信息。
    - **GaitSet 的逻辑：** 像 PointNet 处理原始点一样，直接处理**原始的帧集合**，从而保留最原始、最完美的视觉特征。

### 3\. 数学基础：置换不变性 (Permutation Invariance)

- **原文引用：** “characterized of the permutations using invariant functions.” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
- 核心概念：
    
    如果要让模型把“一组图”看作一个整体，必须使用一种特殊的数学函数，满足：$f(A, B, C) = f(C, A, B) = f(B, C, A)$， 即：输入顺序改变，输出结果不变。
    
- GaitSet 的实现：通常使用 Max Pooling（最大池化） 来实现这一点。
    
    - 假设你提取了3帧的特征值分别是：3、9、5。
    - 无论顺序怎么排（3,9,5 或 5,3,9），取最大值结果永远是 **9**。
    - 这就是文中提到的“Invariant Functions”，它是 Set-based 方法的数学灵魂。

### 4\. 总结：这段话的逻辑链

1. **现状：** 大家以前都只玩规则的数据（视频序列、图像网格）。
2. **跨界：** 3D 视觉里的 PointNet 证明了“处理无序集合”不仅可行，而且效果极好（因为它不丢细节）。
3. **推广：** 这种思想已经被推荐系统、图像描述等领域借用了。
4. **填补空白：** 但是！在步态识别领域，除了我们（作者团队），还没人深入研究过这个方向。

### 这里的关键洞察

作者在这里通过引用 PointNet，实际上是在为 GaitSet 能够**保留细粒度空间信息**提供理论背书：

> 既然 PointNet 通过把点云看作 Set，避免了 Voxelization 带来的精度损失；
> 
> 那么 GaitSet 通过把步态看作 Set，也能避免 GEI/Sequence 带来的时空信息损失。

[image]

```
GaitSet的框架。SP代表集合池化。梯形代表卷积和池化区块，同一列中的区块具有相同的配置，用大写字母的矩形所示。请注意，虽然MGP中的区块与主pipeline中的区块配置相同，但参数仅在主pipeline中的区块之间共享，而不与MGP中的区块共享。HPP代表水平金字塔池化。
```

## 3.1 核心数学建模（Problem Formulation）

它主要做了两件事：

1. **理论定义**：从统计学角度解释为什么可以把步态看作“集合”而忽略时间。
2. **公式推导**：给出了整个 GaitSet 模型的宏观数学公式 $f_i = H(G(F(\cdot)))$ 。

下面我为你逐层拆解这段形式化的描述：

1\. 核心假设：步态即“分布” ( $\mathcal{P}_i$)

作者为了证明“不看时间顺序”是合理的，引入了统计学概念。

- **原文符号：**
    
    - $y_i$ ：第 $i$个人的身份（Label）。
    - $\mathcal{P}_i$ ：第 $i$个人 独有的**步态分布（Distribution）**。
    - $x_{i}^j$ ：第 $i$个人的第$j$张剪影图。
- 通俗解释：
    
    作者假设每个人走路的姿态，都服从一个独特的概率分布 $\mathcal{P}_i$ 。
    
    - 这就好比：张三走路的姿态是一个“抽奖箱” ($\mathcal{P}_{\text{张三}}$ )。
    - 我们可以把视频里的每一帧剪影，看作是从这个箱子里**随机抽出**的一张张卡片 $x \sim \mathcal{P}_i$。
    - **结论：** 既然是随机抽样，那么**抽出来的卡片顺序就不重要了**。只要抽出来的卡片够多，就能描绘出这个“抽奖箱”的特征。

这就是为什么原文说：“can be regarded as a set of n silhouettes” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3)) （可以看作 $n$个 剪影的集合）。

### 2\. 宏观公式：三步走战略

论文给出了全篇最重要的公式 (1)：$f_i = H(G(F(x_{i}^1), F(x_{i}^2), ..., F(x_{i}^n)))$

这个公式描述了数据流动的三个阶段，分别对应模型的三个模块。我们可以把它想象成一条**流水线**：

#### **第一步：F(Frame-level Feature Extraction)**

- **输入：** 单张剪影图 $x$ 。
- **功能：** **“单图感知”**。用卷积神经网络 (CNN) 处理每一张图。
- **操作：** 就像这里有$n$张照片，我有$n$个完全一样的放大镜，同时去观察这$n$张照片，提取出每张图的特征。
- **结果：** 得到$n$个独立的特征向量（Frame-level features）。

#### **第二步：G (Set Pooling / Permutation Invariant)**

- **输入：**上一步得到的$n$个特征向量。
- **功能：“无序聚合”**。这是一个满足**置换不变性**的函数。
- **操作：** 这就是前文提到的PointNet思想，通常是 **Max Pooling**。无论这$n$个特征谁先谁后，扔进这个函数，出来的结果是一样的。
- **结果：** 将 $n$个特征融合成了**1 个**集合级特征（Set-level feature）。这代表了把“视频”压缩成了“特征块”。

#### **第三步：H(Horizontal Pyramid Mapping)**

- **输入：** 集合级特征。
- **功能：** **“多尺度精修”**。这就是上一轮对话中解释的 **HPM**（切片操作）。
- **操作：** 把聚合好的特征切成不同尺度的条带（Strips），映射到最终的判别空间。
- **结果：** 最终的身份特征 $f_i$ ，用来算欧氏距离进行识别。

### 3\. 数据维度的变化 (The Input Tensor)

原文最后提到：

> “The input X_i is a tensor with four dimensions: set dimension, image channel dimension, image height dimension, and image width dimension.” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))

为了方便你写代码，我们可以把这个维度的变化过程具象化：

1. **输入Input:** $[S, C, H, W]$
    
    - $S$ : Set Size (集合大小，即视频的帧数，比如30帧)
    - $C$ : Channel (通常是 1，即灰度图)
    - $H, W$ : 图片高宽 (如64, 44)
2. **经过**$F$ **(CNN Backbone):**
    
    - 变成了 $[S, C', H', W']$ 。
    - 注意：这里$S$还在，每一帧特征依然独立。
3. **经过**$G$**( Set Pooling):**
    
    - 在$S$这个维度上进行Max Pooling。
    - 变成了 $[1, C', H', W']$ 。
    - **关键点：** 此时$S$维度消失了，多帧变成了一个整体特征。
4. **经过**$H$**( HPM):**
    
    - 进行切片和全连接映射。
    - 最终变成 $[1, D]$ (特征向量)，用于分类或比对。

### 总结

这段话用数学语言确立了GaitSet 的合法性：它告诉审稿人，我们不是在“瞎搞”乱序，而是基于统计分布假设。我们将整个识别过程解耦为：

**1.看 (**$F$ **)  2.聚 (**$G$ **)  3.切 (**$H$ **)**

接下来，论文应该会详细展开介绍 $G$( Set Pooling) 和 $H$(HPM) 的具体设计细节。

* * *

## **3.2 Set Pooling (集合池化)**

它的任务很简单：**“如何把一堆独立的帧特征（Set），压缩成一个用来代表这个人的整体特征（Set-level feature）？”**

如果说上一节是把“书”拆成了“页”，这一节就是把“页”读完后写出一句“中心思想”。

以下是详细拆解：

### 1\. 核心任务与两大约束 (Constraints)

**公式定义** $z = G(V)$

- $V$**(Input):** 很多张图的特征集合 $\{v_1, v_2, ..., v_n\}$ 。
- $z$**(Output):** 一个单一的特征张量，代表这个人。
- $G$ **:** 集合池化函数。

#### **必须满足的两大条件**

这是设计$G$函数的死规定：

1. **置换不变性 (Permutation Invariant):**
    
    - 不管输入的帧顺序怎么乱（$1,2,3$还是 $3,1,2$ ），输出的$z$必须一模一样。
    - 这就是为什么不能用 RNN/LSTM的原因。
2. **任意基数 (Arbitrary Cardinality):**
    
    - 不管输入是 10 帧还是 100 帧，输出$z$的尺寸必须固定。
    - 这样才能喂给后续的全连接层。

### 2\. 作者尝试的三种“武器”

作者并没有只用一种方法，而是探讨了三种不同复杂度的池化策略，试图找到最优解。

[image]

```
集合池化（SP）的七个不同实例。1×1C和cat分别代表1×1卷积层和连接操作。这里，n表示集合中的特征图数量，c、h和w分别表示特征图的通道数、高度和宽度。a.三种基本统计SP和两种联合SP。b.像素级注意力SP。c.帧级注意力SP。
```

#### **第一种：基础统计函数 (Basic Statistical Functions) —— “简单粗暴”**

最直接满足上述两个条件的方法就是统计学运算。作者对比了三个：

1. **Max(**$\cdot$**)：**取所有帧中对应位置的最大值。
    
    - *特点：***最保留细节**。只要某一帧里这儿有个衣角，Max就会把它留下来。这对保留**细粒度空间信息**最有效。
2. **Mean(**$\cdot$**)：**取平均值。
    
    - *特点：*类似 GEI，容易模糊，抗噪性一般。
3. **Median(**$\cdot$**)：**取中位数。
    
    - *特点：*鲁棒性强，能去除偶尔出现的极值（比如闪烁噪声）。

#### **第二种：联合函数 (Joint Functions) —— “我全都要”**

既然 Max 留细节，Mean 留大局，不如合起来用？

- **Eq (3) 加法版：** $z = \text{max}(V) + \text{mean}(V) + \text{median}(V)$ 。
- Eq (4) 融合版：$z = 1\times1\_Conv(\text{cat}(\text{max}(V), \text{mean}(V), \text{median}(V)))$
    
    - 先在通道维度把三个结果拼起来（Concat）。
    - 然后用一个$1\times1$卷积层（相当于把通道揉在一起）来学习，到底 Max 重要还是 Mean 重要（自动学习权重）。

#### **第三种：注意力机制 (Attention) —— “有的放矢”**

作者借鉴了当时最火的Attention机制，设计了两种方案：

1. **像素级注意力 (Pixel-wise Attention):**
    
    - **逻辑：** 并不是特征图上每个像素点都重要（比如背景像素就不重要）。
    - **做法：** 先算一个全局统计信息，用它来生成一张“掩膜图（Attention Map）”，告诉模型每一帧的哪个像素该保留，哪个该丢弃。最后再做 Max Pooling。
2. **帧级注意力 (Frame-wise Attention):**
    
    - **逻辑：** 并不是每一帧都重要（比如被遮挡的帧、或者姿态不清晰的帧应该权重低）。
    - **做法：** 给每一帧$v_j$ 打分（Score, $a_j$ ）。清晰的帧给高分，模糊的给低分。
    - **公式：** $z = \sum a_j v_j$（ 加权求和）。

### 3\. 实验结论（剧透）

虽然作者设计了这么多复杂的 Attention 和 Joint Functions，但在文末的实验部分（Section 4）有一个非常**反直觉且重要**的发现：

> “although different instantiations of SP do influence the performances, they do not produce significant differences” ([Chao 等, 2021, p. 3469](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=3))
> 
> 虽然搞得很花哨，但其实效果差别并没有那么大。

在实际应用和后续的代码复现中（如 OpenGait 库），大家发现：

简单的 Max Pooling 往往效果最好，或者性价比最高。

原因分析：

步态识别最怕的是“丢细节”。

- **Mean** 会把细节抹平（Blur）。
- **Attention** 虽然好，但如果训练数据不够多，容易过拟合。
- **Max** 最忠实地记录了“这个特征在视频中出现过吗”。只要出现过一次（比如脚踢到了最高点），Max 就会把它抓住。这完美契合了**细粒度空间信息**的保留需求。

* * *

## **3.3** Horizontal Pyramid Mapping **(**水平金字塔映射**)**

Section 3.3 主要讲述了 GaitSet 如何解决“看清细节”的问题。如果说前一步的 **集合池化 (Set Pooling)** 解决了“时间”上的压缩问题，那么 **水平金字塔映射 (Horizontal Pyramid Mapping, HPM)** 就是为了解决“空间”上的**细粒度 (Fine-grained)** 和 **多尺度 (Multi-scale)** 问题。

简单来说，HPM 的作用就像是拿着不同倍数的放大镜，把人体从头到脚切成片，一块一块地仔细看。

[image]

```
水平金字塔映射的结构。
```

以下是结合论文原理与代码实现的通俗解读：

### 1\. 为什么要“切片”？(Splitting into Strips)

在传统的卷积神经网络（CNN）中，处理到最后通常会做一个**全局平均池化 (Global Average Pooling, GAP)**，把整张图变成一个数值向量。

- **痛点**：这样做会丢失**空间位置信息 (Spatial Information)** 。
    
    - **通俗解释**：如果你把整个人“平均”了一下，网络就分不清“头”在哪里，“脚”在哪里。一个头大脚小的人和一个头小脚大的人，平均后的数值可能是一样的。
- **灵感来源**：作者借鉴了**行人重识别 (Person Re-ID)** 领域的**水平金字塔池化 (Horizontal Pyramid Pooling, HPP)** 。
- **做法**：把特征图在**高度 (Height)** 方向上横着切几刀，强迫网络单独看“头部区域”、单独看“躯干区域”、单独看“腿部区域”。

### 2\. 什么是“金字塔”？(Pyramid Structure)

作者并没有只切一种尺寸，而是构建了一个**多尺度 (Multi-scale)** 的金字塔结构，既看整体，又看局部 。

- **论文设定**：
    
    - 假设有$S$个 尺度 (Scales)。
    - 对于第$s$个尺度，把图片切成$2^{s-1}$ 份。
- **具体切法 (**$S=5$**的 情况)**：
    
    - **Scale 1 (**$2^0=1$ **)**：不切，看**全身 (Global)**。
    - **Scale 2 (**$2^1=2$ **)**：切成 2 半，看**上半身、下半身**。
    - **Scale 3 (**$2^2=4$ **)**：切成 4 份，看**头、胸、腿、脚**。
    - ... 以此类推，直到切成 16 份。
- **代码中的数字 "31"**：
    
    - 这就是代码配置中经常出现的特征维度的来源。
    - 总条带数 = $1 + 2 + 4 + 8 + 16 = \mathbf{31}$个 条带。

### 3\. 如何提取特征？(Dual Statistical Pooling)

切完片后，如何把每一片图像变成特征向量？GaitSet 采用了“双保险”策略。

- **公式**：$Feature = \text{Global Max Pooling} + \text{Global Avg Pooling}$。
- **通俗解释**：
    
    - **Max Pooling (最大池化)**：提取**最显著的特征**。比如这一块区域里，脚抬得最高的那一瞬间的轮廓。
    - **Avg Pooling (平均池化)**：提取**背景或平均状态**。比如这一块区域里，腿部的平均粗细。
- **为什么合起来？** 论文指出，两者结合的效果比单独使用任何一个都好。

### 4\. 核心创新：独立映射 (Independent Mapping / FCs)

这是 GaitSet 相比于传统 Re-ID 方法最大的改进。传统的做法是在池化后接一个 $1 \times 1$的 **卷积层 (Convolutional Layer)**，但这在步态识别里不够好。

- **论文原话**：我们不使用卷积层，而是为每一个池化后的特征使用**独立的全连接层 (Independent Fully Connected Layers, FCs)** 。
- **为什么不用卷积？**
    
    - 卷积核是**参数共享 (Shared Parameters)** 的。这意味着它用同一套逻辑（权重）去分析头部和脚部。
    - **通俗解释**：头部的运动通常很平稳（平移），而脚部的运动很剧烈（大幅摆动）。如果强迫网络用“同一双眼睛”去看这两个完全不同的部位，网络会“精神分裂”，学不精。
- **GaitSet 的做法 (Independent FCs)**：
    
    - **专人专职**：给这 31 个条带，分配 **31 个完全独立的全连接层** 。
    - 第 1 个 FC 专门学全身；第 31 个 FC 专门学脚踝。大家互不干扰，各自在自己的**判别空间 (Discriminative Space)** 里做到最好。

* * *

## 3.4 **多层全局流水线 (Multilayer Global Pipeline, MGP)**

如果说主干网络是在“看每一帧”，那么 MGP 就是在“看整个视频的演变”，并且它不仅看最后的结果，还一边看一边把中间的细节吸收到自己的理解中。

以下是结合论文原理与代码实现的通俗解读：

### 1\. 核心动机：为什么需要 MGP？

在深度卷积网络中，有一个基本常识：**不同深度的层，看到的“风景”是不一样的**。

- **浅层网络 (Shallow Layers)**：感受野（Receptive Field）小，关注**局部、细粒度**的信息。
    
    - *例子*：这里有一条边缘，那里有一个纹理，鞋子的颜色。
- **深层网络 (Deep Layers)**：感受野大，关注**全局、粗粒度**的信息。
    
    - *例子*：这是一个人的腿，他在向前走，整体姿态是弯腰的。

**痛点**：通常的 CNN 只在最后一层提取特征。这意味着我们虽然得到了高级语义（是谁），但可能丢掉了浅层的细节（鞋子样式、衣角摆动），而这些细节在步态识别中往往很重要。

**MGP 的目标**：把浅层的“细节”和深层的“语义”**融合**在一起，让最终的特征既有宏观的体态，又有微观的细节。

### 2\. 架构设计：双流并进 (Dual Streams)

为了实现这个目标，作者设计了两条并行的流水线：

1. **主流水线 (Main Pipeline)**：处理 **帧级 (Frame-level)** 特征。
    
    - 它负责盯着每一张图看，从 `Set Block 1` 走到 `Set Block 3`。
2. **MGP 流水线 (MGP)**：处理 **集合级 (Set-level)** 特征。
    
    - 它不看单张图，而是看经过**集合池化 (Set Pooling, SP)** 压缩后的“全视频总结”。

融合机制 (The Addition)：这就像是接力跑。主流水线每跑完一段（例如 Block 1），就把自己提取到的特征压缩一下（SP），然后加 (Add) 到 MGP 流水线中。MGP 带着这份信息继续跑下一段，再接收新的信息。

> “we add the set-level features extracted by different layers to MGP.” ([Chao 等, 2021, p. 3470](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=4))

### 3\. 代码实现的透视 (Mapping to Code)

在 `gaitset.py` 的 `forward` 函数中，MGP 的逻辑体现得淋漓尽致。请看以下逐行解析：

```
# 1. 第一阶段 (浅层细节)
outs = self.set_block1(sils)           # 主流：提取第1层帧级特征 (细节多)
gl = self.set_pooling(outs, ...)       # 转化：压缩成集合级特征
gl = self.gl_block2(gl)                # MGP流：开始处理，准备进入下一层

# 2. 第二阶段 (中层特征融合)
outs = self.set_block2(outs)           # 主流：提取第2层帧级特征
# 【关键动作】融合！把主流当前的特征池化后，直接加到MGP流 (gl) 中
gl = gl + self.set_pooling(outs, ...)
gl = self.gl_block3(gl)                # MGP流：带着融合后的信息继续处理

# 3. 第三阶段 (深层语义融合)
outs = self.set_block3(outs)           # 主流：提取第3层帧级特征
# 【关键动作】再次融合！此时gl里已经包含了第1层和第2层的信息，现在加上第3层
gl = gl + self.set_pooling(outs, ...)
```

**代码中的变量含义**：

- `outs`：代表 **Main Pipeline**，始终保持 `[n, s, c, h, w]` (有时间维度 $s$)。
- `gl`：代表 **MGP Pipeline**，始终保持 `[n, c, h, w]` (时间被压缩了，是全局特征)。
- `gl = gl + ...`：这就是论文中提到的类似 ResNet 的残差连接思想，将不同层级的信息累加。

### 4\. 独特的 HPM 处理

论文最后提到一个细节：

> “the HPM that executes after MGP does not share parameters with the HPM that executes after the main pipeline.” ([Chao 等, 2021, p. 3470](zotero://select/library/items/GCIUWLN4)) ([pdf](zotero://open-pdf/library/items/BKFIIXDV?page=4))

这意味着，最后输出时，我们会有两组特征：

1. **主干特征 (**`feature1`)：来自 `outs`，代表最深层的语义。
2. **MGP 特征 (**`feature2`)：来自 `gl`，它是浅层、中层、深层信息的混合体。

```
feature1 = self.HPP(outs)  # 处理主干特征
feature2 = self.HPP(gl)    # 处理 MGP 特征 (参数不共享，因为是两个独立的函数调用)
feature = torch.cat([feature1, feature2], -1) # 最后拼起来
```

### 5\. 直观类比：人类认知 (Human Cognition)

作者用人类认知做了一个精彩的类比：

- **主流水线 (Main Pipeline) $\approx$ 看轮廓 (Profile)**：
    
    - 就像我们认人，第一眼看的是整体轮廓，“哦，这是个高个子”。
- **MGP 流水线 (MGH/MGP) $\approx$ 看动态细节 (Walking Movements)**：
    
    - MGP 就像是我们仔细观察，“他的左脚有点微跛，摆臂幅度很大”。它保留了那些在深层抽象中容易丢失的动态细节。

### 总结

**Multilayer Global Pipeline (MGP)** 是 GaitSet 实现“既见森林（全貌），又见树木（细节）”的关键。

- **结构上**：它是主干网络旁边的一条**平行通道**。
- **操作上**：它通过**侧向连接 (Lateral Connections)** 不断吸收主干网络在不同深度的信息。
- **效果上**：它让最终的步态特征极其丰富，包含了从边缘纹理到整体姿态的所有层级信息。

* * *

## 3.5 损失函数和训练策略

Section 3.5 详细阐述了 GaitSet 是如何“学习”的。为了让模型既能**分类**（认出是谁），又能**比对**（判断两个样本是否属于同一人），作者结合了深度学习中两类最经典的损失函数，并提出了一种分阶段的训练策略。

### 1\. 两大“核心教官” (Loss Functions)

GaitSet 并没有只使用单一的监督信号，而是结合了 **分类 (Classification)** 和 **度量学习 (Metric Learning)** 两种思维。

#### **A. 交叉熵损失 (Cross Entropy Loss)** —— 负责“身份分类”

- **角色**：把步态识别当作一个标准的**多分类问题**（比如训练集有 100 个人，就是一个 100 分类任务）。
- **原理**：衡量“预测的概率分布”和“真实标签”之间的差距。
- **在 GaitSet 中的特殊应用**：
    
    - 由于 GaitSet 使用了 HPM（水平金字塔映射），输出的不是一个特征，而是 **多个**（$2 \times \sum 2^{s-1}$ ）独立的特征条带。
    - **计算方式**：不仅是对最后拼起来的大特征算 Loss，而是对**每一个细分条带 (Strip)** 都单独计算一次Cross Entropy Loss，最后把它们**加起来 (Sum)** 作为总损失。
    - **目的**：强迫每一个局部切片（比如光看“脚部”或光看“头部”）都必须具备独立识别身份的能力。

#### **B. 三元组损失 (Triplet Loss)** —— 负责“拉近拉远”

- **角色**：这是度量学习（Metric Embedding Learning）的核心，最早用于人脸识别（FaceNet）。它的目标不是“分类”，而是学习“距离”。
- **核心逻辑**：
    
    - **拉近 (Pull)**：让语义相似的点（同一个人）在空间中靠得更近。
    - **推远 (Push)**：让语义不同的点（不同的人）在空间中离得更远 。
- **三元组结构 (**$r = (a, b, g)$ **)** ：
    
    - **Anchor (**$a$ **)**：基准样本（张三的视频 A）。
    - **Positive (**$b$ **)**：正样本（张三的视频 B）。
    - **Negative (**$g$ **)**：负样本（李四的视频 C）。
- 公式解析：$L(r) = ReLU(\xi + D_{a,b} - D_{a,g})$
    
    - $D_{a,b}$ ：同类距离（越小越好）。
    - $D_{a,g}$ ：异类距离（越大越好）。
    - $\xi$( Margin)：边界。即我们希望“异类距离”至少要比“同类距离”大出一个 $\xi$的 距离，才算合格 。
- **具体版本 (Batch All,** $BA_+$ **)**：
    
    - GaitSet 使用的是 **Batch All** 策略，意味着在一个 Batch 里，计算所有可能的合格三元组的 Loss，而不是随机挑几个。这能挖掘出更多“难样本”进行学习。

### 2\. 独特的“组合拳”策略 (Training Strategy)

作者之前的版本（AAAI-19）只用了 Triplet Loss，虽然效果好，但为了追求极致，这次采用了 **“先分类，后精修”** 的策略。

- **Step 1: 交叉熵热身 (Cross Entropy First)**
    
    - 先只用 Cross Entropy Loss 训练网络，目的是让网络快速**收敛 (Converge)** 。
    - 让模型先学会“大概怎么区分这些人”。
- **Step 2: 三元组精修 (Triplet Refinement)**
    
    - 在模型收敛后，加入 Triplet Loss（通常使用**更小的学习率**）。
    - **目的**：在已经分得清的基础上，进一步优化特征空间，让特征分布更加紧凑和具有判别性（Discriminant metric space）。

### 3\. 代码视角的总结

如果对应到代码实现，这个策略通常体现为损失函数的权重配置和学习率调度：

1. **多尺度计算**：代码会在 `for` 循环中遍历 HPM 输出的62个（或 31x2 个）特征向量，分别计算 Loss 并求和。
2. 联合损失：虽然论文强调分步，但在实际代码配置中，往往是两种 Loss 同时存在，公式如下：$L_{total} = L_{cross\_entropy} + L_{triplet}$
3. **HPM 的意义**：再次强调，Loss 是施加在**每一个切片**上的。这意味着每一个 `Independent FC` 都在被单独监督，确保了模型不仅看整体，也看局部细节。

* * *

## 3.6 **实战演习**

它规定了我们该如何把数据喂给模型去训练（Training），以及训练好之后，如何用它来抓坏人（Testing/Inference）。

为了适应三元组损失（Triplet Loss）的需求，GaitSet 采用了一种特殊的**采样策略**，这在代码实现中非常关键。

### I. 训练阶段 (Training Phase)：

在训练时，我们不能随便抓一把数据扔进去，必须精心设计 **批次 (Batch)** 的结构。

1\. $P\times K$采 样器 (The Sampler)

- **为什么要这么做？**
    
    - 这是为了配合 **三元组损失 (Triplet Loss)**。
    - 在一个 Batch 里，必须保证每个人都有“同伴”（用于拉近距离），同时也必须有“外人”（用于推远距离）。
    - 如果不强制 $p \times k$ ，随机采样很可能导致一个 Batch 里全是不同的人，根本没法学“同类相似性”。
- **代码透视**：在 `OpenGait` 或 PyTorch 的 DataLoader 中，这通常被称为 `BalancedSampler` 或 `TripletSampler`。

#### 2\. 数据来源的约束

- **原文指出**：虽然 GaitSet 理论上支持把不同视频里的帧混在一起作为一个集合输入，但在**训练阶段**，一个样本（Sample）只包含**来自同一个视频序列 (One Sequence)** 的轮廓图 。
- **原因**：为了保持训练数据的纯净和简单，防止过拟合或引入复杂的标签噪声。

### II. 测试阶段 (Testing Phase)：检索与比对

测试过程本质上是一个 **“以图搜图”** 或 **“特征检索”** 的过程。

#### 1\. 角色定义

- **Query (**$Q$ **) / Probe**：**查询样本**。就好比监控拍到的嫌疑人视频，我们想知道他是谁。
- **Gallery (**$G$ **)**：**注册集 / 底库**。就好比警察局里已经存好档案的嫌疑人数据库。
- **目标**：拿着 $Q$ 去 $G$ 里面找，看谁最像。

#### 2\. 特征生成 (Feature Concatenation)

- **输入**：把 $Q$和  $G$里 的样本都扔进训练好的 GaitSet 模型。
- **HPM 输出**：模型会吐出很多个细分特征（HPM 那些被切成条带的特征）。
    
    - 具体数量是 $2 \times \sum_{s=1}^{S} 2^{s-1}$个 多尺度特征。
- **拼接 (Concatenation)**：
    
    - 把这几十个特征条带首尾相连，**拼接**成一个超级长的向量，作为这个人的 **最终特征表示 (Final Representation,** $\mathcal{F}$ **)**。
    - 这就是代码中 `torch.cat([feature1, feature2], -1)` 的结果。

#### 3\. 距离计算与排名 (Ranking)

- **算距离**：计算 $Q$的 特征向量 $\mathcal{F}_{\mathcal{Q}}$与 底库中每一个样本 $\mathcal{F}_{G}$的  **欧氏距离 (Euclidean Distance)** 。
    
    - 距离越小，代表两个人越像。
- **Rank-1 Accuracy (首选识别率)**：
    
    - 把底库里所有人按距离从小到大排队。
    - 如果**排在第一位 (Rank-1)** 的那个人，ID 刚好和 $Q$的  ID 一样，就算识别成功。
    - Rank-1 Accuracy 就是识别成功的百分比。

### 总结 (Summary)

- **训练时**：使用 $P\times K$策 略，确保模型能在一个 Batch 里同时看到“同类”和“异类”，从而学会区分人。
- **测试时**：是一个**全库检索**过程。先把切片特征**拼接**成大向量，然后算**欧氏距离**找最近邻。

* * *

## 3.7 后处理降维 (Post Feature Dimension Reduction)

### 效果与代码映射

- **效果**：通过这种方法，可以将特征维度压缩到非常小（例如 1024 甚至更低），同时保持**有竞争力的识别准确率 (Competitive Accuracy)**。
- **代码实现**：
    
    - 在 `OpenGait` 或其他开源复现中，这通常不是默认开启的（因为学术界更刷榜 Rank-1）。
    - 但在**工业落地**时，这是一个必不可少的模块。通常表现为一个独立的脚本：加载训练好的模型权重 -> 定义一个 Linear 层 -> 冻结 Backbone -> 训练 Linear 层 -> 导出用于推理的“Backbone + Linear”小模型。

### 总结 (Summary)

Section 3.7 实际上是在说：**GaitSet 的原始特征太“重”了，不适合大规模实时检索。我们通过由繁入简的“蒸馏”手段（后处理线性投影），在保留识别精度的同时，极大地提升理和检索的速度。**

* * *

## 4.1.1 **CASIA-B**

对于研究 GaitSet 或任何步态识别模型来说，理解 CASIA-B 的数据结构和评测协议（Protocol）是跑通代码和看懂实验结果的第一步。

### I. 数据集解剖：CASIA-B 是什么？

CASIA-B 是一个多视角、多状态的步态数据集，它的核心特点是“小而精”**，涵盖了步态识别中最难的两个变量：视角变化和**外部状态干扰。

#### 1\. 基础构成 (Basic Composition)

- **对象 (Subjects)**: 共有 **124** 个不同的人（ID 编号 001-124）。
    
- **视角 (Views)**: 每个人都在 **11 个不同的角度**下被拍摄。
    
    - 范围：$0^\circ$到  $180^\circ$ ，每隔 $18^\circ$一个机位。
    - $0^\circ/180^\circ$是 正对/背对镜头（信息量最少），$90^\circ$是 完全侧面（步态信息最清晰）。
- **总视频数**: $124 \text{人} \times (6+2+2) \text{序列} \times 11 \text{视角} = \mathbf{13,640}$个 视频 。

#### 2\. 三种行走状态 (Walking Conditions)

这是 CASIA-B 最大的难点，也是 GaitSet 想要征服的目标。每个人有 10 组行走序列：

- **NM (Normal Walking)**: **正常行走**。
    
    - 每人 **6** 组序列 (NM #1-6) 。
    - 这是最标准的步态数据。
- **BG (Walking with a Bag)**: **背包行走**。
    
    - 每人 **2** 组序列 (BG #1-2)。
    - *挑战*：背包会破坏人体的轮廓，改变重心。
- **CL (Wearing a Coat/Clothing)**: **穿着外套行走**。
    
    - 每人 **2** 组序列 (CL #1-2)。
    - *挑战*：这是最难的（GaitSet 论文中通常 CL 的准确率最低）。厚外套会完全遮挡手臂摆动和腿部线条，导致轮廓发生剧烈变化。

### II. 三大实验协议 (Training/Test Partitions)

由于 CASIA-B 官方发布时没有规定哪个是训练集，哪个是测试集，学术界（和 GaitSet 论文）约定俗成了三种切分方式。

[image]

你在阅读 `OpenGait` 配置文件（如 `config.yaml`）时，经常会看到 `ST`, `MT`, `LT` 的选项，指的就是这个：

<table><tbody><tr><td data-colwidth="115" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">协议名称</span></strong></p></td><td data-colwidth="264" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">中文名</span></strong></p></td><td data-colwidth="189" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">训练集 (Training)</span></strong></p></td><td data-colwidth="185" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">测试集 (Testing)</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">特点</span></strong></p></td></tr><tr><td data-colwidth="115" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">ST</span></strong></p></td><td data-colwidth="264" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">小样本训练</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> (Small-sample)</span></p></td><td data-colwidth="189" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">前 </span><strong><span style="background-color: rgba(0, 0, 0, 0)">24</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 人 (001-024)</span></p></td><td data-colwidth="185" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">后 </span><strong><span style="background-color: rgba(0, 0, 0, 0)">100</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 人</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">训练数据极少，容易过拟合，现在较少使用。</span></p><p></p></td></tr><tr><td data-colwidth="115" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">MT</span></strong></p></td><td data-colwidth="264" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">中样本训练</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> (Medium-sample)</span></p></td><td data-colwidth="189" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">前 </span><strong><span style="background-color: rgba(0, 0, 0, 0)">62</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 人 (001-062)</span></p></td><td data-colwidth="185" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">后 </span><strong><span style="background-color: rgba(0, 0, 0, 0)">62</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 人</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">五五开，以前常用 。</span></p><p></p></td></tr><tr><td data-colwidth="115" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">LT</span></strong></p></td><td data-colwidth="264" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">大样本训练</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> (Large-sample)</span></p></td><td data-colwidth="189" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">前 </span><strong><span style="background-color: rgba(0, 0, 0, 0)">74</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 人 (001-074)</span></p></td><td data-colwidth="185" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">后 </span><strong><span style="background-color: rgba(0, 0, 0, 0)">50</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> 人</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p></p><p><strong><span style="background-color: rgba(0, 0, 0, 0)">当前主流标准</span></strong><span style="background-color: rgba(0, 0, 0, 0)">。数据量最大，最能体现深度学习模型的性能。</span></p><p></p></td></tr></tbody></table>

> **注意**：训练集和测试集的 ID 是**互不重叠 (No overlap)** 的 。这意味着模型在测试时，看到的都是从未见过的陌生人，必须学习到通用的步态特征（泛化能力），而不能死记硬背 ID。

### III. 评测逻辑：注册与查询 (Gallery vs. Probe)

在测试阶段（Testing Phase），模型是如何计算准确率的？GaitSet 遵循了标准的 **“注册-查询”** 模式。

#### 1\. 注册集 / 底库 (Gallery Set)

- **定义**：警察局里的“标准档案”。
- **数据**：测试集中，每个人的**前 4 组正常行走 (NM #1-4)**。
- **逻辑**：我们默认已知每个人正常走路的样子。

#### 2\. 查询集 / 探针集 (Probe Set)

- **定义**：监控摄像头抓拍到的“未知视频”，需要去底库里匹配。
- **数据**：测试集中剩下的 6 组视频，被分为三个难度的子集：
    
    1. **NM Subset (简单)**: NM #5-6。用正常走路去匹配正常走路。
    2. **BG Subset (中等)**: BG #1-2。用背包走路去匹配正常走路。
    3. **CL Subset (困难)**: CL #1-2。用穿大衣走路去匹配正常走路。

#### 3\. 实际意义

这种设计模拟了真实场景：我们通常只有嫌疑人正常走路的档案（Gallery NM），但嫌疑人作案时可能背了包（Probe BG）或者换了衣服（Probe CL）。**Rank-1 Accuracy** 就是看模型能不能在这种情况下依然把他认出来。

### 代码中的映射 (Mapping to Code)

在 `OpenGait` 等代码库中，这部分逻辑通常不需要你手动写，而是通过配置文件控制。例如：

- **数据加载**：通常有一个 `partition.json` 文件，里面写死了哪些 ID 属于 Train，哪些属于 Test（对应 ST/MT/LT）。
- **评估代码**：在 `evaluation.py` 中，会有逻辑判断：
    
    - `if "nm-01" in seq_type: is_gallery = True`
    - `if "cl-01" in seq_type: is_probe = True`
    - 然后计算 Probe 和 Gallery 之间的特征距离。

* * *

## 4.1.2 **OU-MVLP**

如果说 CASIA-B 是用来测试模型“抗干扰能力”（背包、穿大衣）的试金石，那么 OU-MVLP 就是用来测试模型 **“在大规模人群中捞针”** 能力的终极考场。

### I. 数据集概览：步态识别的“大数据”时代

**OU-MVLP (OU-ISIR Multi-View Large Population Dataset)** 是目前全球最大的公开步态数据集。它的核心特点就是**人多**

#### 1\. 惊人的规模

- **对象 (Subjects)**: 包含了 **10,307** 个不同的人。
- **对比**: 相比于 CASIA-B 的 124 人，OU-MVLP 的规模扩大了近百倍。这就好比从“村口识别”升级到了“城市级监控识别”。
- **深度学习的意义**: 只有在这样大规模的数据集上，深度学习模型（如 GaitSet）才能真正发挥出吃数据的优势，证明其泛化能力。

#### 2\. 视角设置 (14 Views)

不同于 CASIA-B 的 $0^\circ \sim 180^\circ$全 覆盖，OU-MVLP 的视角设计更为特殊，共 **14 个视角**：

- **前侧视角**: $0^\circ, 15^\circ, \dots, 90^\circ$ （共 7 个）。
- **后侧视角**: $180^\circ, 195^\circ, \dots, 270^\circ$ （共 7 个）。
- **特点**: 它主要关注人体的**左侧**和**右侧**，以及**正前**和**正后**，是对称分布的。

#### 3\. 序列设置 (Sequences)

- 每人每个视角下只有 **2 组序列**，编号为 **#00** 和 **#01**。
- 相比 CASIA-B (每人 10 组)，这里的单人数据量较少，但总人数极大。

### II. 实验协议 (Protocol)

OU-MVLP 的划分方式非常简单粗暴，就是 **五五开**。

#### 1\. 训练与测试划分

- **训练集 (Training)**: 使用 **5,154** 个人的数据。
- **测试集 (Testing)**: 使用剩余的 **5,154** 个人的数据 。
- **注意**: 这里同样遵循 ID 不重叠原则，模型在测试时面对的全是陌生人。

#### 2\. 注册与查询 (Gallery vs. Probe)

在测试集中，识别任务是如何进行的？

- **注册集 / 底库 (Gallery)**: 序列 **#01**。
    
    - 这组数据被存入系统作为档案。
- **查询集 / 探针 (Probe)**: 序列 **#00**。
    
    - 这组数据作为待识别的样本，去底库里匹配#01。

### III. 总结：OU-MVLP vs. CASIA-B

理解这两个数据集的区别，对于评估模型性能至关重要：

<table><tbody><tr><td data-colwidth="155" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">特性</span></strong></p></td><td data-colwidth="569" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">CASIA-B</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">OU-MVLP</span></strong></p></td></tr><tr><td data-colwidth="155" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">核心难点</span></strong></p></td><td data-colwidth="569" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">状态干扰</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> (背包、大衣)</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">数据规模</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> (万人级检索)</span></p></td></tr><tr><td data-colwidth="155" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">人数</span></strong></p></td><td data-colwidth="569" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">124 人 (小样本)</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">10,307 人</span></strong><span style="background-color: rgba(0, 0, 0, 0)"> (超大样本)</span></p></td></tr><tr><td data-colwidth="155" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">视角</span></strong></p></td><td data-colwidth="569" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">11 个 (<span class="math">$0^\circ \sim 180^\circ$</span> )</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">14 个 (<span class="math">$0^\circ \sim 90^\circ, 180^\circ \sim 270^\circ$</span> )</span></p></td></tr><tr><td data-colwidth="155" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">评估侧重</span></strong></p></td><td data-colwidth="569" style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">侧重模型对</span><strong><span style="background-color: rgba(0, 0, 0, 0)">外外观变化</span></strong><span style="background-color: rgba(0, 0, 0, 0)">的鲁棒性</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">侧重模型在</span><strong><span style="background-color: rgba(0, 0, 0, 0)">海量干扰项</span></strong><span style="background-color: rgba(0, 0, 0, 0)">下的辨识精度</span></p></td></tr></tbody></table>

在 GaitSet 的论文中，作者在这两个数据集上都取得了当时最好的成绩（SOTA），证明了该模型既能抗干扰（CASIA-B 实验），又能在大规模数据上跑得通（OU-MVLP 实验）。

* * *

## 4.2 **参数设置(Parameter Setting)**

### 1\. Input Data

- **图像尺寸 (Image Size)**: $64 \times 44$。
    
    - **代码对应**: 在 OpenGait 的配置文件中，通常对应 `resolution` 或 `img_size: [64, 44]`。这是步态识别的标准尺寸，既保留了轮廓信息，又不会让计算量过大。
- **集合大小 (Set Cardinality)**: **30**。
    
    - **含义**: 在**训练阶段**，模型每次并不是把一个人所有的视频帧都读进去，而是随机抽取 **30 帧** 作为一个集合（Set）进行训练。
    - **代码对应**: 对应配置文件中的 `sample_type: fixed_ordered` 或 `frame_num: 30`。
    - *注意*：测试时通常使用全部帧，不受此限制。

### 2\. Hyperparameters

- **优化器 (Optimizer)**: 使用 **Adam**。深度学习中最稳健的选择。
- **三元组间隔 (Margin)**: $0.2$。
    
    - **含义**: 在计算 Triplet Loss 时，我们要求“异类距离”必须比“同类距离”大出 $0.2$才 算达标。
    - **代码对应**: `loss_cfg` 中的 `margin: 0.2`。
- **HPM 尺度 (Scales)**: $S=5$。
    
    - **含义**: 金字塔切分层数为 5，对应切分份数为 $1, 2, 4, 8, 16$ ，总共 $31$个 条带。

### 3\. 针对不同数据集的模型变体

这是本节最关键的信息点：**模型的大小是根据数据集规模调整的**。

#### **A. CASIA-B 配置 (小模型)**

- **背景**: CASIA-B 数据量较小（124人）。
- **通道设置 (Channels)**:
    
    - Layer 1 & 2: **32**
    - Layer 3 & 4: **64**
    - Layer 5 & 6: **128**
- **目的**: 防止过拟合。参数少一点，模型不容易“死记硬背”。
- **计算量**: 8.6 GFLOPs。

#### **B. OU-MVLP 配置 (大模型)**

- **背景**: OU-MVLP 数据量巨大（1万人，是 CASIA-B 的 20 倍）
- **通道设置**:
    
    - Layer 1 & 2: **64** (翻倍)
    - Layer 3 & 4: **128** (翻倍)
    - Layer 5 & 6: **256** (翻倍)
- **目的**: 增加模型容量 (Capacity)。数据多了，需要更大的“脑容量”来记住更多样的步态特征。
- **代码对应**: 在 `gaitset.py` 的 `build_network` 中，通过 `model_cfg['in_channels']` 传入。如果是跑 OU-MVLP，这个列表会被配置为 `[64, 128, 256]`。

### 4\. 训练策略表 (Table 1 概览)

这张图片展示了论文中的 **Table 1**，它详细列出了 GaitSet 模型在不同数据集和不同训练策略下的**超参数设置 (Hyperparameters)**。对于复现论文结果（Reproduction）或者配置 OpenGait 的 `config.yaml` 文件来说，这张表就是“操作手册”。

[image]

```
OU-MVLP和CASIA-B的三种设置（CASIA-ST、CASIA-MT和CASIA-LT）：批量大小(BSÞ)、学习率(LRÞ)和训练迭代(Iter)。
使用交叉熵损失训练时，从训练集中随机抽取一个小批次。使用三元组损失训练时，小批次的组成如第 3.6 节所述。
```

以下是这张表的详细解读，我将其拆解为 **训练模式**、**采样策略** 和 **数据集差异** 三个维度来解释。

#### 1\. 三种训练模式 (The Three Rows)

表格的行（Rows）对应了论文 Section 3.5 和 4.5.2 中讨论的三种训练/损失函数策略：

##### **A. CE (Cross Entropy)**

- **含义**：仅使用 **交叉熵损失** 进行训练。这是为了让模型先学会“分类”，快速收敛。
- **配置特点**：
    
    - **Batch Size (**$B_S$ **)**：128。
    - **采样方式**：图片底部的注释说明了原因 —— *"When being trained with cross entropy loss, a mini-batch is randomly selected"*。因为分类任务不需要成对比较，所以直接随机抽 128 张图就行。

##### **B. Triplet Only**

- **含义**：仅使用 **三元组损失 (Triplet Loss)** 进行训练。这是 AAAI-19 原版论文的策略。
- **配置特点**：
    
    - **Batch Size (**$B_S$ **)**：变成了$p \times k$的 格式（如 $p=8, k=16$ ）。
    - **采样方式**：遵循 Section 3.6 的采样器，必须保证一个 Batch 里有 $p$个不同的人，每人有 $k$张图，以便构建三元组。
    - **总 Batch Size**：$8 \times 16 = 128$ ，和 CE 保持一致，但内部结构变了。

##### **C. Triplet Tune (Refinement)**

- **含义**：这是“先分类，后精修”策略的第二步。在 CE 训练收敛的基础上，加上 Triplet Loss 进行微调。
- **关键差异**：
    
    - **学习率 (**$L_R$ **)**：设为 **1e-5**。
    - **原因**：相比于主训练阶段的 `1e-4`，微调阶段的学习率缩小了**10倍**。这是为了防止大幅度的梯度更新破坏模型已经学好的特征，只是做微小的“打磨”。
    - **迭代次数 (Iter)**：从 70k-90k 减少到了 30k-60k，因为只需要跑较短的时间就能完成精修。

#### 2\. 数据集规模的差异 (The Columns)

表格的列（Columns）展示了模型配置如何随着数据量的增加而“扩容”。

##### **CASIA-B (ST / MT / LT)**

- **规模**：相对较小（训练集 24 ~ 74 人）。
- **Batch Size**：固定为 **128** ($8 \times 16$ )。
- **迭代次数 (Iter)**：随着数据量增加略微增加（ST 70k $\to$LT 90k）。

##### **OU-MVLP (The Giant)**

- **规模**：巨大（训练集 5000+ 人）。
- **Batch Size**：扩大到 **512**。
    
    - **结构**：$p=32, k=16$ 。
    - **解读**：为了在茫茫人海中找到有效的“难样本（Hard Negatives）”，必须在一个 Batch 里放入更多不同的人（$p$从 8 增加到 32）。
- **迭代次数**：激增至 **800k** (80万次)。
    
    - 这是 CASIA-B 的 10 倍以上，说明在大数据上训练模型非常耗时。
- **学习率调度 (Scheduler)**：
    
    - 引入了 **StepLR** 策略：在150k 次迭代后，学习率从 `1e-4` 降到 `1e-5`。这有助于模型在训练后期跳出局部最优解，更稳定地收敛。

#### 3\. 代码实现的映射 (Mapping to OpenGait)

使用 `OpenGait` 复现这个配置，通常会在 `config.yaml` 的 `solver` 或 `data` 部分看到这些参数：

```
# 对应Table 1中的CASIA-LT + Triplet Tune设置
data:
dataset: CASIA-B
sampler:
batch_size: 128     # Total Batch Size
sample_type: fixed_ordered  # 对应 p x k
type: TripletSampler
p: 8                # Table 1: p=8
k: 16               # Table 1: k=16
solver:
optimizer: Adam
lr: 0.00001           # Table 1: Triplet Tune L_R = 1e-5
max_iter: 60000       # Table 1: Iter = 60k
```

#### 总结

这张表的核心逻辑是：

1. **分阶段训练**：先用大火（`1e-4`, CE）把模型煮熟，再用文火（`1e-5`, Triplet）慢炖入味。
2. **因地制宜**：小数据（CASIA）用小 Batch（128），大数据（OU-MVLP）用大 Batch（512）和超长训练周期。

### 总结 (Summary)

这一节实际上告诉了你在复现代码时如何修改 `config.yaml`：

1. **改尺寸**: 确认输入是 $64 \times 44$ 。
2. **改抽样**: 训练时每样本抽 30 帧。
3. **改通道**: 跑 CASIA-B 用 `[32, 64, 128]`，跑 OU-MVLP 用 `[64, 128, 256]`。
4. **改 Loss**: Margin 设为 0.2。

* * *

## 4.3 Brief Introduction of Compared Methods

关于步态识别（Gait Recognition）领域中不同对比方法的综述。它主要回顾了在深度学习爆发初期及之前，解决“跨视角（Cross-view）”步态识别问题的几种主流技术路线。

为了帮助你更好地理解，我将这些方法归纳为三大类，并对每一个方法的核心机制、优缺点及专有名词进行详细的**展开讲解**。

### 1\. 传统投影与统计学习方法 (Traditional Projection & Statistical Methods)

这一类方法不依赖深度神经网络，而是通过数学变换将不同视角的数据映射到同一个空间。

#### **ViDP (View-invariant Discriminative Projection)**

- **核心概念：** 视角不变判别投影。
- **详细原理解读：**
    
    - 步态识别最大的难点在于：同一个人从 $0^\circ$走 过和从 $90^\circ$走 过，看起来完全不同。
    - ViDP 旨在寻找一个**潜在空间（Latent Space）**。它使用一个酉线性投影（Unitary Linear Projection）矩阵，将不同视角的步态模板（Templates）投影到这个公共空间中。
    - 在这个空间里，视角的差异被忽略（View-invariant），而人的身份差异被保留。
- **一句话总结：** 通过数学旋转/变换，把不同角度的图像“拉”到一个统一的平面上进行对比。

#### **CMCC (Correlated Motion Co-Clustering)**

- **核心概念：** 相关运动联合聚类。
- **详细原理解读：**
    
    - **第一步（Motion Co-clustering）：** 它不像一般方法那样看“整体”，而是看“局部”。它将不同视角下运动模式最相关的身体部位（如手臂摆动、腿部迈步）聚类到同一组。
    - **第二步（CCA - Canonical Correlation Analysis）：** 典型相关分析。这是一种统计方法，用于寻找两组变量之间相关性最大的线性组合。
    - CMCC 利用 CCA 最大化不同视角之间步态信息的**相关性**。
- **一句话总结：** 先把身体动起来像的部分归类，再用统计学方法确立它们在不同视角下的对应关系

### 2\. 基于 CNN 的早期深度学习方法 (Early CNN-based Methods)

这一类方法开始利用卷积神经网络（CNN）强大的特征提取能力，处理步态能量图（GEI）或视频序列。

#### **Wu et al. [7] 提出的三种模型**

这篇论文非常经典，它探索了 CNN 在步态识别中的三种不同用法：

1. **CNN-LB (Local Bottom-up):**
    
    - **输入：** 它是**成对**输入的。将两个步态序列的 GEI（步态能量图，即所有帧的平均图）作为两个通道（Channels）输入网络。
    - **机制：** 这是一个验证（Verification）模型。它的任务不是分类“这是谁”，而是判断“这两张图是不是同一个人”。
    - **结构：** 3层 CNN。
2. **CNN-3D:**
    
    - **输入：** 视频帧序列（不仅仅是静态图片）。
    - **机制：** 使用 **3D-CNN**。普通的 CNN (2D) 只能处理空间信息（长宽），3D-CNN 可以同时处理**时间信息**（帧与帧之间的运动）。
    - **操作：** 它在9个相邻帧上运行，最后取16个样本的预测平均值作为结果。这比单张 GEI 包含更多的运动细节。
3. **CNN-Ensemble (集成模型):**
    
    - **机制：** 俗称“打群架”。它训练了8个不同的神经网络，然后汇总它们的输出。
    - **结果：** 在该项工作中表现最好，证明了模型集成可以有效提升鲁棒性。

#### **GEINet [18]**

- **核心概念：** 专门针对 GEI 设计的浅层网络。
- **详细原理解读：**
    
    - 这是一个相对简单的分类网络。
    - **结构：** 2层卷积层 + 2层全连接层。
    - **输入：** 不同人的 GEI。
    - **输出：** 直接分类出这个 GEI 属于哪一个人（ID）。它是步态识别领域早期的标准Baseline 之一。

### 3\. 生成模型与高级特征学习 (Generative & Advanced Feature Learning)

这一类方法更加复杂，试图通过“生成”或“特殊损失函数”来解决视角差异问题。

#### **Yu et al. [21] - AutoEncoder (AE)**

- **核心概念：** 自编码器。
- **详细原理解读：**
    
    - **AutoEncoder** 是一种无监督学习模型，包含编码器（Encoder）和解码器（Decoder）。
    - **目的：** 这里利用 AE 能够压缩数据的特性，强迫网络忽略掉图像中的“噪音”（如视角变化、背景杂波），只提取出最本质的、视角不变（View-invariant）的特征来代表这个人的身份。

#### **He et al. [21] - MGAN (Multi-task GAN)**

- **核心概念：** 多任务生成对抗网络。
- **详细原理解读：**
    
    - **背景：** 侧面（$90^\circ$ ）的步态信息最丰富，正面（$0^\circ$ ）最难识别。
    - **机制：** 利用 **GAN (Generative Adversarial Networks)** 的生成能力，将某个视角的步态特征**投影/生成**为另一个视角的特征。
    - **举例：** 比如把所有角度的图都“画”成侧面图，然后再进行识别，这样就消除了视角的干扰。

#### **ACL (Angle Center Loss) [23]**

- **核心概念：** 角度中心损失。
- **详细原理解读：**
    
    - 这是对损失函数（Loss Function）的改进。传统的 Softmax Loss 有时不够强。
    - **Center Loss：** 目的是让同一类人的特征在空间里靠得更紧（聚类）。
    - **Angle Center Loss：** 专门针对步态中的“角度”问题进行了优化。6它训练出的特征对于**局部部位缺失**（比如脚被遮挡）和**时间窗口大小**（走的步数多少）具有很强的鲁棒性。

#### **Takemura et al. [4] - Triplet Loss (三元组损失)**

- **核心概念：** 改进了 Wu et al. 的结构，引入 Triplet Loss。
- **详细原理解读：**
    
    - **Triplet Loss** 是人脸识别和步态识别中的大杀器。
    - **输入（三张图）：**
        
        1. **Anchor (A):** 基准图（比如张三的侧面）。
        2. **Positive (P):** 同一个人的另一张图（比如张三的正面）。
        3. **Negative (N):** 不同人的图（比如李四的侧面）。
    - **目标：** 训练网络使得 $Distance(A, P)$尽 可能小，同时 $Distance(A, N)$尽 可能大。
    - **意义：** 这种方法比单纯的分类更适合跨视角识别，因为它直接学习“相似度”。

[image]

### 总结与对比表

<table><tbody><tr><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">方法名称</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">核心技术</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">主要特点/优势</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">类别</span></strong></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">ViDP</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">线性投影</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">寻找潜在的视角不变空间</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">传统方法</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">CMCC</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">聚类 + CCA</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">利用局部运动相关性，最大化视角间关联</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">传统方法</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">CNN-LB</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">CNN (孪生结构)</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">验证两张 GEI 是否属于同一人</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">深度学习 (2D)</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">CNN-3D</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">3D-CNN</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">利用时空信息 (9帧序列)</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">深度学习 (3D)</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">GEINet</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">浅层 CNN</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">结构简单，直接对 GEI 分类</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">深度学习 (2D)</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">AE (Yu)</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">自编码器</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">提取压缩后的视角不变特征</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">生成/重构</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">MGAN</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">GAN (生成对抗)</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">将一个视角的特征转换为另一个视角</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">生成/重构</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">ACL</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">损失函数优化</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">对局部遮挡和时间长度鲁棒</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">特征学习</span></p></td></tr><tr><td style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">Takemura</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">Triplet Loss</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">通过三元组学习拉近同类、推开异类</span></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">度量学习</span></p></td></tr></tbody></table>

* * *

## 4.4.1 主要成果：CASIA-B

[image]

```
三种不同实验设置下CASIA-B的平均排名-1准确率，不包括相同视角案例
```

基于提供的 **GaitSet** 论文图片（表 2）及相关文本内容，以下是对该实验结果的详细**中文分析**：

### **1\. 总结 (Executive Summary)**

文档中的 **表 2 (Table 2)** 展示了 **GaitSet** 方法与当时其他先进方法（如 CNN-LB, AE, MGAN, ACL 等）在 **CASIA-B** 步态数据集上的对比结果。

**核心结论：** GaitSet 在几乎所有的实验设置（小样本、中样本、大样本训练）和测试条件下（正常、背包、穿大衣），准确率都显著优于其他方法。

### **2\. 实验设置 (Experimental Setup)**

表格根据**训练集规模**和**测试条件**对结果进行了分类：

- **三种训练规模 (Training Sets):**
    
    - **ST (小样本训练):** 仅使用 24 个受试者。
    - **MT (中样本训练):** 使用 62 个受试者。
    - **LT (大样本训练):** 使用 74 个受试者。
- **三种测试/探针条件 (Probe Conditions):**
    
    - **NM:** 正常行走 (Normal Walking)。
    - **BG:** 背包 (Carrying a Bag)。
    - **CL:** 穿大衣 (Wearing a Coat/Clothing)。

### **3\. 核心性能分析 (Key Performance Analysis)**

#### **A. GaitSet 的统治级表现**

GaitSet（表中加粗的行）在绝大多数类别中都取得了最高分。

- **在 ST (小样本) 中：** 即使训练数据很少，GaitSet 在正常行走 (NM) 下的平均准确率达到了 **83.3%**，远超竞争对手（如 CNN-LB 的76.1%）。
- **在 LT (大样本) 中：** 数据量增加后，GaitSet 在正常行走 (NM) 下达到了 **96.1%** 的惊人准确率，证明了其良好的扩展性。

#### **B. "穿大衣" (CL) 的挑战**

**CL (穿大衣)** 条件始终是所有模型中最难处理的。

- **原因：** 文本解释说，长款大衣会完全改变人的外观（体型看起来更大），并且遮挡了四肢和躯干的运动细节。
- **结果：** 虽然其他模型在此条件下表现大幅下降（例如 AE 在 MT 设置下仅有24.2%），GaitSet 在大样本 (LT) 设置下仍达到了**70.3%**，比之前的最佳结果（CNN-LB）高出**15%以上**。

### **4\. 视角分析 (Analysis of Viewing Angles)**

文本中提到了一个非常有趣的发现，即**视角与准确率之间的关系**呈现出特定的规律：

- **准确率的 "W" 型波动：** 准确率并非在所有角度都一致。
    
    - **低谷 (**$0^\circ, 90^\circ, 180^\circ$ **):** 在正前/正后 ($0^\circ/180^\circ$ ) 和正侧面 ($90^\circ$ )，准确率通常较低。
    - **高峰 (**$36^\circ, 144^\circ$ **):** 中间角度通常准确率最高。
- $90^\circ$**( 侧面) 的局部低值：** 理论上侧面最能看清步幅，为什么准确率会下降？
    
    - **解释：** 文本指出，在 $90^\circ$ 时，丢失了垂直于行走方向的特征信息（如身体或手臂的**左右摆动**）。
- $0^\circ / 180^\circ$**( 正前/正后) 的低值：**
    
    - **解释：** 这些角度可以看清左右摆动，但丢失了平行于行走方向的特征（如**步幅**）。
- **最佳视角 (**$36^\circ$**&**  $144^\circ$ **):** 这些角度取得了最佳平衡，既能捕捉到步幅（平行信息），也能捕捉到摆动（垂直信息），因此在 LT 设置下准确率甚至超过 99%。

### **5\. 为什么 GaitSet 如此成功？ (Why is GaitSet Successful?)**

根据文本描述，GaitSet 尤其是在小样本 (ST) 训练中表现优异，主要归功于两点创新：

1. **基于集合的输入 (Set-based Input):** 模型不将视频视为严格的时间序列，而是视为一堆帧的“集合”。这意味着它可以利用**数千张**独立的轮廓图 (Silhouette) 进行训练（文中提到一个 batch 有 3,840 张图），而传统基于模板的方法可能只能利用 128 个模板。
2. **随机采样 (Random Sampling):** 训练阶段从序列中随机抽取帧，迫使神经网络学习更鲁棒的特征（集合特征学习），而不是死记硬背帧的顺序。

* * *

## 4.4.2 主要成果：**OU-MVLP**

这段内容展示了**GaitSet**在**OU-MVLP**数据集上的实验结果。OU-MVLP是目前世界上最大的步态数据集之一，包含大量受试者和视角。

[image]

```
OU-MVLP的平均 Rank-1 精确度，不包括相同视角案例。GEINet：[18].3in+2diff：[4]
```

以下是对 **表 3 (Table 3)** 及相关文本 **4.4.2 章节** 的详细中文展开说明和分析：

### 1\. 核心结论：碾压级的性能优势

**表 3** 展示了 GaitSet 与其他两种现有方法（GEINet 和 3in+2diff）在 Rank-1 准确率上的对比。结果非常惊人，GaitSet 展现了显著的优势。

- **全视角对比 (Gallery All 14 Vews):**
    
    - 当注册集（Gallery）包含所有 14 个视角时，**GaitSet 的平均准确率达到了 87.9%**。
    - 相比之下，竞争对手 **GEINet 的平均准确率仅为 35.8%**。
    - **解读：** 这是一个巨大的性能飞跃（提升了超过 50 个百分点），证明了 GaitSet 在大规模数据集上的泛化能力极强。
- **特定视角对比 (Gallery** $0^\circ, 30^\circ, 60^\circ, 90^\circ$ **):**
    
    - 为了公平比较（因为有些旧方法只在部分视角上做了测试），作者列出了仅使用 4 个典型视角作为注册集的结果。
    - **极端视角下的鲁棒性：** 请注意 $0^\circ$ （正前/正后）的测试结果。GEINet 只有 **8.2%** 的准确率（几乎不可用），而 GaitSet 依然保持了 **79.6%** 的高准确率。这再次印证了上一节提到的观点：GaitSet 对视角变化具有极强的鲁棒性。

### 2\. 高效的计算速度 (Computational Efficiency)

文本中强调了一个关键的工程优势：**推理速度极快**。

- **原理：** 传统的基于视频的方法可能需要复杂的时序计算，而 GaitSet 将输入视为“集合”。对于每个样本，其特征表示（Representation）只需要计算一次。
- **数据支撑：**
    
    - 测试集规模：**133,780 个序列**（非常庞大）。
    - 硬件环境：4 块 NVIDIA 1080TI GPU。
    - 耗时：**仅需 14 分钟**。
- **意义：** 这意味着该模型不仅准确率高，而且非常适合在现实世界的大规模监控系统中进行实时部署。

### 3\. 数据集缺陷与“真实”准确率 (The "Missing Data" Caveat)

这一段包含了一个非常重要的细节，解释了为什么最高准确率没有接近 100%：

- **数据集问题：** OU-MVLP 数据集中存在数据缺失的情况。也就是说，探针集（Probe，即待测样本）里的一些受试者，在注册集（Gallery，即数据库）里根本没有对应的视频序列。
- **影响：** 这种情况导致在测试时，模型无论如何都不可能匹配成功（因为正确答案根本不存在）。这拉低了统计出来的准确率。
- **修正后的准确率：** 作者指出，如果剔除这些“本身就没有对应样本”的无效案例，GaitSet 的平均 Rank-1 准确率实际上是从 **87.9% 上升到了 94.1%**。
    
    - 这说明 87.9% 这个数字实际上低估了模型的真实能力。

### 4\. **训练策略与损失函数 (Fig. 5 解读)**

[image]

**Fig. 5** 展示了训练迭代次数 (Training Iterations) 与测试准确率 (Accuracy) 之间的关系，揭示了作者采用的**两阶段训练策略**：

- **第一阶段（蓝色线）：交叉熵损失 (Cross Entropy Loss)**
    
    - **操作：** 使用学习率 $lr=1E-4$和  CE Loss 进行训练。
    - **现象：** 准确率快速上升，在约 80K 次迭代时达到瓶颈，稳定在 **83.9%** 左右。
    - **作用：** 快速学习基本的分类特征。
- **第二阶段（橙色线）：三元组损失微调 (Triplet Loss Tuning)**
    
    - **操作：** 保持学习率 $lr=1E-4$ ，但在损失函数中加入 **Triplet Loss** 进行微调。
    - **现象：** 准确率出现显著跳跃（从 83.9% 跃升至 85.9%），并持续攀升至 **87.6%**。
    - **作用：** Triplet Loss 通过拉近同类样本距离、推远异类样本距离，进一步细化了特征空间，挖掘出更细微的步态差异。
- **第三阶段（灰色线）：降低学习率**
    
    - **操作：** 将学习率降至 $lr=1E-5$继 续使用 Triplet Loss 训练。
    - **结果：** 最终将准确率推至 **87.9%** 的最高点。

### **总结**

这一章节通过 OU-MVLP 实验证明了 GaitSet 的三个核心强项：

1. **大规模泛化性强：** 在万人级数据集上准确率远超现有方法。
2. **训练策略有效：** 证明了 "CE Loss + Triplet Loss" 组合拳的有效性。
3. **工程落地性好：** 推理速度极快，具备极高的实用价值。

* * *

## 4.5 **消融实验（Ablation Experiments）** 和 **模型研究**

什么是消融实验？

在深度学习研究中，为了证明模型中每一个改进点（组件）都是有效的，研究者会通过“控制变量法”，逐一移除或改变某个组件，观察性能变化。这就像通过拆掉机器的一个零件来看看它还能不能转，以此来证明该零件的重要性。

以下是对 **表 4、表 5、表 6** 及其对应文本的详细中文解读和展开分析：

### **1\. 表 4 (Table 4)：核心架构的有效性验证**

[image]

```
使用集合LT在CASIA-B上进行的消融实验
这些结果是所有11个视图的排名-1 准确率的平均值，不包括相同视图的情况。括号中的数字表示每列中第二高的结果。这里的 "att "是注意力的缩写。
```

#### **关键发现与解读：**

- **Set (集合) vs. GEI (能量图):**
    
    - **实验设置：** 第一行是使用 GEI（将所有帧平均成一张图）输入网络；第二行是使用 GaitSet 的集合输入方式。网络结构完全相同。
    - **结果：** Set 方法全面碾压 GEI。
        
        - **NM (正常):** 提升超过 **6%**。
        - **CL (穿大衣):** 提升超过 **19%** (从 50.7% 提升到 69.9%)。
    - **原因分析：** 文本中提到两点原因：
        
        1. GEI 压缩成一张图会丢失时序信息，而 GaitSet 在高层特征图上提取集合特征，保留了更多信息。
        2. 将步态视为“集合”极大地扩充了训练数据量（不再受限于模板数量）。
- **池化策略的选择 (The Impact of SP):**
    
    - 作者尝试了 Max（最大值）、Mean（平均值）、Median（中位数）、Attention（注意力机制）等多种将帧特征聚合的方式。
    - **结果：** 虽然基于像素的注意力 (Pix-att) 在 NM 和 BG 上分数最高，但 **Max (最大池化)** 在 CL (最难场景) 上表现最好，且结构最简单。
    - **决策：** 为了平衡性能和模型简洁性，最终版本选择了 **Max**。
- **MGP 的作用:**
    
    - 对比第二行（无 MGP）和最后一行（有 MGP），加入 MGP 后，所有三个子集（NM, BG, CL）的准确率都有明显提升。这证明了融合网络不同深度的特征（浅层细节+深层语义）对识别很有帮助。

### **2\. 表 5 (Table 5)：HPM 结构的参数优化**

[image]

```
在CASIA-B上使用LT集合进行的不同HPM尺度表和HPM权重独立性实验的影响
```

这个表格研究了 **水平金字塔映射 (HPM)** 的两个关键参数：**尺度 (Scales)** 和 **权重独立性 (Weight Independence)**。

#### **关键发现与解读：**

- **尺度 (Scales) 的影响:**
    
    - 从 Scale=1 (即不使用 HPM，只有全局特征) 到 Scale=4，随着切割的尺度变多，准确率稳步上升。
    - 这说明将特征图进行水平切割（例如切成头部、躯干、腿部等不同区域），能提取更细粒度的局部特征，从而提升识别率。
- **权重独立 vs. 共享 (Independent vs. Shared):**
    
    - 这是非常关键的一点。在 Scale=5 的设置下，作者比较了全连接层是否共享权重。
    - **Shared (共享权重):** 准确率很低（NM 91.1%）。
    - **Independent (独立权重):** 准确率大幅提升（NM 96.1%）。
    - **结论：** 在 CL (穿大衣) 集合上，独立权重带来了 **近 10%** 的提升。文本指出，使用独立权重不仅准确率高，还能让网络**收敛得更快**。这说明不同身体部位（头、脚）的特征分布差异很大，必须用不同的参数去学习。

### **3\. 表 6 (Table 6)：训练策略与损失函数**

[image]

```
使用LT在CASIA-B上进行的不同损失函数分析
```

这个表格探讨了如何训练这个模型才能达到最佳效果，主要涉及 **损失函数 (Loss Function)** 和 **正则化手段 (BN & Dropout)**。

#### **关键发现与解读：**

- **损失函数的“组合拳”:**
    
    - 单独使用 **CE Loss (交叉熵损失)**：效果一般 (NM 95.6%)。
    - 单独使用 **Triplet Loss (三元组损失)**：效果也不错 (NM 95.3%)，但在 CL 上表现最好。
    - **组合策略 (CE + Triplet):** 效果最佳 (NM 96.1%, BG 90.8%, CL 70.3%)。
    - **结论：** 先用 CE 学习大致分类，再用 Triplet 细化类内类间距离，是最佳方案。
- **正则化的重要性 (BN & Dropout):**
    
    - **Dropout:** 表格显示，如果只用 CE Loss 而不加 Dropout，准确率会暴跌（第一行，NM 只有 90.9%）。这说明 Dropout 对防止过拟合至关重要。
    - **Batch Normalization (BN):** 加入 BN 能提升所有策略的表现。
    - **最终方案：** 同时使用 BN 和 Dropout，配合组合损失函数，能达到最高性能。

### **4\. 总结 (Summary)**

这一章节通过详尽的实验数据，确定了 GaitSet 的最终形态：

1. **输入：** 必须把步态当做“集合 (Set)”而非“图像 (GEI)”。
2. **聚合：** 使用简单的 **Max Pooling** 即可。
3. **结构：** 必须加上 **MGP** (多尺度特征) 和 **HPM** (局部特征)。
4. **细节：** HPM 的全连接层必须**权重独立**。
5. **训练：** 使用 **CE Loss + Triplet Loss** 联合训练，并加上 **BN** 和 **Dropout**。

正是这些精细的组件选择和参数调优，使得 GaitSet 在当时取得了 State-of-the-Art (SOTA) 的成绩。

* * *

## 4.6 特征降维 (Feature Dimension Reduction)

### **1\. 为什么要进行特征降维？ (The Problem)**

在标准架构下，GaitSet 将所有 HPM（水平金字塔映射）的输出拼接后，得到的特征向量维度极其巨大：$256 \times 31 \times 2 = 15,872 \text{ 维}$

- **痛点：** 如此高维的特征会导致测试（推理）效率低下，存储成本高昂，不利于实际应用。
- **目标：** 在尽可能不损失识别准确率的前提下，大幅压缩特征维度。

作者提出了**两种**降维方法：

### **2\. 方法一：调整 HPM 输出维度 (Method 1: HPM Output Dimensions)**

这种方法是在**特征生成之前**就进行控制。即调整网络内部 HPM 模块中全连接层（FC）的输出大小。

[image]

```
图6.识别准确率与HPM输出维度之间的关系。从左到右依次为CASIA-B NM、BG和CL子集的识别结果。如图中的不同线条所示，不同的训练策略会产生不同的关系。
```

图 6 展示了将 HPM 输出维度设置为 32, 64, 128, 256, 512, 1024 时，不同训练策略下的准确率变化。

- **"金发姑娘原则" (适中最好):**
    
    - **维度太低 (32):** 准确率下降。原因：限制了全连接层的学习能力 (Learning Capacity)，无法表达复杂的步态特征。
    - **维度太高 (1024):** 准确率也下降（特别是绿色的 CE Loss 线在 CL 图中下降明显）。原因：参数过多导致**过拟合 (Overfitting)**。
    - **最佳点 (Sweet Spot):** **256** 维左右通常能达到最佳平衡。
- **训练策略的稳定性:**
    
    - **橙色线 (Triplet tune):** 在所有维度和子集（NM, BG, CL）中表现最稳定且最高。
    - **绿色线 (CE Loss):** 在高维（1024）且困难场景（CL）下，鲁棒性最差，容易过拟合。

**结论：** 虽然这种方法可以将最终维度压缩到原来的 1/4，但在 BG 和 CL（背包和穿衣）数据集上会有明显的性能损失

### **3\. 方法二：引入新的全连接层 (Method 2: New FC Layer)**

这种方法是在**特征生成之后**进行后处理。即模型训练好后，保持主干不变，在最后加一层新的全连接层，专门用于将 15,872 维压缩到低维，并用 Triplet Loss 进行微调。

[image]  

表 7 展示了将 15,872 维特征直接压缩到不同维度后的准确率。

- **压缩效果惊人:**
    
    - 当压缩至**1024 维**时，NM（正常行走）的准确率仍维持在 **95.0%**。
    - **计算：** $1024 \div 15872 \approx 6.5\%$ 。这意味着保留了原始精度的同时，数据量减少了**93.5%**。
- **性能阈值:**
    
    - 从1024 降到 512，性能损失很小（95.0% -> 94.4%）。
    - 但如果过度压缩（如降到 128 维），CL（穿衣）条件下的准确率会从 69.1% 跌至 62.5%，损失较大。

### **4\. 总结与应用建议 (Conclusion)**

- **最佳实践：** 推荐使用**方法二**（引入新 FC 层）。
- **权衡 (Trade-off):** 虽然方法二引入了一个后处理步骤（不再是严格意义上的端到端直接输出），但它极其有效。
- **实际意义：** 将特征向量从 1.5 万维压缩到 1024 维，使得 GaitSet 在保持高精度的同时，具备了在现实世界大规模系统中部署的**实用性 (Practicality)**。

* * *

## 4.7 实用性 Practicality

探讨了 **GaitSet** 在更复杂、更接近现实应用场景下的**实用性 (Practicality)**。

作者设计了三个极具现实意义的实验场景，并在**不重新训练模型**的前提下（直接使用之前训练好的 LT 模型），验证了 GaitSet 的灵活性和鲁棒性。

### **场景一：极少帧数的输入 (Limited Silhouettes)**

**现实痛点：** 在刑侦取证中，摄像头可能只抓拍到嫌疑人断断续续的几张剪影，无法获得连续完整的视频序列。

**实验设计：**

- 不再输入完整的视频，而是随机抽取$N$张轮廓图组成一个“集合”。
- **Fig. 7 (折线图)** 展示了抽取帧数（横轴）与 Rank-1 准确率（纵轴）的关系。

[image]

**结果分析：**

1. **极少帧也能识别：** 即使只用 **7 张** 随机抽取的轮廓图，准确率也能达到 **82%**。这证明了模型并不依赖帧的连续性，而是利用了“集合”中的信息。
2. **准确率随帧数上升：** 随着输入帧数增加，准确率单调上升。
3. **25 帧达到饱和：** 当帧数超过 **25 帧** 时，准确率曲线趋于平缓，接近使用所有图像的效果。
    
    - **原因：** 25 帧大约对应**一个完整的步态周期 (Gait Cycle)**。这意味着只要获取了一个周期的数据，GaitSet 就能提取出足够完整的特征。

### **场景二：多视角融合 (Multiple Views)**

**现实痛点：** 现实监控中，可能会通过多个不同角度的摄像头同时捕捉同一个人。

**实验设计 (Table 8)：**

- 将两个不同视角（但在相同行走条件下）的序列混合，组成输入集合。
- 为了控制变量，限制每个集合的总帧数为 10 帧（例如：视角 A 的 5 帧 + 视角 B 的 5 帧）。
- **Table 8** 展示了不同视角差（View Difference）下的准确率。

[image]

**结果分析：**

- **多视角 > 单视角：** 表格最后两列显示，使用 10 帧混合视角的数据（平均 **97.25%** 左右），通常比使用 10 帧单一视角的数据（**89.5%**）准确率要高得多。
- **互补性原理：** 这种提升印证了之前提到的“视角互补”理论：
    
    - 侧面视角（如 $90^\circ$ ）提供了**步幅 (Parallel)** 信息。
    - 正面/背面视角（如 $0^\circ/180^\circ$ ）提供了**身体摆动 (Vertical)** 信息。
    - 将它们混合输入到 GaitSet 中，模型能自动聚合这些互补特征，从而获得比单一视角更全面的描述。

### **场景三：多行走状态融合 (Multiple Walking Conditions)**

**现实痛点：** 一个人在行走过程中可能会改变状态，比如走着走着穿上了外套，或者拿起了包。

**实验设计 (Table 9)：**

- 混合不同行走条件下的帧（例如：正常行走 NM + 背包 BG）。
- **Table 9** 对比了纯单一状态（如 NM 10帧）和混合状态（如 NM 10帧 + BG 10帧）的效果。

[image]

```
括号中的数字表示每个输入集合中剪影编号的限制条件。  
```

**结果分析：**

1. **数据量依然是关键：** 总体上看，帧数越多（从 10 增加到 20），准确率越高。
2. **混合噪声的影响 (Complex Interaction)：**
    
    - **NM + BG/CL (正常+异常)：** 准确率不如纯 NM 高。例如 `NM(20)` 是 94.2%，而 `NM(10)+BG(10)` 是 93.0%。
        
        - **原因：** NM（正常行走）是最干净的数据。掺入 BG 或 CL 会引入“噪声”或遮挡信息，反而干扰了最纯净的特征。
    - **BG + CL (背包+穿衣)：** 这种组合有时会有奇效。
        
        - **原因：** 作者认为 BG 和 CL 虽然都包含干扰（噪声），但它们包含的是**互补的噪声和信息**。在某些情况下，结合这两种困难样本，反而能帮助模型提取出更顽强的特征。

### **总结**

这一章节强有力地证明了 **GaitSet (基于集合的方法)** 相比于传统方法的巨大优势：**灵活**。

因为它不依赖时序，所以我们可以像“搭积木”一样，随意组合来自不同时间、不同视角、不同状态的几张图片丢给模型，它都能很好地处理。这使得它在数据破碎、不完整的现实监控场景中具有极高的落地价值。

* * *

## **5 结论 (Conclusion)**

这段简短的结语是对 **GaitSet** 核心价值的最终升华。结合前面所有的实验分析，我们可以将这篇论文的贡献和意义总结为以下几个关键点：

### **1\. 核心范式的转变：步态即集合 (Gait as a Set)**

这是整篇论文的**灵魂**所在。

- **传统视角：** 以前的方法要么把步态看作“序列”**（Sequence，如 LSTM，受限于时间顺序），要么看作**“模板”（Template，如 GEI，压缩损失大）。
- **GaitSet 视角：** 将步态视为“集合”（Set）。
    
    - 这意味着输入数据是**无序的 (Permutation Invariant)**。
    - 这一视角的转变，使得模型在提取时空信息时，比传统方法**更有效**（准确率更高）且**更高效**（计算速度更快）。

### **2\. 已经被验证的统治力 (Proven Performance)**

- **SOTA 表现：** 在两个基准数据集（**CASIA-B** 和 **OU-MVLP**）上的实验结果表明，GaitSet 在当时取得了最高的识别准确率，击败了所有最先进的算法。
- **抗干扰能力：** 特别是在复杂的环境变化下（如**跨视角**、**背包**、**穿大衣**），GaitSet 展现出了极强的鲁棒性。

### **3\. 极强的灵活性与落地潜力 (Flexibility & Practicality)**

这一段特别提到了 GaitSet 能够“聚合来自不同序列的信息”。

- 正如在 4.7 节（实用性实验）中看到的，你不需要给模型一个完美的连续视频。
- 你可以给它几张零散的截图，或者把不同视角的片段混在一起丢给它，它都能利用“集合”的特性把这些碎片信息整合起来，从而提升识别率。
- 这种特性让它在现实世界的监控、刑侦等**实际应用 (Practical Applications)** 中具有巨大的潜力。

### **4\. 未来的扩展性：不仅仅是步态 (Broad Applicability)**

作者在最后提出了一个非常有远见的观点：

- **“集合假设 (Set Assumption)”** 是一种通用的数据处理逻辑。
- 它不仅适用于步态识别，还可以通过微小的改动，迁移到其他生物特征识别任务中，例如：
    
    - **行人重识别 (Person Re-identification, Re-ID)**
    - **基于视频的人脸识别 (Video-based Face Recognition)**
- 因为在这些任务中，我们往往也是拥有一组图片（视频帧），而不需要过分关注它们的时间顺序，更关注的是这组图片共同指向的那个人的身份特征。

### **总结：GaitSet 为什么经典？**

GaitSet 之所以成为步态识别领域的经典之作，是因为它**跳出了“时序建模”的思维定势**。它证明了对于步态识别这种周期性的运动，**“哪一帧在前，哪一帧在后”并不重要，重要的是“这堆帧里包含了什么特征”**。这一简单的假设，换来了性能、速度和灵活性上的全面提升。

# 下一步思考

从**数学原理**、**理论深度**以及**与其他技术路线的对比**这三个维度，对 GaitSet 的核心组件进行剖析。

GaitSet 的本质是**用空间维度的统计特征来近似时间维度的运动特征**，这是一种通过“降维打击”来换取鲁棒性的策略。

## 1\. Set Pooling (SP) 的局限性与数学本质

**理论核心：** SP 将视频序列 $V = \{v_1, v_2, ..., v_n\}$压 缩为一个特征张量 $z$ 。GaitSet 最终选择了 **Max Pooling** 作为主要手段。

### **1.1 数学上的“信息丢失”与“特征保留”**

在数学上，Max Pooling 操作定义为 $z_{c,h,w} = \max_{t=1}^{n} (v_{t,c,h,w})$ 。

- **丢失的信息 (Temporal Loss)：** 这种操作是一个非单射函数（Non-injective）。我们知道在位置 $(h,w)$处 ，通道 $c$ 曾达到过最大值 $v_{max}$ ，但我们**完全丢失了** $t$ **（时间索引）**。即，我们不知道这个动作是发生在第 1 帧还是第 30 帧。
- **保留的信息 (The "Hull" of Motion)：** 对于步态这种周期性运动，Max Pooling 实际上捕获了**运动的外包络 (Motion Hull)**。
    
    - *举例*：一个人挥动手臂。Mean Pooling 会得到手臂运动轨迹的“模糊平均影”；而 Max Pooling 会保留手臂摆动到**最前端**和**最后端**时的清晰特征。
    - *结论*：对于步态识别，识别“能踢多高、步幅多大”比识别“在哪一毫秒踢到最高”更重要。因此，SP 丢失的是对于身份识别**冗余**的时序相位信息，保留的是**关键**的极值特征 。

### **1.2 周期性与采样率的博弈**

- **假设：**GaitSet 假设输入集合包含至少一个完整的步态周期。
- **局限性分析：**
    
    - 如果输入帧数 $n$远小于一个周期（例如只输入 3 帧），Max Pooling 得到的特征是不完整的分布采样，此时识别率会显著下降（如原文图7所示，7 帧以下准确率陡降）。
    - **高阶矩缺失：** SP 目前只利用了 Max (类似 $L_\infty$范 数)。如果在 SP 中引入**方差 (Variance)** 或 **偏度**，理论上可以捕获运动的“动态程度”，但这会增加计算量。GaitSet 选择了最简的 Max 以换取计算效率 7。

## 2\. 置换不变性 (Permutation Invariance) 的双刃剑

**理论核心：** 模型满足 $f(v_1, v_2, \dots) = f(v_{\pi(1)}, v_{\pi(2)}, \dots)$ ，其中 $\pi$是 任意置换。

### **2.1 鲁棒性优势 (Robustness)**

- **抗丢帧与乱序：** 在现实监控中，网络卡顿导致掉帧，或者检测算法漏检导致帧序列不连续是常态。LSTM 或 3D-CNN 依赖严格的 $t \to t+1$链 式法则，一旦中间断裂，误差会累积。GaitSet 将视频视为“袋子 (Bag of Frames)”，天然免疫这种噪声。
    
- **多源融合的数学基础：** 正因为 $f(A \cup B) = \text{Pool}(f(A), f(B))$ ，GaitSet 可以在数学上合法地将“摄像机A拍到的5帧”和“摄像机B拍到的5帧”直接合并为一个集合输入，无需对齐时间轴。这在论文Table 8 的实验中得到了验证。

### **2.2 牺牲的“微动”敏感性**

- **局限性：** 这种无序性意味着模型无法理解**因果关系**或**精细的时序模式**。
- **对比 GaitPart/SkeletonGait：**
    
    - 后续模型（如 GaitPart）引入了 "Micro-motion Capture" 模块，因为有些人的步态特征隐藏在“脚落地时的细微抖动”或“加速度变化”中。GaitSet 的 Max Pooling 会将这些瞬时变化平滑掉或混淆，导致在极其相似的受试者（Hard Negatives）区分上可能不如时序敏感模型。

## 3\. HPM 的信息瓶颈与解耦机制

**理论核心：** 水平金字塔映射 (HPM) 将特征图 $z$切 分为 $S$个 尺度下的多个条带，并使用**独立的全连接层 (Independent FCs)**。

### **3.1 空间解耦 (Spatial Disentanglement)**

- **为什么需要独立 FC？**
    
    - **数学解释：** 设 $z_{head}$为 头部特征， $z_{legs}$ 为腿部特征。腿部运动幅度大（像素方差大），头部运动幅度小（主要是平移）。
    - 如果使用共享权重的卷积或 FC（即 $W \cdot z_{head}$ 和 $W \cdot z_{legs}$ ），共享矩阵 $W$会 倾向于拟合方差大的特征（腿部），导致头部微小的特征被淹没。
    - **HPM 策略：** 使用 $W_{head} \cdot z_{head}$和  $W_{legs} \cdot z_{legs}$ 。每个条带拥有独立的投影空间，强制网络去挖掘该部位特有的细粒度特征（Fine-grained discriminative features）。

### **3.2 与 3D-CNN 和 Transformer 的对比**

- **vs. 3D-CNN：** 3D-CNN 是刚性的局部时空聚合（Local Spatiotemporal）。HPM + SP 是“全局时间 + 局部空间”。HPM 在空间上的切分是硬编码的（Hard-coded），假设人总是站立的。这使得它不具备旋转不变性（如果人躺着走，HPM 就失效了），但在监控场景下这通常不是问题。
- **vs. Vision Transformer (ViT)：** ViT 通过 Self-Attention 自动学习关注哪些 Patch。HPM 可以看作是一种**先验知识指导的“硬注意力”**——我们强行告诉模型：“这部分是头，那部分是脚，分开看”。相比 ViT，HPM 参数量更小，在小样本（如 CASIA-B）上更不容易过拟合。

## 4\. 损失函数 (Loss Function) 的度量空间构建

**理论核心：** $L_{total} = L_{CE} + L_{Triplet}$ 。这里的关键在于 Triplet Loss 是如何作用于 HPM 输出的**每一个切片**上的。

### **4.1 细粒度度量学习 (Fine-grained Metric Learning)**

- **公式细节：** 论文中提到 $L = \sum_{s=1}^S \sum_{t=1}^{2^{s-1}} L_{triplet}(f_{s,t})$。
- **数学意义：** 这不是对整个人计算一个距离，而是进行了 $31 \times 2 = 62$**次 独立的度量学习**。
    
    - 这意味着模型被强迫构建 62 个不同的子空间。在“头部子空间”里，必须拉近同一个人的头部特征；在“脚部子空间”里，必须拉近同一个人的脚部特征。
    - 最终的欧氏距离是这些子空间距离的**线性叠加**。这类似于一种**集成学习 (Ensemble Learning)**，极大地提高了泛化能力。如果一个人的大衣遮住了腿（脚部子空间失效），头部和躯干子空间依然能提供有效的距离度量。

### **4.2 训练策略的动力学 (Dynamics)**

- 先 CE 后 Triplet：
    
    - **CE Loss (Cross Entropy):** 优化的是决策边界（Hyperplane），让特征具有可分性，收敛快，但特征在类内可能很松散。
    - **Triplet Loss:** 优化的是流形结构（Manifold），直接压缩类内方差，扩大类间方差。
    - **图 5 分析：** 论文图 5 显示，CE 训练后准确率卡在 83.9%，引入 Triplet 后跃升至 87.9%。这在数学上对应于特征点在高维球面上从“大致分开”变成了“紧致聚类”。

## 总结

GaitSet 在数学上的成功，在于它极其精准地找到了步态识别中的**不变性 (Invariance)** 与 **判别性 (Discriminability)** 的平衡点：

1. 利用 **Set Pooling** 舍弃冗余的**相位信息**，换取对帧率和乱序的**不变性**。
2. 利用 **HPM** 和 **独立 FC** 舍弃特征图内部的**空间相关性**，换取对局部细节的**强判别性**。
3. 利用 **多尺度独立 Loss** 将单一识别问题转化为几十个局部特征比对的**集成问题**。

# 🚶‍♂️ GaitSet：从“论文公式”到“代码实现”的深度透视

按照 **核心创新理念** $\to$**核心技术映射** $\to$ **代码架构细节** 的逻辑进行组织，确保从高层概念到具体代码实现的平滑过渡

GaitSet([Chao 等, 2019](zotero://select/library/items/96XFGZSH))的核心创新在于打破了将步态视为“时序序列”的传统，转而将其视为 **“图像集合”**。

## I. 核心创新与整体架构概述

### 1\. 核心理念：步态即“集合”

GaitSet 的核心创新在于，它不关心帧的先后顺序，而是关注这组图中是否包含了具有辨识度的特征。

- **理念转变**：将步态视频视为一个**无序的图像集合** $\{v_1, v_2, ..., v_T\}$ 。
- **实现目标**：实现对帧顺序的**置换不变性**（即打乱帧的顺序，输出结果不变）。
- **局限性**：GaitSet 因此对**短时序微动**（Micro-motion）和**动作顺序**不敏感。这正是后续 **SkeletonGait++** 或 **GaitPart** 等模型试图通过引入精确时序模块来解决的痛点。

### 2\. GaitSet 整体架构：MGL 结构

[image]

```
GaitSet的框架：SP为集合池化。梯形为卷积和池化区块，同一列中的区块具有相同的配置，用大写的矩形表示。请注意，虽然MGP中的区块与主pipeline中的区块配置相同，但参数只在主pipeline中的区块之间共享，而不与MGP中的区块共享。HPP为水平金字塔池化。
```

GaitSet 的框架包括 **SP（集合池化）**和 **HPP（水平金字塔池化）**等核心组件。它采用了独特的 **MGL (Multilayer Global Pipeline / Multiscale Global Level) 架构**，实现了双流特征融合：

> **浅层特征：**来自网络浅层（如 set_block1），包含纹理、边缘等细节；**深层特征：**来自网络深层（如 set_block3），包含语义、形状等抽象信息。

- **主 Pipeline (Frame-level 流)：**负责提取每一帧的局部细节特征。
- **MGP (Set-level 流)**：负责处理经过 **SP 聚合**后的全局信息。

```
注意：虽然 MGP 中的区块（gl_block）与主 pipeline 中的区块（set_block）配置相同，但参数只在主 pipeline 中的区块之间共享，而不与 MGP 中的区块共享。
```

> - **代码体现**：`gl = gl + self.set_pooling(...)`
>     
>     - `gl` 是全局特征流。每次经过一个 `set_block` 提取出新的**帧级特征**后，都要对其进行 SP，然后用加法（Feature Addition）融入到 `gl` 中。
>     - **意义**：让最终的特征 `gl` 同时包含了从浅到深**所有层级**的、经过时间聚合的**集合信息**，增强了特征的表达能力。

## II. 核心技术映射与 Tensor 维度流动

GaitSet 的成功主要归功于两大技术：**Set Pooling (SP)** 和 **Horizontal Pyramid Mapping (HPM)**。

### 1\. 核心技术一：集合池化 (Set Pooling, SP)

[image]

Set Pooling 是实现“集合”理念的灵魂组件。它的作用是将输入的一个视频（$N$ 帧图像）压缩成一个单一的特征向量，从而**消灭时间维度** $s$ 。

<table><tbody><tr><td data-colwidth="117" style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">概念</span></strong></p></td><td style="background-color: rgb(239, 239, 239);"><p><strong><span style="background-color: rgb(239, 239, 239)">详细解释</span></strong></p></td></tr><tr><td data-colwidth="117" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">论文公式</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)"><span class="math">$ z = \mathcal{G}(V) = \max_{t=1}^{N}(v_t)$</span> 。通常使用Max Pooling，但 GaitSet 通常会</span><strong><span style="background-color: rgba(0, 0, 0, 0)">融合MaxMean Pooling</span></strong><span style="background-color: rgba(0, 0, 0, 0)">。</span></p></td></tr><tr><td data-colwidth="117" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">Tensor 输入</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">5 维 Tensor：<span class="math">$(n, c, \mathbf{s}, h, w)$</span> 。（<span class="math">$s$</span>即Sequence Length，即帧数<span class="math">$N$</span>）</span></p></td></tr><tr><td data-colwidth="117" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">代码逻辑</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">在 </span><code>forward</code> 函数中，操作如下：<code>max_pool = x.max(2)[0]</code> 和 <code>mean_pool = x.mean(2)</code>，即在第 <strong>2 维度</strong>（<span class="math">$s$</span>维度）上进行池化。</p></td></tr><tr><td data-colwidth="117" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">Tensor 输出</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><span style="background-color: rgba(0, 0, 0, 0)">4 维 Tensor：<span class="math">$(n, c, h, w)$</span> 。 <span class="math">$s$</span></span><strong><span style="background-color: rgba(0, 0, 0, 0)">维度彻底消失</span></strong><span style="background-color: rgba(0, 0, 0, 0)">。</span></p></td></tr><tr><td data-colwidth="117" style="background-color: rgba(0, 0, 0, 0);"><p><strong><span style="background-color: rgba(0, 0, 0, 0)">无序原理</span></strong></p></td><td style="background-color: rgba(0, 0, 0, 0);"><p><code>max</code> 操作只关心“这个特征在某一帧里出现过吗”，而不关心“它是第几帧出现的”。因此，倒着走的序列和正常走的序列输出的集合特征是完全一样的。</p></td></tr></tbody></table>

GaitSet对**短时序微动**（Micro-motion）和**动作顺序**不敏感。这正是后续 **SkeletonGait++** 或 **GaitPart** 引入时序模块或骨骼流（包含精确的时序坐标）想要解决的痛点 。

### 2\. 核心技术二：水平金字塔映射 (Horizontal Pyramid Mapping, HPM)

[image]SP 解决了时序问题，HPM 则是为了解决空间上的**多尺度**和**局部细节**问题。它模拟了人类看步态的方式：既看整体，也看局部（头、躯干、腿）。

- **操作原理**：将 SP 后的特征图 $(n, c, h, w)$ 在**高度方向**（$h$维度）上切分成 $S$个 条带（Strips）。通常使用多个尺度，例如 $S=1$ (全图), $S=2$( 上下半身), $S=4$ (更细的切分)。
- **代码逻辑**：本质上是对 Tensor 的**切片 (**`chunk`) 或 **分块 (**`Slicing`) 操作，并在每个条带上进行 **Global Average Pooling (GAP)**。
- Tensor 流动视角：
    
    $(n, c, h, w) \xrightarrow{\text{HPM (Scales 1,2,4...)}} \text{List of } [(n, c), (n, c)...] \xrightarrow{\text{Concat}} (n, c \times \text{total\_parts})$
    

## III. OpenGait 代码实现透视

在阅读 `OpenGait/modeling/models/gaitset.py` 时，需要重点关注 `build_network`（定义网络层）和 `forward`（定义数据流向）两个函数。

### 1\. 核心组件解析 (`build_network`)

这个函数负责模型初始化，定义了处理步态序列所需的各个模块。

#### A. 特征提取块 (`set_block1` ~ `set_block3`)

这是模型的主干 CNN，结构类似 VGG。

- **结构**：每一块包含 2 个卷积层（`BasicConv2d`）和 `LeakyReLU` 激活函数。Block 1和Block 2包含最大池化（MaxPool）以降低分辨率，Block 3保持分辨率。
- `SetBlockWrapper`：这是一个自定义包装器。它会将输入 5 维张量 $[N, S, C, H, W]$中 的 $N$ 和 $S$ 维度合并为 $N \times S$ ，通过2D CNN处理完后再还原，从而实现对**每一帧图像的独立特征提取**。

#### B. 全局特征流水线 (`gl_block2`, `gl_block3`)

- **用途**： `set_block` 的**深拷贝（Deep Copy）**。它们不处理原始视频序列，而是处理经过 **Set Pooling** 聚合后的**“全局特征”**。
- **MGL 作用**：融合不同分辨率下的集合级特征。论文认为，通过融合不同层级（浅层和深层）的集合特征，模型能同时捕获浅层的细节（纹理/边缘）和深层的语义（形状/姿态）。

#### C. 集合池化 (`set_pooling`)

- **核心实现**：`PackSequenceWrapper(torch.max)`。
- **核心逻辑**：在时间维度 $S$上取最大值（Max Pooling），实现“置换不变性”。

#### D. 头部与多尺度处理 (`HPP` & `Head`)

- **HPP (Horizontal Pooling Pyramid)**：水平金字塔池化。它将特征图在高度方向上切分成不同的条带（Bins），以提取**局部特征**（如头部、躯干、腿部的特征）。
- **Head (SeparateFCs)**：独立的全连接层。它将 HPP 提取的每一个条带特征映射到度量空间，用于计算 Triplet Loss。

> **Triplet Loss**：一种度量学习（Metric Learning）的损失函数。它的目标是：让**同一个人**的步态特征（Embedding）在空间中**靠得更近**，让**不同的人**的步态特征**离得更远**。

### 2\. 数据流向与 MGL 融合 (`forward`)

`forward` 函数展示了特征提取、SP 聚合和 MGL 融合的复杂逻辑。

#### 阶段一：特征提取与 MGL 融合 (Feature Addition)

模型采用了类似ResNet残差连接的结构，将帧级特征聚合后融入全局特征流，是 MGL 架构的核心。

> ### 步态轮廓图 (Silhouettes)
> 
> - **专业概念**：步态轮廓图（Silhouettes），代码中简称为 `sils`。
> - **通俗解释**：它不是普通的视频帧，而是经过**前景分割**（背景去除）处理后的**二值图**（黑白图）。简单来说，就是把人从背景中抠出来，只留下一个**黑色人形轮廓**，背景是白色。
> - **为什么用轮廓图？**：为了让模型专注于人体的**形态和运动**，排除衣物颜色、背景、光照等无关干扰因素。`sils` 是 GaitSet 的输入。

1. **Block 1 初始化**：
    
    - `outs = self.set_block1(sils)`：提取基础帧级特征。
    - `gl = self.set_pooling(outs)`：将帧级特征在时间轴上压缩，得到**初始的全局特征** `gl`。
    - `gl = self.gl_block2(gl)`：对初始全局特征进行进一步卷积处理。
2. **Block 2 融合（关键）**：
    
    - `outs = self.set_block2(outs)`：继续提取帧级特征。
    - `gl = gl + self.set_pooling(...)`：**关键步骤**。将 Block 2 的帧级特征池化后，通过加法（Add）融合进之前的全局特征流 `gl` 中。
    - `gl = self.gl_block3(gl)`：继续处理全局特征。
3. **Block 3 融合**：
    
    - `outs = self.set_block3(outs)`：最后一层帧级提取。
    - `outs = self.set_pooling(...)`：池化得到最后的集合特征。
    - `gl = gl + outs`：最后一次融合。

> **总结流程**：帧级特征 $\xrightarrow{SP}$集 合特征 $\xrightarrow{\text{Add}}$全 局特征流 `gl`。这种残差连接式的融合（Feature Addition）是 MGL 捕获**多尺度全局步态模式**的关键。

#### 阶段二：水平金字塔映射 (HPM)

模型将最终得到的两个高度抽象的特征张量在**高度方向**切片（Bins），以提取细粒度特征。此时，模型拥有：

- `outs`：来自最后一层的主干特征（池化后）。
- `gl`：融合了多尺度信息的全局特征流。

1. `feature1 = self.HPP(outs)`：对主干特征进行水平切分池化。
2. `feature2 = self.HPP(gl)`：对多尺度全局特征进行水平切分池化。
3. `feature = torch.cat([...], -1)`：将两组细粒度特征拼接，形成最终的特征表示。

#### 阶段三：输出

- `embs = self.Head(feature)`：通过全连接层降维，生成最终用于计算距离的 **Embedding**。
- **返回值**：以字典格式返回，适配 OpenGait 训练框架。

> Embedding：是模型最终输出的一个**低维、密集**的特征向量，它代表了一个人的步态身份。这个向量之间的距离可以用于判断两个人是否是同一身份。

## IV. 总结：GaitSet 成功的关键

结合代码逻辑，GaitSet 的成功可以归结为以下三点：

1. **Set Pooling (SP)**：彻底抛弃了对时间顺序的依赖，使得模型极其灵活，且计算效率高（不需要复杂的 LSTM/Attention 计算）。
2. **MGL 结构**：代码中 `gl = gl + ...` 的设计，让最终的特征同时包含了浅层的细节（Block 1, 2）和深层的语义（Block 3），捕获了**多尺度的全局步态模式**。
3. **HPP 与 SeparateFCs**：强制模型关注人体的不同部位，提取了对视角变化鲁棒的**细粒度特征** (Fine-grained features)，这对跨视角识别非常重要。

* * *

* * *

# 对 **Set Pooling (SP)** 和 **Horizontal Pyramid Mapping (HPM)** 的深度解析

基于提供的论文原文和深度解读文档，以下是对 **Set Pooling (SP)** 和 **Horizontal Pyramid Mapping (HPM)** 的深度解析，重点回答您关于维度变换和时序信息丢失的思考。

## 1\. Set Pooling (SP)：时序维度的压缩机

Set Pooling 是 GaitSet 将“视频序列”转化为“集合”的核心操作。它的任务是将 $N$ 帧图像特征压缩成 1 个能够代表该序列整体特征的向量。

### **Tensor 维度的变换**

在 OpenGait 的代码实现（如 `SetPooling` 类的 `forward` 函数）中，数据流动的维度变化如下：

- **输入 (Input):** $\mathbf{X} \in \mathbb{R}^{n \times c \times s \times h \times w}$
    
    - $n$ : Batch Size (批次大小，例如 8 个人)
    - $c$ : Channels (通道数，例如 128)
    - $s$ : **Sequence Length (帧数/集合大小，例如 30 帧)** —— **这是 SP 操作的目标维度。**
    - $h, w$ : Height, Width (特征图的高和宽)
- **操作 (Operation):** 在 $s$  维度上应用统计函数（通常是 Max 或 Mean）。
    
    - **Max Pooling:** $z_{max} = \max_{i=1}^{s} (x_{n, c, i, h, w})$
    - **Mean Pooling:** $z_{mean} = \frac{1}{s} \sum_{i=1}^{s} (x_{n, c, i, h, w})$
    - GaitSet 最终主要采用 **Max Pooling**，因为它能更好地保留细粒度特征。
- **输出 (Output):** $\mathbf{Z} \in \mathbb{R}^{n \times c \times h \times w}$
    
    - **结果：** $s$维 度消失（变为 1 并被压缩掉）。现在的特征图不再包含“时间”概念，而是该序列在所有时间步上的统计摘要。

### **思考：为什么这种操作会丢失时序信息？**

这种操作丢失时序信息的根本原因在于它是**置换不变的 (Permutation Invariant)**。

1. **数学原理（置换不变性）：**
    
    - 对于任意的帧顺序排列 $\pi$ ，都有 $G(\{v_1, v_2, ..., v_n\}) = G(\{v_{\pi(1)}, v_{\pi(2)}, ..., v_{\pi(n)}\})$ 。
    - **直观例子：** 假设一个动作序列是“抬腿(A) -> 迈步(B) -> 落脚(C)”。
        
        - 正常顺序输入：$\{A, B, C\}$ $\xrightarrow{Max}$特 征 $Z$ 。
        - 倒放顺序输入：$\{C, B, A\}$ $\xrightarrow{Max}$特 征 $Z$ 。
        - 乱序输入：$\{B, A, C\}$ $\xrightarrow{Max}$ 特征 $Z$ 。
    - 因为 Max 操作只关心“某个特征值是否在序列中出现过”**，而不关心它**“是在第几帧出现的”。因此，所有关于动作先后顺序、因果关系（C 是否由 A 引起）的时间信息都被抹去了。
2. **物理意义（运动包络）：**
    
    - SP 生成的特征实际上是人体在一段时间内运动的“外包络” (Motion Hull)。它记录了手臂摆动的最远位置、腿抬起的最高位置，但丢弃了肢体在这些极值点之间移动的轨迹相位信息。

## 2\. Horizontal Pyramid Mapping (HPM)：空间维度的切片机

在 SP 将时间维度 $s$压 缩掉之后，我们得到的是一张静态的特征图 $(n, c, h, w)$ 。HPM 的任务是在**空间维度 (**$h$ **)** 上进一步提取多尺度信息。

### **操作流程**

1. **切分 (Splitting):**
    
    - 在**高度 (**$h$ **)** 维度上将特征图切分为多个条带 (Strips)。
    - 采用**金字塔结构**（Scales $S$ ）：
        
        - Scale 1: 不切分 (取全身)。
        - Scale 2: 切成 2 份 (取上/下半身)。
        - Scale 3: 切成 4 份... 以此类推。
    - 总共有 $\sum_{s=1}^{S} 2^{s-1}$个 条带（例如 $S=5$时 有 31 个条带）。
2. **全局池化 (Global Pooling on Strips):**
    
    - 对每一个切出来的条带进行 Global Max Pooling (GMP) 和 Global Average Pooling (GAP) 。
    - 这会将空间维度 $h, w$压 缩掉，每个条带变成一个特征向量。
3. **独立映射 (Independent Mapping):**
    
    - **关键点：** 对每一个条带向量，使用**独立的全连接层 (Independent FCs)** 进行映射。
    - 这意味着网络使用一套参数去专门学习“头部”特征，用另一套完全不同的参数去学习“脚部”特征，从而提取出更具判别力的局部细粒度信息。

## 总结

- **Set Pooling (SP)** 作用于 $s$**(S equence)** 维度：通过取最值/均值，将视频序列“压扁”成一张图。代价是丢失了动作发生的先后顺序（时序信息），换取了对输入帧数和顺序的鲁棒性。
- **Horizontal Pyramid Mapping (HPM)** 作用于 $h$ **(Height)** 维度：通过切片和独立全连接层，模拟人类对“局部细节”（如头、躯干、腿）的关注，提取多尺度的空间特征。

* * *

# **2D 轮廓图的固有缺陷和多模态融合的可行性**

## 1\. GaitSet 2D 轮廓图 (Silhouette) 输入的局限性

虽然 GaitSet 通过 Set Pooling 在 CASIA-B 和 OU-MVLP上取得了极佳的效果，但单纯依赖二值化的 2D 轮廓图（Silhouettes）存在天然的物理信息瓶颈：

- **内部纹理与深度信息的彻底丢失**
    
    - **现象**：轮廓图是二值的（0 或 1），这意味着人体内部的所有纹理、颜色和 3D 结构信息都被丢弃了。
    - **后果**：
        
        - **遮挡无法区分**：当手臂摆动到躯干前方时，手臂的像素与躯干的像素融合，二值图上看不出手臂的“前后”关系。
        - **自遮挡导致的信息缺失**：在 $90^\circ$（侧面）视角下，左右腿重叠时很难区分哪条腿在前；在 $0^\circ/180^\circ$ （正视/后视）视角下，步幅（Stride）信息因为缺乏深度而被压缩，导致识别率在这些角度出现局部低值。
- **对衣物与携带物的高度敏感 (Covariate Shift)**
    
    - **现象**：论文实验显示，GaitSet 在 **CL (Wearing a Coat)** 条件下的准确率显著低于 **NM (Normal)** 条件（例如在 ST 设置下，CL 仅为 59.4%，而 NM 为 83.3%）。
    - **原因**：
        
        - **外观改变**：大衣会完全改变人的体型轮廓（看起来比实际更胖/宽）。
        - **运动掩盖**：厚重的衣物会遮挡肢体（四肢和躯干）的运动细节，导致模型提取不到关键的运动特征。单纯的轮廓图无法透过衣物看到内部骨骼的运动。

## 2\. 多模态拓展：融合骨骼 (Skeleton) 与 深度 (Depth)

GaitSet 的 **MGL (Multilayer Global Pipeline)** 和 **Set Pooling (SP)** 架构具有极强的通用性，非常适合拓展到多模态融合。以下是如何将骨骼或深度信息融入 GaitSet 架构的具体思路：

### **A. 骨骼信息 (Skeleton) 的融合策略**

骨骼数据通常表现为关节点的坐标序列（例如 OpenPose 输出的 18 个关键点坐标）。相比轮廓，骨骼对**衣物变化**具有天然的鲁棒性。

- **能否用 Set Pooling 聚合骨骼特征？**
    
    - **答案是肯定的**。GaitSet 的核心假设是“步态即集合”，这一假设同样适用于骨骼序列。
    - **操作逻辑**：
        
        1. **输入**：将骨骼序列视为一组“姿态集合” (Set of Poses)，而非严格的时间序列。
        2. **特征提取 (Frame-level)**：使用轻量级的全连接层 (MLP) 或 图卷积网络 (GCN) 提取每一帧骨骼图的特征。
        3. **聚合 (Set Pooling)**：对 $T$ 帧骨骼特征在时间维度上应用 **Max Pooling**。
        4. **物理含义**：这将提取出该人在行走过程中，通过骨骼关键点表现出的“最大运动幅度”或“特定姿态的极值”（例如膝盖抬起的最大高度），这与 GaitSet 处理轮廓图的逻辑完全一致。

### **B. MGL 架构的拓展 (Dual-Stream Fusion)**

GaitSet 的 MGL 架构通过将不同层级的 Set-level 特征相加来融合信息。我们可以构建一个 **双流 (Dual-Stream)** 或 **多流** 网络，在 MGL 层级进行融合。

- **架构设计**：
    
    1. **轮廓流 (Silhouette Stream)**：保持原有的 GaitSet 主干网络，提取轮廓的 Global Features ($GL_{sil}$ ).
    2. **辅助流 (Skeleton/Depth Stream)**：
        
        - 对于 **深度图 (Depth)**：可以使用与轮廓流相同的 CNN 结构（SetBlock），提取深度的 Global Features ($GL_{depth}$ ).
        - 对于 **骨骼 (Skeleton)**：使用 GCN/MLP 提取特征后，通过 SP 得到骨骼的 Global Features ($GL_{skel}$ ).
    3. **融合点 (Fusion Strategy)**：
        
        - MGP 级融合：论文中 MGP 通过 gl = gl + set_pooling(outs) 累积特征。我们可以在此步骤引入多模态特征：
            
            $GL_{total} = GL_{sil} + \lambda \cdot \phi(GL_{skel})$
            
            其中$\phi$是 一个投影层，用于将骨骼特征维度对齐到轮廓特征维度。
            
        - **互补性利用**：轮廓流提供整体体态信息（胖瘦、发型），骨骼流提供精确的运动结构（不受大衣干扰），深度流补充 $0^\circ/90^\circ$下丢 失的 3D 空间信息。

### **总结**

GaitSet 的 **"Set" 思想** 和 **MGL 架构** 实际上为多模态融合提供了一个非常灵活的框架。只要将其他模态的数据（如骨骼、深度）也视为“集合”，利用 Set Pooling 压缩时间维度，就可以方便地在特征层（Feature-level）与原始轮廓特征进行加法或拼接融合，从而有效解决 2D 轮廓图在 **遮挡** 和 **换衣** 场景下的局限性。