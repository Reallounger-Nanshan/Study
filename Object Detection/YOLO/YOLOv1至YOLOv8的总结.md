# YOLOv1至YOLOv8的总结

## 一、背景及相关知识

### 1.1 YOLO (You Only Look Once) 系列发展时间线

* ![YOLO系列时间线](.\YOLO系列时间线.png)

### 1.2 目标检测指标

#### 1.2.1 精确率 (Precision, P)

* 精确率P用于描述正确检测出的样本数在总样本数中的占比

* $$
  P=\frac{TP}{TP+FP}
  $$

* TP (True Positive) 表示正确识别出的样本数，FP (False Positive) 表示误检的样本数

#### 1.2.2 召回率 (Recall, R)

* 召回率R用于描述检测出的样本数在总样本数中的占比

* $$
  R=\frac{TP}{TP+FN}
  $$

* FN (False Negative) 表示漏检的样本数

#### 1.2.3 IoU (Intersection over Union)

* $$
  IoU=\frac{A\bigcap B}{A\bigcup B}
  $$

* 衡量实际目标 GT 框 (Ground Truth box) A 和预测 box B 之间的重叠程度

#### 1.2.4 平均精度(Average Precision, AP)

* AP 是在单个类别下的平均精度，mAP (mean Average Precision) 是 AP 值在所有类别下的平均精度

* $$
  AP=\int^1_0p\left(r\right)dr
  $$

* 实际计算方式

  * VOC 数据集
    * 对于每个类别，通过改变模型预测的置信度阈值，计算出 PR 曲线
    * 使用精度-召回曲线的内插 11 点抽样，计算每个类别的 AP
    * 对所有 20 个类别中的 AP 取平均来计算最终的 AP
  * **COCO数据集（当前的标准计算方式）**
    * 对于每个类别，通过改变模型预测的置信度阈值，计算出 PR 曲线
    * 使用 101-recall 阈值计算每个类别的 AP——从0到1的101个 R 阈值的精度
    * 在不同的 IoU 阈值下计算 AP，通常从 0.5 到 0.95 ，步长为 0.05 。更高的 IoU 阈值需要更准确的预测才能被认为 TP
    * 对于每个 IoU 阈值，取所有 80 个类别的 AP 的平均值
    * 对每个 IoU 阈值计算的AP取平均来计算总体 AP

### 1.3 非极大值抑制 (Non-Maximum Suppression, NMS)

* 目标检测算法中的一种后处理技术，用于减少重叠box的数量，提高整体检测质量

* ![NMS算法](.\NMS.png)

### 1.4 Backbone, Neck, and Head

#### 1.4.1 背景

* 现代目标检测模型的架构可以描述为 Backbone, Neck 和 Head——YOLOv4 开始遵循此标准

#### 1.4.2 Backbone

* 通常是在大规模图像分类任务上训练的 CNN
* **从不同尺度的图像中提取重要特征**
  * 在较浅的层中提取较低级别的特征
    * 例如边缘和纹理
  * 在较深层中提取较高级别的特征
    * 例如对象部分和语义信息

#### 1.4.3 Neck

* **改善**这些**特征表示**，从而**增强空间信息和语义信息**
* 可能包括额外的卷积层、特征金字塔网络 (Feature Pyramid Networks, FPN) 或其他机制

#### 1.4.4 Head

* 用得到的精细特征来进行目标检测

* 通常**由一个或多个特定于任务的子网络组成**，这些子网络执行分类、定位以及实例分割和姿势估计
* **后处理步骤**（例如NMS）会过滤掉重叠的预测结果，仅保留置信度最高的结果

## 二、YOLOv1

### 2.1 基本信息

#### 2.1.1 作者

* Joseph Redmon (University of Washington)
* Santosh Divvala (University of Washington, Allen Institute for AI)
* Ross Girshick (Facebook AI Research)
* Ali Farhadi (University of Washington, Allen Institute for AI)

#### 2.1.2 提出时间

* 2015

#### 2.1.3 论文

* Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

#### 2.1.4 主要贡献

* ==首次提出一种**实时**的**端到端**的目标检测方法==

### 2.2 模型简析

#### 2.2.1 工作方式

* 将输入图像划分为 S * S 的网格，同时检测所有 bounding box (bbox)，从而统一目标检测步骤
* 每个网格至多检测两个目标，且检测为同一类别

#### 2.2.2 输出

* S \* S \* (B \* 5 + C) 维的张量

  * S = 7

  * B: bbox数，不大于2
  * C: 目标类别总数，实际为20

* ![YOLOv1输出](.\YOLOv1输出.png)
* 参数
  * P~c~: 置信度分数
  * b~x~, b~y~: bbox 中心点坐标
  * b~h~, b~w~: bbox 宽高
  * c~i~: 目标为 i 类的概率

#### 2.2.3 网络架构

* 24 * 卷积层 + 2 \* 全连接层

  * 全连接层用于预测 bbox 坐标和置信度

* 除了最后一层为线性激活函数外，其余均为 leakyReLU (Rectified Linear Unit)

  * ReLU

    * $$
      f\left(x\right)=\begin{dcases}0, & x\le 0;\\ x, & x > 0.\end{dcases}
      $$

    * 缺点

      * 坏死: ReLU 强制的稀疏处理会减少模型的有效容量（即特征屏蔽太多，导致模型无法学习到有效特征）。由于 ReLU 在 x < 0 时梯度为0，这样就导致负的梯度在这个 ReLU 被置零，而且这个神经元有可能再也不会被任何数据激活，称为神经元“坏死”。
      * 无负值: ReLU 和 Sigmoid 的一个相同点是结果是正值，没有负值。

  * LeakyReLU

    * $$
      f\left(x\right)=\begin{dcases}\alpha x, & x\le 0;\\ x, & x > 0.\end{dcases}
      $$

      * α 极小

    * 优点
      * **避免在 x < 0时，网络无法学习的情况**
      * **避免过拟合情况**
      * **计算简单有效**
      * 比 Sigmoid 函数和 Tanh 函数**收敛快**

* 使用**1 \* 1卷积层**来**减少特征图的数量**并保持相对**较低的参数量**

#### 2.2.4 模型训练

* 预训练
  * 使用 ImageNet 数据集在224 \* 224的分辨率下对 YOLO 的前 20 层进行预训练
  * 用随机初始化的权重增加了最后四层
  * 并在 448 \* 448 的分辨率下用 PASCAL VOC 2007 和 VOC 2012 数据集对模型进行微调，以增加细节
* 数据增强
  * 最大为图像大小 20% 的随机缩放和平移
  * HSV 色彩空间中上端系数为 1.5 的随机曝光和饱和度
* 损失函数
  * 由多个误差平方和和组成
  * ![YOLOv1损失函数](.\YOLOv1损失函数.png)

### 2.3 优缺点

#### 2.3.1 优点

* 结构简单
* **全图像一次回归技术实现更快的检测速度**

#### 2.3.2 缺点

* 相较于 Fast R-CNN (Fast Regions with CNN features) 等较为先进的网络模型，YOLOv1 有较大的定位误差
  * 原因
    * 每个网格单元最多只能检测两个同类物体，限制了其预测附近物体的能力
    * **难以预测训练集中宽高比不存在的对象**
    * 由于**下采样**，导致其**只能从粗略的物体特征中学习**



## 三、YOLOv2

### 3.1 基本信息

#### 3.1.1 作者

* Joseph Redmon (University of Washington, XNOR.ai)
* Ali Farhadi (University of Washington, Allen Institute for AI, XNOR.ai)

#### 3.1.2 提出时间

* 2016

#### 3.1.3 论文

* Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7263-7271.

### 3.2 基于YOLOv1的改进

#### 3.2.1 ==批量归一化 (Batch Normalization, BN)==

* 基本信息
  * 作者
    * Sergey Ioffe
    * Christian Szegedy
    * 均为Google Proceedings of the 32nd International Conference on Machine Learning (PMLR)团队成员
  * 论文
    * Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]//International conference on machine learning. pmlr, 2015: 448-456.

* 训练与测试

  * 训练（对于每个 mini-batch）

    * 求出 mini-batch 的 m 和 ρ^2^

    * 对该批次训练数据进行归一化处理

      * $$
        \hat{x}_i\gets\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}}
        $$

      * ε 是为了避免分母为 0 时所使用的微小正数

    * 尺度变换和偏移

      * $$
        y_i\gets\gamma\hat{x}_i+\beta\equiv BN_{\gamma,\beta}\left(x_i\right)
        $$

      * γ 是尺度因子，β 是平移因子——用以解决归一化后由于 x~i~ 被限制在正态分布下所导致的网络表达能力下降的问题

  * 测试

    *  m 和 ρ^2^ 用训练时的 m 和 ρ^2^ 的均值替代

* 作用

  * 对所有卷积层进行 **BN 提高收敛性**
  * 将其充当一个**正则器以减少过拟合**

#### 3.2.2 高分辨率分类器

* 与 YOLOv1 一样，YOLOv2在 224 × 224 的 ImageNet 上预训练模型
* 并在分辨率为 448 × 448 的 ImageNet 上对模型进行了 10 个 epoch 的微调，提高了网络在更高分辨率输入上的性能

#### 3.2.3 ==全卷积==

* 去掉了密集层 (Dense Layer)，采用了全卷积架构
  * 密集层：多层线性层堆叠

#### 3.2.4 ==使用 Anchor 来预测边界盒==

* **Anchor**: 在图像上预设好的不同大小、不同长宽比的参照框

* 使用一组**具有预定义的形状的先验框 Anchor，用于匹配物体的原型形状**
* 每个网格单元都定义了多个 Anchor，系统预测每个 Anchor 的坐标和类别
* 网络输出的大小与每个网格单元的 Anchor 数量成正比

#### 3.2.5 维度聚类

* 好的 Anchor 有助于网络学习预测更准确的 bbox
  * 对训练中的 bbox 进行 k-means 聚类，以找到好的先验
  * 选择五个 Anchor ，在召回率和模型复杂性之间进行了较好的权衡

#### 3.2.6 直接预测位置

* 网络为每个单元预测五个 bbox，每个bbox有五个值 t~x~ , t~y~ , t~w~ , t~h~ , t~o~ 其中 t~o~ 相当于 YOLOv1 的 P~c~

#### 3.2.7 细粒度的特征

* 去掉一个池化层，对于 416×416 的输入图像，得到 13 \* 13 的特征图

#### 3.2.8 ==多尺度训练==

* 由于不使用全连接层，**输入可以是不同尺寸**
* 为了使 YOLOv2 **对不同的输入尺寸具有鲁棒性**，随机训练模型，每十批改变一次输入尺寸 (从 320 \* 320到 608 \* 608)

### 3.3 网络架构

#### 3.3.1 主干架构

* Darknet-19
  * 19 \* 卷积层 + 5 \* max pooling 层

#### 3.3.2 与YOLOv1类似架构

* 在 3 \* 3 之间使用 1 \* 1 的卷积，以减少参数的数量

### 3.4 YOLO9000 (更强的 YOLOv2)

#### 3.4.1 特点

* 使用来自 **COCO** 的**检测标记数据**来学习来自 **ImageNet** 的 **bbox 坐标和分类数据**，以增加模型可以检测的类别数量
* 在训练过程中，结合两个数据集
  * 当使用检测训练图像时，它会反向传播检测网络
  * 当使用分类训练图像时，它会反向传播架构的分类部分

#### 3.4.2 结果

* 改进后的 YOLOv2 模型能够检测 9000 多个类别，因此命名为 YOLO9000



## 四、YOLOv3

### 4.1 基本信息

#### 4.1.1 作者

* Joseph Redmon (University of Washington)
* Ali Farhadi (University of Washington)

#### 4.1.2 提出时间

* 2018

#### 4.1.3 论文

* Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.

### 4.2 基于YOLOv2的改进

#### 4.2.1 Bounding box预测

* 使用逻辑回归 (Logistic Regression, LR) 为每个 bbox 预测一个目标分数
  * LR：一个典型的二值分类器
* 这个分数对于与 GT 框重合度最高的 Anchor 来说是 1，对于其他 Anchor 来说是 0 ——类似 NMS
* 与 Faster R-CNN 相比，YOLOv3 只为每个 GT 框分配一个 Anchor
* 如果没有为一个对象分配 Anchor，它只会产生分类损失，而不会产生定位损失或置信度损失

#### 4.2.2 ==类预测==

* 他们没有使用 Softmax 进行分类，而是使用二元交叉熵来训练独立的 LR 分类器——多标签分类问题
  * Softmax：概率最大的类别作为目标类别
  * 二元交叉熵：对于标签 y 为 1 的情况，如果预测值 p(y) 趋近于 1，那么损失函数的值应当趋近于 0；反之，如果此时预测值 p(y) 趋近于 0，那么损失函数的值应当非常大——符合 log 函数的性质
* **可以给同一个 bbox 分配多个标签**，这可能发生在一些标签重叠的复杂数据集上
  * 例如，同一个物体可以是一个人和一个男人

#### 4.2.3 ==新的主干网络==

* 更大的特征提取器，由带有残差连接的 53 个卷积层组成

#### 4.2.4 ==空间金字塔池 (Space Pyramid Pooling, SPP)==

* 在主干架构中加入了一个改进的 SPP 块，它连接了多个最大池化输出而而无需二次采样，每个具有不同的内核大小 k \* k，其中 k = 1、5、9、13 且允许更大的感受野

  * 感受野
    * 定义：指特征图上的某个点能看到的输入图像的区域，即特征图上的点是由输入图像中感受野大小区域的计算得到的
    * 特点：神经元感受野的值越大表示其能接触到的原始图像范围就越大，也意味着它可能蕴含更为全局、语义层次更高的特征；相反，值越小则表示其所包含的特征越趋向局部和细节
    * 意义：感受野的值可以用来大致判断每一层的抽象层次

  * 输入输出
    * 输入：任意尺寸的 N 维图像
    * 输出：21 \* N 的向量

  * 处理方式
    * 直接对整个特征图池化，每一维得到一个池化后的值，构成一个 1 \* N 的向量
    * 将特征图分成 2 \* 2 共 4 份，每份单独进行池化得到一个 1 \* N 的向量，最终得到 4 个 1 \* N 的向量
    * 将特征图分成 4 \* 4 共 16 份，每份单独进行池化，得到一个 1 \* N 的向量，最终得到 16 个1 \* N 的向量
  * 作用
    * **可以直接向网络中输入原图而不必 resize**

#### 4.2.5 ==**多尺度预测**==

* 类似 FPN，YOLOv3 在三个不同尺度上预测三个 bbox

  * Feature map:
    * 使用神经网络某一层输出的 feature map 进行预测，一般是网络的最后一层 feature map（例如 Fast R-CNN、Faster R-CNN 等）
    * 靠近网络输入层的 feature map 包含粗略的位置信息，导致预测的 bbox 不准确
    * 靠近最后网络最后一层的 feature map 会忽略小物体信息。
  * Feature Pyramid: 使用不同尺寸的 feature map 进行预测

  * FPN: 对最底层的特征进行向上采样，并与该底层特征进行融合，得到高分辨率、强语义的特征
    * 即**加强特征提取，有助于获得更精细的 bbox，并有利于小目标的检测**

#### 4.2.6 **Bounding box先验**

* 与 YOLOv2 一样， YOLOv3也使用 k-means 来确定 Anchor 的 bbox 预设
* 不同的是， 在 YOLOv2 中，他们每个单元共使用了五个先验 bbox，而在 YOLOv3 中，他们使用了三个不同尺度的先验 bbox

### 4.3 网络架构

#### 4.3.1 主干架构

* Darknet-53
  * 52 \* 卷积层 + 23  \* 残差连接 + 全局平均尺寸 + 全连接层 + Softmax 层（YOLOv3 没有使用最后三层）
    * 卷积层：二维卷积层 + BN + LeakyRELU
    * 残差连接将整个网络中 1  \* 1 卷积的输入与 3  \* 3 卷积的输出连接起来

### 4.4 性能评估

* 当 YOLOv3 发布时，目标检测的基准已从 PASCAL VOC 更改为 Microsoft COCO，因此所有 YOLO 都在 MS COCO 数据集中进行评估
* YOLOv3-spp 在 20 FPS 下实现了 36.2% 的 AP 和 60.6% 的 AP~50~，达到了当时最先进的水平，且速度提高了 2 倍



## 五、YOLOv4

### 5.1 基本信息

#### 5.1.1 作者

* Alexey Bochkovskiy
* Chien-Yao Wang (Institute of Information Science, Academia Sinica, Taiwan)
* Hong-Yuan Mark Liao (Institute of Information Science, Academia Sinica, Taiwan)

#### 5.1.2 提出时间

* 2020

#### 5.1.3 论文

* Bochkovskiy A, Wang C Y, Liao H Y M. Yolov4: Optimal speed and accuracy of object detection[J]. arXiv preprint arXiv:2004.10934, 2020.

#### 5.1.4 作者变更

* YOLOv4 保持了 YOLO 前面 3 个版本的理念——实时、开源、端到端和 DarkNet 框架，且改进非常令人满意，社区迅速接受了这个版本作为官方的 YOLOv4

### 5.2 基于YOLOv3的改进

#### 5.2.1 具有 Bag-of-Specials (BoS) 集成的增强架构

* Backbone

  * 使用具有跨阶段部分连接的 Darknet-53 结构 (Cross Stage Partial Connections Network, CSPNet) 和 Mish 激活函数
    * CSPNet: 将梯度的变化从头到尾地集成到特征图中，增强CNN网络的学习能力，减少了计算量以降低模型计算瓶颈和算法的内存成本，同时可以保持模型精度
    * Mish激活函数: 一种自正则的非单调神经激活函数，平滑的激活函数允许更好的信息深入神经网络，从而得到更好的准确性和泛化
  * Darknet-53中添加的 CSPNet 有助于减少计算量，且同时保持相同的精度

* Neck

  * 使用了 来自 YOLOv3-spp 的 SPP 块和 YOLOv3 中的多尺度预测，但使用修改后的路径聚合网络 (Path Aggregation Network, PANet)  和空间注意模块 (Spatial Attention Module, SAM) 代替 FPN
    * PANet: 提出了一个自顶向下和自底向上的双向融合主干网络，同时在最底层和最高层之间添加了一条 short-cut，用于缩短层与层之间的路径
    * SAM: 区别于通道注意力 (Channel Attention) 是为了筛选出哪些通道的信息是和目前这个认为是相关的，**空间注意力则是去关心对于特征图来说哪些位置的信息是和目前认为相关的**

  *  SPP 块在不影响推理速度的情况下增加了感受野
  * 修改后的 PANet 被用于连接特征，而不是像原始 PANet 论文中那样增加特征

* Head

  * 使用 YOLOv3 中的 Anchor

* 模型名称：CSPDarknet53-PANet-SPP

#### 5.2.2 集成 Bag-of-Freebies (BoF) 以实现高级训练方法

* 数据增强
  * 常规数据增强方法
    * 随机亮度、对比度、缩放、裁剪、翻转和旋转
  * ==**马赛克 (Mosaic) 增强**==
    * 操作：将四张图像进行随机裁剪，再拼接到一张图像上作为训练数据
    * 优点：丰富了图像的背景，且四张图像拼接在一起变相地提高了 batch_size
    * 意义：可用于BN的大量小图像数据
  
*  **正则化**
  
  * 目的

    * **防止模型过拟合**
  
  * 原理
  
    * 在损失函数上加上某些规则/限制，以缩小解空间，从而减少求出过拟合解的可能性
  
    * $$
      J\left(\theta\right)=L\left(\theta\right)+\lambda\Phi\left(\theta\right)
      $$
  
    * J (θ) 为模型目标函数，L (θ) 为损失函数，λ 为正则化系数，Φ (θ) 为正则化函数——一般是一个过原点的凸函数
  
  * YOLOv4 方法
    * **用 DropBlock 替代 Dropout**
      * Dropout: 随机删除一些输入数据——对于 2 维图像而言，一些散点的删除并不会影响语义信息
      * DropBlock: 随机删除一些连续输入数据
  * 用于卷积神经网络以及类标签平滑
  
* 检测器

  * ==**添加了 CIoU 损失**==

    * $$
      CIOU_{LOSS}=1-IOU+\frac{\rho^2\left(b,b_{GT}\right)}{c^2}+\frac{v^2}{\left(1-IOU\right)+v}\\
      v=\frac{4}{\pi^2}\left(\arctan\left(\frac{w_{GT}}{h_{GT}}\right)-\arctan\left(\frac{w}{h}\right)\right)^2
      $$

    * 𝜌 (b, b~GT~) 为预测框和 GT 框的中心点的欧氏距离，v 为衡量长宽比一致性的参数，h~GT~ 和 w~GT~ 是 GT 框的长宽大小，h 和 w 为 GT 框的长宽大小

    * 与前代相比优化

      * IoU: 无法优化两个框不相交的情况
      * GIoU: 无法解决预测框在 GT 框内部且大小一致的情况，即无法区分相对位置关系
      * DIoU: 无法解决预测框和 GT 框中心距离相同但尺寸不一样的问题

  * 添加了 Cross mini-Batch Normalization (CmBN)

    * 用于从整个批次收集统计数据，而不是像常规BN那样从单个小批次收集统计数据

#### 5.2.3 ==自我对抗训练 (Self-adversarial Training, SAT)==

* 为了使模型对扰动更加鲁棒，对输入图像进行对抗性攻击来制造一个欺骗——即 GT 对象不在图像中，但保留原始标签，从而检测正确的对象。

#### 5.2.4 ==使用遗传算法进行超参数优化==

* 为了找到用于训练的最佳超参数，在前 10% 的周期中使用遗传算法 (Genetic Algorithm, GA)，并使用余弦退火调度程序来改变训练期间的学习率 (learning rate, lr)
  * GA
    * 特点：随机全局搜索优化方法
    * 主要思想
      * 模拟了自然选择和遗传中发生的复制、交叉和变异等现象
      * 从任一初始种群出发，通过随机选择、交叉和变异操作，产生一群更适合环境的个体
      * 使群体进化到搜索空间中越来越好的区域
      * 一代代不断繁衍进化
      * 最后收敛到一群最适应环境的个体——从而求得问题的优质解
  * 梯度下降
    * 基本思想：当越来越接近 Loss 值的全局最小值时，学习率应该变得更小来使得模型尽可能接近这一点
    * 缺点：可能陷入局部最优解
  * 余弦退火
    * 基本思想
      * Warm up stage: lr 由较小值逐渐增大的过程
      * Annealing stage：先缓慢下降，然后加速下降，最后再次缓慢下降

### 5.3 性能评估

* 在 MS COCO 数据集 test-dev 2017 上进行评估，YOLOv4 在 NVIDIA V100 上以超过 50 FPS 的速度实现了 43.5% 的 AP 和 65.7% 的 AP~50~



## 六、YOLOv5

### 6.1 基本信息

#### 6.1.1 作者

* Glenn Jocher (Founder and CEO of Ultralytics)

#### 6.1.2 提出时间

* 2020

#### 6.1.3 开源代码

* G. Jocher,“YOLOv5 by Ultralytics.”https://github.com/ultralytics/yolov5, 2020. Accessed: February 30, 2023.

### 6.2 基于YOLOv4的改进

#### 6.2.1 简介

* 相较于YOLOv4，YOLOv5 并无较大的理论提升，主要改进体现在实际**工程应用**方面，最主要的区别在于YOLOv5 摒弃了前代 YOLO 系列的 Darknet 框架而选择使用 **Pytorch**

#### 6.2.2 网络架构

* Backbone
  * **修改后的 Darknet-53 结构**
    * Stem 层：具有大窗口大小的跨步卷积层，以减少内存和计算成本
    * 卷积层：从输入图像中提取相关特征
    *  快速空间金字塔池化 ( Space Pyramid Pooling - Fast, SPPF) + 卷积层：处理各种尺度的特征
      * SPPF层旨在通过将不同尺度的特征池化为固定大小的特征图来加速网络的计算
    * 上采样层：增加特征图的分辨率
    * 每个卷积之后都进行 BN 和 SiLU (Sigmoid Linear Unit) 激活
      * SiLU: 相对于ReLU函数，**SiLU函数在接近零时具有更平滑的曲线**，并且由于其使用了Sigmoid函数，可以**使网络的输出范围在0和1之间**

* Neck
  * FPN + PAN 结构
    * FPN层自顶向下传达强语义特征
    * 包含两个PAN结构的特征金字塔则自底向上传达强定位特征
    * 两者从不同的主干层对不同的检测层进行参数聚合。
  * **借鉴 CSPNet 设计 CSP2 结构，从而增强网络的特征融合能力**
* Head
  * 类似于 YOLOv3
  * 和YOLOv4一样采用 CIoU-Loss 函数作为损失函数

#### 6.2.3 数据增强

* Mosaic、复制粘贴、随机仿射、MixUp、HSV 增强、随机水平翻转以及来自 albumentations 包的其他增强功能

#### 6.2.4 ==自适应图片缩放==

* 背景
  * 对于目标检测算法，通常需要对图片进行缩放到一个固定的尺寸，并在较短的两端填充黑边，再将其送至检测网络中
  * 由于实际使用中的图片的长宽比不同，因此缩放后填充的黑边大小也不甚相同，如果**填充的黑边过多会导致大量的冗余信息的存在，从而影响整个算法的推理速度**

* 作用

  * 添加最少的黑边到缩放后的图片中

* 方法

  * 计算原始图片大小与输入到网络的图片大小的缩放比

    * $$
      \begin{dcases}r_l=height_{scale}/height_{raw},\\r_w=width_{scale}/width_{raw}.\end{dcases}
      $$

    * r~l~ 和 r~w~ 分别为图片长宽的缩放比，height~scale~ 和width~scale~ 是输入到网络的图片的长宽大小，height~raw~ 和 width~raw~ 是原始图片的长宽大小

  * 计算缩放后图片一端的黑边的填充数值

    * $$
      d=\frac{1}{2}mod\left(|height_{raw}*r_l-width_{raw}*r_w|, 32\right)
      $$

    * mod [param1, param2] 为取余函数

    * 因为 YOLOv5 网络执行了 5 次下采样操作，所以需要对缩放后长宽差值的绝对值除以 2^5^

#### 6.2.5 ==自动锚框 (AutoAnchor)==

* 如果锚框不适合数据集和训练设置（例如图像大小），此预训练工具会检查和调整锚框
  * 首先将 k-means 函数应用于数据集标签，以生成遗传进化 (Genetic Evolution, GE) 算法的初始条件
  *  然后GE 算法默认将这些 Anchor 演化超过 1000 代，使用 CIoU-Loss 和最佳可能召回 (Best Possible Recall, BPR) 作为其适应度函数。 

#### 6.2.6 其它改进

* 提高了网格灵敏度，使其对失控梯度更加稳定

### 6.3 特点

#### 6.3.1 ==开源==

* 由 Ultralytics 积极维护，有超过 250 名贡献者，并且经常有新的改进

#### 6.3.2 ==易于使用、训练和部署==

* 环境配置简单
* 操作简便
* Ultralytics 提供适用于 iOS 和 Android 的移动版本以及许多用于标签、训练和部署的集成。 

#### 6.3.3 ==提供了一组不同大小的模型==

* YOLOv5提供了五个基础的缩放版本：YOLOv5n、YOLOv5s、YOLOv5m、YOLOv5l 和 YOLOv5x，其中卷积模块的宽度和深度各不相同，以**适应特定的应用和硬件要求**
* YOLOv5n 和 YOLOv5s 是针对低资源设备的轻量级模型
* YOLOv5x 则以牺牲速度为代价针对高性能进行了优化
* 目前 YOLOv5 提供了**支持目标检测、目标分类和实例分割**的不同版本。

### 6.4 性能评估

* 在 MS COCO 数据集 test-dev 2017 上进行评估，YOLOv5x 在图像大小为 640 像素的情况下获得了 50.7% 的 AP
* 使用 32 的 batch_size，可以在 NVIDIA V100 上实现 200 FPS 的速度
* 使用 1536 像素的较大输入尺寸和测试时增强 (Test-Time Augmentation, TTA)，YOLOv5 实现了 55.8% 的 AP
  * TTA: 在推理（预测）阶段，将原始图片进行水平翻转、垂直翻转、对角线翻转、旋转角度等数据增强操作，得到多张图，分别进行推理，再对多个结果进行综合分析，得到最终输出结果



## 七、Scaled-YOLOv4

### 7.1 基本信息

#### 7.1.1 作者

* Chien-Yao Wang (Institute of Information Science, Academia Sinica, Taiwan)
* Alexey Bochkovskiy (Intel Intelligent Systems Lab)
* Hong-Yuan Mark Liao (Institute of Information Science, Academia Sinica; Department of Computer Science and Information Engineering, Providence University, Taiwan)

#### 7.1.2 提出时间

* 2021

#### 7.1.3 论文

* Wang C Y, Bochkovskiy A, Liao H Y M. Scaled-yolov4: Scaling cross stage partial network[C]//Proceedings of the IEEE/cvf conference on computer vision and pattern recognition. 2021: 13029-13038.

### 7.2 基于YOLOv4的改进

#### 7.2.1 深度学习框架

* 和 YOLOv5 一样，Scaled-YOLOv4 采用 Pytorch 框架

#### 7.2.2 ==放大和缩小技术==

* 放大意味着生成一个模型，以降低速度为代价来提高精度

* 缩小规模需要生成一个提高速度但牺牲准确性的模型，同时对算力的要求更低，并且可以在嵌入式系统上运行
* **该技术思路与 YOLOv5 提供的一组不同大小的模型相同**

### 7.3 性能评估

#### 7.3.1 缩小后的模型架构：YOLOv4-tiny

* 它是为低端 GPU 设计的
* 可以在 Jetson TX2 上以 42 FPS 运行
* 在 RTX2080Ti 上使用 TensorRT FP6 进行推理可以达到 1774 FPS
  * TensorRT: Nvidia 的一个高性能的深度学习推断的优化器和运行的引擎
* 在 MS COCO 上实现 22% AP

#### 7.3.2 放大后的模型架构：YOLOv4-large

* 包括三种不同尺寸的 P5  、P6 和 P7
* 该架构专为云 GPU 设计，实现了最先进的性能
* YOLOv4-P7 在 MS COCO 上以 56% 的 AP 超越了之前的所有模型



## 八、YOLOR (You Only Learn One Representation)

### 8.1 基本信息

#### 8.1.1 作者

* Chien-Yao Wang (Institute of Information Science, Academia Sinica, Taiwan)
* I-Hau Yeh (Elan Microelectronics Corporation, Taiwan)
* Hong-Yuan Mark Liao (Institute of Information Science, Academia Sinica, Taiwan)

#### 8.1.2 提出时间

* 2021

#### 8.1.3 论文

* Wang C Y, Yeh I H, Liao H Y M. You only learn one representation: Unified network for multiple tasks[J]. arXiv preprint arXiv:2105.04206, 2021.

### 8.2 主要贡献

#### 8.2.1 ==引入隐式知识==

* 当需要同时训练一个被多个任务共享的模型时，由于损失函数的联合优化过程是必须执行的，因此在执行过程中往往会出现多方相互拉动的情况，这种情况将导致最终的整体性能比单独训练多个模型然后集成它们要差
* 为了解决这个问题，YOLOR 为训练多任务训练了一个规范的表征，给每个任务分支引入隐式表征增强表征能力

#### 8.2.2 ==核空间对齐==

* 核空间
  * 对于矩阵 A，使得 Ax = 0 的所有向量 x 所组成的空间
* 背景
  * 在多任务和多 Head 神经网络中，核空间不对齐是经常发生的问题

* 方法
  * 对输出特征和隐式表征进行加法和乘法运算，这样就可以对核空间进行变换、旋转和缩放，以对齐神经网络的每个输出核空间

#### 8.2.3 预测细化

* 使用简单的向量隐式表征和加法算子，在 YOLOR 的每一个输出层添加隐式知识进行预测细化，大部分指标都获得到了一定的增益

### 8.3 性能评估

* 在 MS COCO 数据集 test-dev 2017 上进行评估，YOLOR 在 NVIDIA V100 上以 30 FPS 的速度实现了 55.4% 的 AP 和 73.3% 的 AP~50~



## 九、YOLOX

### 9.1 基本信息

#### 9.1.1 作者

* Zheng Ge
* Songtao Liu
* Feng Wang
* Zeming Li
* Jian Sun
* 均为 Megvii Technology 成员

#### 9.1.2 提出时间

* 2021

#### 9.1.3 论文

* Ge Z, Liu S, Wang F, et al. Yolox: Exceeding yolo series in 2021[J]. arXiv preprint arXiv:2107.08430, 2021.

### 9.2 基于YOLOv3的改进

#### 9.2.1 简介

* YOLOX 基于 Ultralytics 的 YOLOv3-spp，使用 Pytorch 进行开发

#### 9.2.2 ==Anchor-Free==

* 自 YOLOv2 以来，所有后续的 YOLO 版本都是基于 Anchor 的检测器
* 受到 CornerNet、CenterNet 和 FCOS 等无 Anchor 最先进的目标检测器的启发，回归到简化训练和解码过程的无 Anchor 架构
* 使AP 增加了 0.9 个点

#### 9.2.3 Multi positive

* 为了补偿由于缺少 Anchor 而产生的巨大不平衡，YOLOX 使用中心采样，将中心 3 \* 3 区域指定为正值
* 这使 AP 增加了 2.1 点

#### 9.2.4 ==解耦 Head==

* 分类置信度和定位精度之间可能存在偏差
* YOLOX 将这 Head 分为两个部分，一个用于分类任务，另一个用于回归任务
* 使 AP 提高了 1.1 个点，并加快了模型收敛速度

#### 9.2.5 先进的标签分配

* 当多个对象的框重叠时，GT 标签分配可能会产生歧义，并将分配过程表述为最优传输 (Optimal Transport, OT) 问题
  * OT: 在可分度量空间中，讨论概率测度间最优传输变换的一类优化问题
* YOLOX 受到这项工作的启发，提出了一个名为 ==**simOTA**== 的简化版本
  * simOTA
    * 逻辑
      * 确定正样本候选区域
      * 计算 Anchor 与 GT 的 IoU
      * 在候选区域内计算 cost
      * 使用 IoU 确定每个 GT 的 dynamic_k
      * 为每个 GT 取 cost 排名最小的前 dynamic_k 个 Anchor 作为正样本，其余为负样本
      * 使用正负样本计算 Loss
    * 优点
      * simOTA 能够做到**自动的分析每个 GT 要拥有多少个正样本**
      * 能自动决定**每个 GT 要从哪个特征图来检测**
      * 相较 OTA，simOTA 运算速度更快
      * 相较 OTA，避免额外超参数
* 使 AP 增加了 2.3 点

#### 9.2.6 强力增强

* YOLOX 使用 ==**MixUP**== 和 Mosaic 增强——使用这些增强功能后，ImageNet 预训练不再有益

  * MixUP

    * 主要思想：对两个样本-标签数据对按比例相加后生成新的样本-标签数据

    * $$
      \begin{dcases}
      \widetilde{x}=\lambda x_i+\left(1-\lambda\right)x_j,\\
      \widetilde{y}=\lambda y_i+\left(1-\lambda\right)y_j.
      \end{dcases}
      $$

    * λ ∼ Beta (α, α), λ ∈ [0, 1]

    * 优点：具有好的泛化性能和鲁棒性，无论对于含噪声标签的数据还是对抗样本攻击，都表现出不错的鲁棒性

* 使 AP 增加 2.4 点

### 9.3 性能评估

* 在 Tesla V100 上实现了速度和准确性之间的最佳平衡，AP 为 50.1%，FPS 为 68.9



## 十、YOLOv6

### 10.1 基本信息

#### 10.1.1 作者

* Chuyi Li
* Lulu Li
* Hongliang Jiang
* Kaiheng Weng
* Yifei Geng
* Liang Li
* Zaidan Ke
* Qingyuan Li
* Meng Cheng 
* Weiqiang Nie
* Yiduo Li
* Bo Zhang
* Xiaoming Xu
* 均为美团视觉AI部门成员

#### 10.1.2 提出时间

* 2022

#### 10.1.3 论文

* Li C, Li L, Jiang H, et al. YOLOv6: A single-stage object detection framework for industrial applications[J]. arXiv preprint arXiv:2209.02976, 2022.

### 10.2 主要创新点

#### 10.2.1 ==网络设计==

* Backbone
  * 背景
    * RepVGG 主干在小型网络中具有**更强的特征表示能力**，但是随着参数和计算成本的爆炸式增长， 其在大模型中难以获得较高的性能
  * 改进
    * 设计了一个高效的可重新参数化的骨干，称为 **EffificientRep**
    * 在小模型中，使用 **RepBlock**
    * 在大模型中，使用 **CSPStackRep Blocks**
  * 具体方法
    * 将 Backbone 中 stride = 2 的普通 Conv 层替换成了 stride = 2 的重新参数化卷积 (Re-parameterized Convolutions, RepConv) 层
    * 将原始的 CSP-Block 都重新设计为 RepBlock，其中 RepBlock 的第一个 RepConv 会做通道维度的变换和对齐
    * 将原始的 SPPF 优化设计为更加高效的 SimSPPF

* Neck
  * 背景
    * 参考 YOLOv4 和 YOLOv5 用的 PAN，结合 Backbone 里的 RepBlock 或 CSPStackRep，提出了一个 **Rep-PAN**
  * 方法
    * Rep-PAN 基于 PAN拓扑方式，用 RepBlock 替换了 YOLOv5 中使用的 CSP-Block，对整体 Neck 中的算子进行了调整
  * 优点
    * 在硬件上达到**高效推理**的同时，保持**较好的多尺度特征融合能力**
* Head
  * 方法
    * 将中间的 3 \* 3 卷积层的数量减少为 1
    * Head 的尺度和 Backbone 及 Neck 同大同小
  * 优点
    * 进一步降低了计算成本，以实现更低的推断延迟

#### 10.2.2 ==标签分配==

* 背景
  * SimOTA 会拉慢训练速度，容易导致训练不稳定
* 方法
  * 采取 TOOD (Task-aligned One-stage Object Detection) 所提出的**任务对齐学习 (TAL)**，该方法比 SimOTA 具有更好的性能改善，使训练更加稳定
    * TAL: 通过设计一个样本分配方案和一个与任务相关的损失来执行的。样本分配通过计算每个 Anchor 的任务对齐度来收集训练样本（正样本或负样本），而任务对齐损失逐渐统一最佳 Anchor，以便在训练期间预测分类和定位

#### 10.2.3 ==新的分类和回归 Loss==

* 分类损失
  * VariFocal Loss (VFL): 非对称的加权操作——针对正负样本有不平衡的问题和正样本中不等权的问题，来发现更多有价值的正样本
* 回归框损失
  * SIoU: 小模型上提升明显
  * GIoU: 大模型上提升明显
  * 分别应用于 YOLOv6 不同大小的模型

#### 10.2.4 ==自蒸馏==

* 知识蒸馏技术
  * 核心思想
    * 先训练一个复杂网络模型，然后使用这个复杂网络的输出和数据的真实标签去训练一个更小的网络
  * 组成
    * 知识蒸馏框架通常包含了一个复杂模型 (Teacher 模型) 和一个小模型 (Student 模型)

* 由于限制了 Teacher 模型与 Student 模型网络结构相同，但经过预训练，因此称为自蒸馏

#### 10.2.5 量化方案

* 使用 RepOptimizer 和逐通道蒸馏进行检测的量化方案，有助于实现更快的检测器

  * 模型量化：将网络模型的权值、激活值等从高精度转化成低精度的操作过程

  * RepOptimizer
    * 梯度重参数化，将先验信息用于修改梯度

### 10.3 性能评估

* 作者提供了八种缩放模型，从 YOLOv6-N 到 YOLOv6-L6
* 在 MS COCO 数据集 test-dev 2017 上进行评估，最大的模型在 NVIDIA Tesla T4 上以大约 29 FPS 的速度实现了 57.2% 的 AP



## 十一、YOLOv7

### 11.1 基本信息

#### 11.1.1 作者

* Chien-Yao Wang (Institute of Information Science, Academia Sinica, Taiwan)
* Alexey Bochkovskiy
* Hong-Yuan Mark Liao (Institute of Information Science, Academia Sinica, Taiwan)

#### 11.1.2 提出时间

* 2022

#### 11.1.3 论文

* Wang C Y, Bochkovskiy A, Liao H Y M. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 7464-7475.

### 11.2 主要创新点

#### 11.2.1 架构改进

* ==**扩展高效层聚合网络 (Extended Efficient Layer Aggregation Network, E-ELAN)**==
  * ELAN 是一种策略，其允许深度模型通过控制最短最长梯度路径来更有效地学习和收敛
  * 适用于具有无限堆叠计算块的模型
  * E-ELAN 通过搅乱和合并基数来结合不同群体的特征，在不破坏原始梯度路径的情况下增强网络的学习能力
* ==**基于串联模型的模型缩放**==
  * 缩放通过调整一些模型属性来生成不同尺寸的模型
  * YOLOv7 的架构是一种基于级联的架构，其中标准缩放技术（例如深度缩放）会导致过渡层的输入通道和输出通道之间的比率发生变化，从而导致模型硬件利用率的降低
  * YOLOv7 提出了一种新的基于串联的缩放模型的策略，其中块的深度和宽度以相同的因子缩放以保持模型的最佳结构

#### 11.2.2 ==BoF 改进==

* 设计新的 RepConv
  * 背景
    * 与 YOLOv6 一样，YOLOv7 的架构也受到 RepConv 的启发
    * RepConv 中的恒等连接破坏了 ResNet 中的残差和 DenseNet 中的串联
  * 方法
    * 删除了身份连接并将其称为 RepConvN
* 辅助 Head 的粗略标签分配和引导 Head 的精细标签分配
  * 主 Head 负责最终输出
  * 副 Head 协助训练
* Conv-bn-activation 中的批量归一化
  * 将 BN 的均值和方差整合到推理阶段的卷积层的偏差和权重中
*  隐式知识
  * 受 YOLOR 启发引入隐式知识
* 指数移动平均线
  * 作为最终的推理模型

### 11.3 性能评估

#### 11.3.1 与 YOLOv4 比较

* 相比 YOLOv4，YOLOv7 减少 75% 参数和 36% 计算量，同时 AP 提高 1.5%
* 相比 YOLOv4-tiny，YOLOv7-tiny 在保持相同 AP 的情况下，分别减少了 39% 和 49% 的参数和计算量

#### 11.3.1 与 YOLOR 比较

* 相比 YOLOR，YOLOv7 的参数和计算量分别减少了 43% 和 15%，AP 略有增加 0.4%

#### 11.3.1 数据集测试

* 在 MS COCO 数据集 test-dev 2017 上进行评估，YOLOv7-E6 在 NVIDIA V100 上以 1280 像素的输入大小和 50 FPS 的速度实现了 55.9% 的 AP 和 73.5% 的 AP50。



## 十二、DAMO-YOLO

### 12.1 基本信息

#### 12.1.1 作者

* Xianzhe Xu
* Yiqi Jiang
* Weihua Chen
* Yilun Huang
* Yuan Zhang
* Xiuyu Sun
* 均为阿里巴巴集团的成员

#### 12.1.2 提出时间

* 2022

#### 12.1.3 论文

* Xu X, Jiang Y, Chen W, et al. Damo-yolo: A report on real-time object detection design[J]. arXiv preprint arXiv:2211.15444, 2022.

### 12.2 主要创新点

#### 12.2.1 ==神经架构搜索 (Neural architecture search, NAS)==

* 使用阿里巴巴开发的一种名为 MAE-NAS 的方法来**自动寻找高效的架构**

#### 12.2.2 较大的 Neck

* 受到 GiraffeDet 、CSPNet 和 ELAN 的启发，作者设计了一种**可以实时工作的 Neck**，称为 Efficient-RepGFPN

#### 12.2.3 较小的 Head

* 只留下一个用于分类的线性层和一个用于回归的线性层，称为 ZeroHead

#### 12.2.4 ==AlignedOTA标签分配==

* 背景
  * OTA 和 TOOD 等动态标签分配方法由于相对于静态方法的显着改进而受到欢迎
  * 然而这类方法依旧无法解决分类和回归之间的错位的问题，部分原因是分类和回归损失之间的不平衡
* 方法
  * AlignOTA 方法将焦点损失引入到分类成本中，并使用预测和 GT 框的 IoU 作为软标签，从而能够**为每个目标选择对齐的样本**，并**从全局角度解决问题**

#### 12.2.5 知识蒸馏

* 策略包括两个阶段
  * 第一阶段由 Teacher 指导 Student
  * 第二阶段由 Student 自主微调
* 融入了两项增强功能
  * 对齐模块
    * 使 Student 特征适应与 Teacher 相同的分辨率
  * 通道动态温度
    * 使 Teacher 和 Student 特征标准化，以减少实际值差异的影响

### 12.3 性能评估

* 最佳模型在 NVIDIA V100 上以 233 FPS 的速度实现了 50.0% 的 AP



## 十三、YOLOv8

### 13.1 基本信息

#### 13.1.1 作者

* Glenn Jocher (Founder and CEO of Ultralytics)

#### 13.1.2 提出时间

* 2023

#### 13.1.3 开源代码

* G. Jocher, A. Chaurasia, and J. Qiu, “YOLO by Ultralytics.” https://github.com/ultralytics/ultralytics, 2023. Accessed: February 30, 2023.

### 13.2 基于 YOLOv5 的改进

#### 13.2.1 Backbone

* 整体主干网络与 YOLOv5 类似
* 用 **C2f (Cross-stage partial bottleneck with two convolutions) 模块代替了 CSP 层**
  * C2f
    * 与 CSP 区别：**增加了更多的跳层连接和额外的 Split 操作**
      * Split: 对训练过程中在生成文件的文件名进行切割，从而得知是哪一轮的数据
    * 意义：将高级特征与上下文信息相结合，以提高检测精度

#### 13.2.2 Neck

* 同样采用 C2f 模块，但与 Backbone 的略有区别

#### 13.2.3 Head

* 使用具有**解耦 Head 的无锚模型**来独立处理对象性、分类和回归任务

  * 允许每个分支专注于其任务并提高模型的整体准确性

* 激活函数

  * Sigmoid 函数

    * 作为 objectness 分数的激活函数
    * 表示 bbox 包含对象的概率

  * Softmax 函数

    * 作为类概率

    * 表示对象属于每个可能类的概率

* 损失函数

  * CIoU + DFL (Distribution Focal Loss) 损失函数

    * DFL
      * 以交叉熵的形式去优化与标签 y 最接近的一左一右 2 个位置的概率，从而让网络更快的聚焦到目标位置的邻近区域的分布
      * 即学习出的分布理论上是在真实浮点坐标的附近，并且以线性插值的模式得到距离左右整数坐标的权重

    * 作为 bbox 损失

  * 二元交叉熵

    * 作为分类损失

  * 效果

    * 提高了目标检测性能，尤其是对于较小目标的检测

### 13.3 特点

* 继承 YOLOv5 特点
* 可以从命令行界面 (Command Line Interface, CLI) 运行
* ==**可以作为 PIP 包安装**==

### 13.4 性能评估

* 在 MS COCO 数据集 test-dev 2017 上进行评估，YOLOv8x 在 640 像素的图像大小下实现了 53.9% 的 AP（相比之下，相同输入大小的 YOLOv5 为 50.7%），在 NVIDIA A100 和 TensorRT 上的速度为 280 FPS



## 十四、PP-YOLO系列

### 14.1 基本信息

#### 14.1.1 作者

* Xiang Long
* Kaipeng Deng
* Guanzhong Wang
* Yang Zhang
* Qingqing Dang
* Yuan Gao
* Hui Shen
* Jianguo Ren
* Shumin Han
* Errui Ding
* Shilei Wen
* 均为百度的研究团队成员

#### 14.1.2 提出时间

* PP-YOLO: 2020

* PP-YOLov2: 2021
* PP-YOLOE: 2022

#### 14.1.3 论文

* PP-YOLO
  * Long X, Deng K, Wang G, et al. PP-YOLO: An effective and efficient implementation of object detector[J]. arXiv preprint arXiv:2007.12099, 2020.

* PP-YOLOv2
  * Huang X, Wang X, Lv W, et al. PP-YOLOv2: A practical object detector[J]. arXiv preprint arXiv:2104.10419, 2021.
* PP-YOLOE
  * Xu S, Wang X, Lv W, et al. PP-YOLOE: An evolved version of YOLO[J]. arXiv preprint arXiv:2203.16250, 2022.

### 14.2 基于 YOLOv3 的改进

#### 14.2.1 简介

* PP-YOLO **基于 PaddlePaddle 深度学习平台**进行开发，最初基于 YOLOv3 进行改进，此后版本则在 PP-YOLO 系列前作上进行改进
* PP-YOLO 对 YOLO 的演变产生了影响
* 遵循从 YOLOv4 开始看到的趋势，PP-YOLO **添加了 10 个现有技巧**来提高检测器的准确性，同时保持速度不变
* 作者表示，提出 PP-YOLO 的目的并不是介绍一种新颖的物体检测器，而是展示如何逐步构建更好的检测器
* PP-YOLO 使用的大多数技巧与 YOLOv4 中使用的技巧不同，并且相同的技巧也采取不同的方式实现

#### 14.2.2 Backbone

* **ResNet50-vd 主干网络取代了 DarkNet-53 主干网络**
* 在最后阶段使用了**可变形卷积 (deformable convolutions net, dcn) 增强的架构**，以及经过提炼的预训练模型——该模型在 ImageNet 上具有更高的分类精度
* 该架构称为 **ResNet5-vd-dcn**

#### 14.2.3 更大的 batch_size

* 将 batch_size 从 64 个增加到 192 个
  * 提高训练稳定性
* 更新了训练计划和 lr

#### 14.2.4 保持移动平均 (Moving Averages, MA)

* MA
  * 基本思想
    * 通过计算过去数据的平均值得到一个数学结果
  * 作用
    * 用来衡量当前趋势的方向

* 保持训练参数的MA，并用它们代替最终的训练值

#### 14.2.5 DropBlock

* 仅应用于 FPN

#### 14.2.6 损失函数

* IoU + L1-Loss

  * L1-Loss: 即平均绝对误差 (Mean Absolute Error, MAE)，指模型预测值 f (x) 和真实值 y 之间绝对差值的平均值

  * 用于 bbox 回归

*  IoU 预测分支 +  IoU 感知损失

* 推理过程与 YOLOv3 的区别

  * YOLOv3: 将分类概率和客观性分数相乘来计算最终的检测结果
  * PP-YOLO: 在YOLOv3 的基础上乘以预测的 IoU 以考虑定位精度

#### 14.2.7 网格敏感方法 (Grid Sensitive approach)

* 采用类似于 YOLOv4 的网格敏感方法来改进网格边界处的 bbox 中心预测

#### 14.2.8 Matrix NMS

* 通过并行运算排序过后的上三角化的 mask-IoU 矩阵，加快计算速度
* 采取分数惩罚机制来抑制冗余 mask

#### 14.2.9 CoordConv

* 将 CoordConv 用于 FPN 的 1 \* 1 卷积和检测 Head 中的第一个卷积层上

* CoordConv 允许网络学习平移不变性，从而改善检测定位

#### 14.2.10 SPP

* 仅在顶部特征图上使用，以增加 Backbone 的感受野

#### 14.2.11 数据增强

* 混合训练
  * 使用从 Beta (α, β) 分布采样的权重进行混合训练
  * α = 1.5, β = 1.5
* 随机颜色失真
* 随机膨胀
* 概率为0.5的随机裁剪和随机翻转
* RGB 通道 z 分数归一化
  * m = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]
* 从 [320,352,384,416,448,480,512,544,576,608] 中均匀抽取多个图像尺寸

### 14.3 PP-YOLOv2 的改进

#### 14.3.1 Backbone

* 用ResNet101 替代 ResNet50

#### 14.3.1 PAN

* 和YOLOv4 一样用 PAN 替代 FPN

#### 14.3.1 Mish激活函数

* 仅在检测模型的 Neck 中应用了 mish 激活函数，以保持带有 ReLU 的 Backbone 不变

#### 14.3.1 更大的输入尺寸

* 将最大输入大小从 608 扩展到 768，并将每个 GPU 的批处理大小从 24 个图像减少到 12 个图像
* 输入大小从 [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768] 中均匀抽取
* 有助于提高小物体的性能

#### 14.3.1 修改后的 IoU 感知分支

* 用软标签格式替代软权重格式修改了 IoU 感知损失的计算
  * 硬标签 (hard label)
    * int 类型
    * 表示一个确定的类别
  * 软标签 (soft label)
    * float类型
    * 表示一个可能的类别——目标是某一类别的概率

### 14.4 PP-YOLOE 的改进

#### 14.4.1 无锚

* 使用无锚架构

#### 14.4.2 新的 Backbone 和 Neck

* 用结合残差连接和密集连接的 RepResBlocks 修改了 Backbone 和 Neck 的架构

#### 14.4.3 TAL

* PP-YOLOE 按照 TOOD 中的建议实现了 TAL，其中包括将动态标签分配与任务对齐损失相结合

#### 14.4.4 高效任务对齐 Head (Efficient Task-aligned Head, ET-head)

* 区别于与 YOLOX 的分类 Head 和位置 Head 解耦
* PP-YOLOE 在 TOOD 的基础上采用单个 Head 来提高速度和准确率

#### 14.4.5 VFL 和 DFL

* VFL 使用目标分数对正样本进行权重损失，对 IoU 高的样本给予更高的权重——在训练期间优先考虑高质量样本

* DFL 将焦点损失从离散标签扩展到连续标签，从而能够成功优化结合质量估计和类别预测的改进表示—— 可以准确描述真实数据中的灵活分布，消除不一致的风险

* 两者都使用 IoU 感知分类分数 (IoU-Aware Classification Score, IACS) 作为目标，允许分类和定位质量的联合学习，从而实现训练和推理之间的一致性

### 14.5 性能评估

#### 14.5.1 PP-YOLO

* 在 MS COCO 数据集 test-dev 2017 上进行评估，PP-YOLO 在 NVIDIA V100 上以 73 FPS 的速度实现了 45.9% 的 AP 和 65.2% 的 AP~50~

#### 14.5.2 PP-YOLOv2

* 在 MS COCO 数据集 test-dev 2017 上进行评估，PP-YOLO 在 NVIDIA V100 上以 69 FPS 的速度实现了 49.5% 的 AP

#### 14.5.3 PP-YOLOE

* 在 MS COCO 数据集 test-dev 2017 上进行评估，PP-YOLO 在 NVIDIA V100 上以 78.1 FPS 的速度实现了 51.4% 的 AP



## 十五、YOLO-NAS

### 15.1 基本信息

#### 15.1.1 作者

* Eugene Khvedchenya
* Harpreet Sahota
* 均为 Deci 的研究团队成员

#### 15.1.2 提出时间

* 2023

#### 15.1.3 开源资料

* R. team, “YOLO-NAS by Deci Achieves State-of-the-Art Performance on Object Detection Using Neural Architecture
  Search.” https://deci.ai/blog/yolo-nas-object-detection-foundation-model/, 2023. Accessed: May 12, 2023.

### 15.2 主要创新点

#### 15.2.1 QSP + QCI

* 结合 8 位量化和重新参数化的优势
* 以最大限度地减少训练后量化期间的精度损失

#### 15.2.2 ==AutoNAC==

* 特点
  * 具有多功能性，可以适应任何任务、数据细节、推理环境以及性能目标的设置
  * 考虑了推理过程中涉及的数据和硬件以及其他元素，例如编译器和量化
* 作用
  * 帮助用户确定最合适的结构，为特定用途提供精度和推理速度的完美结合
* YOLO-NAS 上的应用
  * 采取 Deci 专有 NAS 技术 AutoNAC 进行自动架构设计
  * RepVGG 块在 NAS 过程中被合并到模型架构中，使其与训练后量化 (Post-Training Quantization, PTQ) 兼容
  * 通过改变 QSP 和 QCI 块的深度和位置生成了三种架构：YOLO-NASS、YOLO-NASM 和 YOLO-NASL

#### 15.2.3 ==混合量化==

* 有选择地量化模型的某些部分，以平衡延迟和准确性，而不是标准量化——后者会使所有层都受到影响

#### 15.2.4 预训练

* **自动标记数据** + 自蒸馏 + 大型数据集
  * 自动标记数据
    * 思路
      * 以一个**初步模型对小批量待标注数据进行检测**——初步模型可以是自己用少批量数据集训练出来的，也可以用网上公布的
      * 对检测出来的结果进行**人为干预纠正**
      * 把纠正后的数据**训练新的模型**
      * **用新模型对中等批量待测数据进行检测**
      * 通过**循环迭代**，可以**逐步求精**
    * 工具
      * Anno-Mage
      * easyDL 智能标注
      * X-AnyLabeling

### 15.3 特点

* 旨在检测小物体，提高定位精度，并提高每次计算的性能比，使其适合实时边缘设备应用
* **开源架构可供研究使用**

### 15.4 性能评估

* 该模型在 Objects365 上进行预训练，其中包含 200 万张图像和 365 个类别，然后使用 COCO 数据集生成伪标签
* 使用 COCO 数据集的原始 118k 训练图像来训练模型
* 已经发布了 FP32、FP16 和 INT8 精度的 3 个 YOLO-NAS 模型，在 16 位精度的 MS COCO 上实现了 52.2% 的 AP。



## 十六、YOLO with Transformer

### 16.1 CNN 的缺点

#### 16.1.1 小问题

* 语义分割需要细节信息：UNet等

* 语义分割需要上下文信息：PSPNet、Deeplab 系列、基于自注意力机制的一系列方法 (Non-Local、DANet、CCNet 等) 等——获取局部、多尺度甚至全局上下文
* 语义分割对物体边缘处的分割效果不理想：Gated-SCNN 等

#### 16.1.2 较大的问题

* **传统的 CNN 卷积核一般不会太大**，导致**模型只能利用局部信息理解输入图像**，从而**影响编码器最后提取的特征的可区分性**——RepLKNet 提出了一种浅深度大卷积核的网络模型

### 16.2 Transformer

#### 16.2.1 作者

* Ashish Vaswani (Google Brain)
* Noam Shazeer (Google Brain)
* Niki Parmar (Google Research)
* Jakob Uszkoreit (Google Research)
* Llion Jones (Google Research)
* Aidan N. Gomez (University of Toronto, Work performed while at Google Brain)
* Łukasz Kaiser (Google Brain)
* Illia Polosukhin (Work performed while at Google Research)

#### 16.2.2 论文

* Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.

#### 16.2.3 特点

* 通过引入注意力机制处理序列数据，从而摒弃 RNN 或 CNN

### 16.3 Vision Transformer (ViT)

#### 16.3.1 作者

* Alexey Dosovitskiy
* Lucas Beyer
* Alexander Kolesnikov

* 均为 Google Brain 团队成员

#### 16.3.2 论文

* Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[J]. arXiv preprint arXiv:2010.11929, 2020.

#### 16.3.3 特点

* **仿照 Transformer 模型的 BERT 架构设计，摒弃 CNN, 从而将 Transformer 移植到图像处理领域**

### 16.4 Transformer 与 YOLO 的结合

#### 16.4.1 You Only Look at One Sequence (YOLOS)

* 将 Transformer 与 YOLO 相结合用于目标检测的第一个模型
* 基于 ViT 的改进
  * 将分类中使用的一个 [CLS] 标记替换为一百个用于检测的 [DET] 标记
  * 将 ViT 中的图像分类损失替换为类似于端到端的二分匹配损失。 使用变压器进行最终物体检测
* 效果
  * 在 MS COCO 数据集上实现了 42.0% AP

#### 16.4.2 ViT-YOLO

* 受到 ViT 对遮挡、扰动和域转移的鲁棒性的启发
* 基于 ViT 的改进
  * Backbone
    * 结合了 CSP-Darknet 和多 Head 自注意力——MHSA-Darknet
  * Neck
    * 双向特征金字塔网络 (Bidirectional Feature Pyramid Networks, BiFPN) 
  * Head
    * 类似于 YOLOv3 的多尺度检测 Head

#### 16.4.3 MSFT-YOLO

* 在 Backbone 和检测 Head 中添加了基于 Transformer 的模块，旨在检测钢材表面的缺陷

#### 16.4.4 NRT-YOLO (Nested Residual Transformer-YOLO)

* 目的
  * 试图解决遥感图像中微小物体的问题
* 方法
  * 添加额外的预测 Head、特征融合层和残差变换器模块

* 效果
  * 在 DOTA 数据集中将 YOLOv5l 提高了 5.4% [128]

#### 16.4.5 DEYO

* 将 YOLO 与检测变换器 (Detection Transformer, DETR) 相结合

* 方法
  * 基于 YOLOv5 的模型，然后是类似 DETR 的模型。 
  * 第一阶段生成高质量的查询和 Anchor，输入到第二阶段

* 效果
  * 结果显示比 DETR 更快的收敛时间和更好的性能
  * 在 MS COCO 检测基准中实现了 52.1% AP



## 十七、总结

### 17.1 主要变化

#### 17.1.1  Anchor

* 发展变化
  * **最初**的 YOLOv1 模型相对简单，**没有使用 Anchor**，而当时最先进的模型依赖于带有 Anchor 的两级检测器
  * **YOLOv2 合并了 Anchor**，从而**提高了边界框预测的准确性**，这种趋势**持续了五年**
  * 直到 **YOLOX 推出了无锚定方法**并**取得了最先进的结果**
  * 此后，**后续的 YOLO** 版本都**放弃使用 Anchor**

* **现在的 Anchor free与 YOLOv1 的Anchor free 的区别**
  * YOLOv1 是直接将 bbox 和特征图上的 cell 绑定在一起， bbox 的中心点落在这个 cell 内
  * 现在的目标检测模型和语义分割一样变成了直接对 pixel 预测

#### 17.1.2 深度学习框架

* YOLOv1 到 YOLOv4 均使用 **Darknet **框架开发
* YOLOv5 开始改为 **PyTorch** 开发——使得数据增强功能激增
* PP-YOLO 采取由百度开发的 **PaddlePaddle** 开源框架

#### 17.1.3 Backbone

* 随着时间的推移，YOLO 模型的 Backbone 发生了重大变化
* 从包含简单卷积层和 max pooling 层的 Darknet 架构开始
* 在 YOLOv4 中合并了 **CSP**
* 在 YOLOv6 和 YOLOv7 中合并了**重新参数化**
* 在 DAMO-YOLO 和 YOLO-NAS 中合并了 **NAS**

#### 17.1.4 性能

* 虽然 YOLO 模型的性能随着时间的推移而不断提高，但通常**优先考虑平衡速度和准确性**，而不是仅仅关注准确性
* 这种权衡使得 YOLO 框架可以跨各种应用程序进行实时对象检测

### 17.2 速度和准确性之间的权衡

#### 17.2.1 YOLO 设计目的

* 旨在不牺牲检测结果质量的前提下提供实时性能
* YOLOv1
  * 主要关注点
    * 实现高速目标检测
  * 方法
    * 利用单个 CNN 直接从输入图像预测对象位置和类别
  * 缺陷
    * **对速度的强调导致了对准确性的妥协——主要体现在处理小物体或具有重叠边界框的物体时**

#### 17.2.2 改进

* YOLOv2
  * 引入了 **Anchor** 和**直通层**来改进对象的定位，从而获得更高的精度
* YOLOv3
  * 采用**多尺度特征提取架构**增强了模型的性能，从而可以在不同尺度上实现更好的目标检测
* YOLOv4
  * 引入**新的 Backbone、改进的数据增强技术和优化的训练策略**等创新技术
* YOLOv5
  * **提供不同的模型规模**来满足特定的应用和硬件要求



