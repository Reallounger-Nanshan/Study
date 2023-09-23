# YOLO 系列与 R-CNN 系列的对比

## 一、R-CNN (Region-based Convolutional Neural Networks) 系列

### 1.1 R-CNN

#### 1.1.1 基本信息

* 作者
  * Ross Girshick (Microsoft Research)
* 论文
  * Girshick R, Donahue J, Darrell T, et al. Region-based convolutional networks for accurate object detection and segmentation[J]. IEEE transactions on pattern analysis and machine intelligence, 2015, 38(1): 142-158.

#### 1.1.2 基本思想

* **从输入图像中提取出若干感兴趣区域** (Region of Interest, RoI)——相当于对目标先定位
* 用一个 CNN 网络去**提取每一个 RoI 的特征**
* 使用一个多分类支持向量机 (Support Vector Machines, SVM) 来**识别每个 RoI 的类别**

#### 1.1.3 主要贡献

* **首次将 CNN 引入目标检测领域**

#### 1.1.4 缺点

* R-CNN运算量非常大
  * 一幅图像有 2 千个 ROI，每个区域生成一个特征向量
  * 需要 CNN（图像分类和特征提取）、SVM（物体识别）和回归模型（调整边界），并且这三个模型数据不共享

### 1.2 Fast R-CNN

#### 1.2.1 基本信息

* 作者
  * Ross Girshick (Microsoft Research, Redmond, WA)
  * Jeff Donahue (the Department of Electrical Engineering and Computer Science, UC Berkeley, Berkeley, CA)
  * Trevor Darrell (the Department of Electrical Engineering and Computer Science, UC Berkeley, Berkeley, CA)
  * Jitendra Malik (the Department of Electrical Engineering and Computer Science, UC Berkeley, Berkeley, CA)
    * Fellow, IEEE
* 论文
  * Girshick R. Fast r-cnn[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1440-1448.

#### 1.2.2 基于 R-CNN 的改进

* ==**共享特征矩阵**==
  * 三个独立模型合并为了一个联合训练框架并共享计算结果——不再为每个特征设置独立的特征向量，而是每个图像采用一个 CNN 正向通道，共享特征矩阵
  * 并采用相同的特征矩阵来构建分类器和边界回归矩阵
* ==**RoI Pooling**==
  * 方法
    * 将输入的 h \* w 大小的 feature map 分割成 H \* W 大小的子窗口
      * 其中H、W为超参数，如设定为7 x 7
    * 对每个子窗口进行 max pooling 操作，得到固定输出大小的 feature map
  * 作用
    * 引入平移不变性、旋转不变性和尺度不变性
    * 完成 feature map 的聚合，实现**数据降维，防止过拟合**
    * **将不同输入尺寸的 feature map 通过分块池化的方法得到固定尺寸的输出**
  * 缺点
    * 量化会导致 RoI 和提取的特征之间出现偏差——虽然不太影响分类，但会严重影响对对象的像素级精确的掩膜的生成（实例分割）

* 损失函数
  * L1 范数：向量中各元素的绝对值之和

### 1.3 Faster R-CNN

#### 1.3.1 基本信息

* 作者
  * Shaoqing Ren
  * Kaiming He
  * Ross Girshick
  * Jian Sun
  * 均为 Microsoft Research 成员
* 论文
  * Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[J]. Advances in neural information processing systems, 2015, 28.

#### 1.3.2 基于 Fast R-CNN 的改进

* 构建由RPN (Region Proposal Network) 和具有共享卷积特征层的 Fast R-CNN 组成的统一模型
  * ==**Two-Stage 工作范式确定**==
    * 使用 RPN 处理输入图像以获得若干个 RoI——**定位**
    * 使用 RoI pooling 和若干全连接层完成对每一个 RoI 的类别识别——**分类**

### 1.4 Mask R-CNN

#### 1.4.1 基本信息

* 作者
  * Kaiming He
  * Georgia Gkioxari
  * Piotr Dollar
  * Ross Girshick
  * 均为 Facebook AI Research (FAIR) 成员
* 论文
  * He K, Gkioxari G, Dollár P, et al. Mask r-cnn[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2961-2969.

#### 1.4.2 基于 Faster R-CNN 的改进

* ==**RoIAlign**==
  * 作用
    * 去除了 RoI Pooling 的严格量化，**将提取的特征与输入正确对齐**
  * 方法
    * 使用双线性插值来计算每个 RoI 中四个定期采样位置处的输入特征的精确值，并用最大值或平均值聚合结果



## 二、YOLO系列

### 2.1 One-Stage

#### 2.1.1 与 Two-Stage 的联系

* Two-Stage 在完成定位任务时，实际上区分了前景与背景
* 将 RPM 区分前后景的二分类任务扩展为多分类任务，实现定位和分类任务的整合——从而得到 One-Stage的基本思想

#### 2.2.1 与 Two-Stage 的区别

* One-Stage 通过简化网络结构，实现更快的计算速度，从而可以应用于实时目标检测任务

### 2.2 YOLO系列网络

* [YOLOv1至YOLOv8的总结](./YOLOv1至YOLOv8的总结.md)