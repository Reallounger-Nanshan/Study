# Matlab Sensor Fusion and Tracking Toolbox

## 一、定义场景

### 1.1 简易场景

* ```matlab
  % Create driving scenario
  s = drivingScenario("SampleTime", 0.05)'
  
  % Add road
  roadCenters = [0 0; 10 0; 20 30; 40 20; 50 20]; % (x, y) m
  roadWidth = 5; % m
  road(s, roadCenters, roadWidth);
  plot(s);
  
  % Add vehicle
  egoCar = vehicle(s);
  waypoints = roadCenters;
  speed = 13.89; % 13.89 m/s = 50 km/hr
  trajectory(egoCar, waypoints, speed);
  
  % Play scenario
  while advance(s)
      pause(s.SampleTime);
  end

* ![SimpleScenario](./SimpleScenario.jpg)

### 1.2 Driving Scenario Designer

### 1.3 导入 Matlab 数据

### 1.4 使用自主系统生成更通用的场景（两种方案）

* 使用指定的加速度和角速度以运动学方式定义轨迹
  * 轨迹：沿着一条路径并由同一目标生成的有序集合
* 使用指定的航路点以及每个航路点的到达时间、速度和方向



## 二、跟踪器接口

### 2.1 objectDetection

#### 2.1.1 作用

* 为了与跟踪器连接，所提供的一个用于检测的通用 API

#### 2.1.2 组成

##### a. 输入

* Time
* Measurement

##### b. 输出

* detection ——Detection report

  * Time

    * 数据类型
      * double
    * Description
      * 检测时间
      * 非负实数标量
      * 不能将此属性设置为名称值对，需改用时间输入参数
    * Importance
      * 需要测量轨迹以精确测量时间并计算距离

  * SensorIndex

    * 数据类型
      * int
      * \> 0
    * Description
      * 传感器标识符
    * Importance
      * 用于区分不同的传感器，并且每个 ID 必须是传感器所独有的
      * 每个传感器分配到一个轨迹

  * Measurement

    * 数据类型

      * double / single vector

    * Description

      * 传感器测量值

      * 不能将此属性设置为名称值对，需改用度量值输入参数

    * Importance

      * 用于评估和校正滤波器

  * MeasurementNoise

    * 数据类型
      * double / single
      * double / single & 实正半定 N \* N matrix
    * Description
      * 测量噪声协方差
      * N 是测量值中元素的数量
      * 对于标量情况，矩阵是具有与测量相同数据解释的正方形对角 N \* N 矩阵
    * Importance
      * 用于评估和校正滤波器

  * MeasurementParameters (optional)

    * 数据类型

      * struct

    * Description

      * 从滤波器状态转换为测量的测量函数参数

      * 在跟踪滤波器中使用自定义测量函数时，可以以任何格式定义测量，只要它与自定义测量函数的定义一致

      * 在跟踪滤波器中使用内置的测量功能（如 cvmeas 和 ctmeas）时，可以使用以下字段的结构来定义矩形或球形坐标系中的测量

        * 内置测量功能
          * cvmeas: 等速 (constant velocity) 运动的测量函数
          * ctmeas: 恒定转弯率 (constant turn-rate) 运动的测量函数

        * 字段
          * Frame
            * 值
              * Rectangular: 检测以直角坐标报告
              * Spherical: 检测以球形坐标报告
            * 注意
              * 在 Simulink 中，创建对象检测总线时，请将 Frame 指定为 fusionCoordinateFrameType.Regular 或 fusionCooordinateFrameType.Spheric 的枚举对象，因为 Simulink 不支持字符向量等可变大小的变量
          * OriginPosition
            * 帧原点相对于父帧的位置偏移
            * [x y z] 实值向量
          * OriginVelocity
            * 帧原点相对于父帧的速度偏移
            * [v~x~ v~y~ v~z~] 实值向量
          * Orientation
            * 帧旋转矩阵
            * 3 \* 3 实值正交矩阵
          * HasAzimuth
            * 指示方位是否包含在测量中的逻辑标量
            * 当 Frame 为 Rectangular 时，此字段不相关
          * HasElevation
            * 指示高程信息是否包含在测量中的逻辑标量
            * 当 Frame 为 Rectangular 时，如果 HasElevation 为 False，则报告的测量假设为 0° 仰角
          * HasRange
            * 指示测量中是否包含范围的逻辑标量
            * 当 Frame 为 Rectangular 时，此字段不相关
          * HasVelocity
            * 指示所报告的检测是否包括速度测量的逻辑标量
            * 当 Frame 为 Rectangular 时，如果 HasVelocity 为 False，则测量报告为 [x y z]；如果HasVelocity为 True，则测量值报告为 [x y z v~x~ v~y~ v~z~]
            * 当 Frame 为 Spherical 时，对于在球形框架中报告的测量，如果 HasVelocity 为 True，则测量包含测距率信息
          * IsParentToChild
            * 指示 Orientation 是否执行从父坐标系到子坐标系的帧旋转的逻辑标量
            * 当 IsParentToChild 为 False 时，Orientation 将执行从子坐标系到父坐标系的帧旋转

    * Importance

      * 用于评估和校正滤波器

  * ObjectClassID (optional)

    * 数据类型
      * int
      * ≥ 0
    * Description
      * 对象类标识符
      * 将此属性指定为非零整数时，可以使用 ObjectClassParameters 属性指定检测分类器统计信息
    * Importance
      * 使用此属性可以区分从不同类型的对象生成的检测
        * 非零整数 i 表示第 i  类物体
        * 0 表示未知的对象类型

  * ObjectParameters (optional)

    * 数据类型
      * struct
    * Description
      * 检测分类器的参数
      * 对于使用多对象跟踪器的类融合，例如 trackerGNN (Sensor Fusion and Tracking Toolbox) 系统对象，可以指定 ConfusionMatrix 字段
        * ConfusionMatrix
          * 检测分类器的融合矩阵
          * N \* N 的实值矩阵，其中 N 是可能的对象类的数量
          * 矩阵的 (i, j) 元素表示如果检测的真实类别是类别 i，则分类器将检测分类为类别 j 的权重或概率

  * ObjectAttributes (optional)

    * 数据类型
      * 单元数组
      * 结构数组
    * Description
      * 通过跟踪器传递的对象属性
      * 这些属性被添加到雷达跟踪器的输出中，但不被跟踪器使用



## 三、跟踪算法

### 3.1 跟踪器

#### 3.1.1 特点

* 开箱即用
* 所有跟踪器组件都是高度可配置的
* 跨库的编程接口

### 3.2 滤波器

#### 3.2.1 作用

* 动态应用时需要估计滤波器来平滑噪声检测信息并将其转换为适合应用的状态表示
* 优化系统性能和计算要求

#### 3.2.2 选择

* 涉及许多不同的方面
  * 环境中物体的基本运动及其可操作性
  * 单精度或双精度计算的数值稳定性
  * 自主平台的计算复杂性与处理能力 

#### 3.2.3 可选择的组件

* 线性卡尔曼滤波器 (Linear Kalman Filter, KF)
* 扩展卡尔曼滤波器 (Extended Kalman Filter, EKF)
* 无迹卡尔曼滤波器 (Unscented Kalman Filter, UKF)
* 容积卡尔曼滤波器 (Cubature Kalman Filter, CKF)
* α-β-γ 滤波器 (Alpha-beta-gamma filter)
* 粒子滤波器 (Particle filter)
* 范围和角度参数化扩展卡尔曼滤波器 (Range- and angle-parameterized EKF)
* 交互式多模型滤波器 (Interacting Multiple Model filter, IMM filter)

#### 3.2.4 特点

* 为了方便交换过滤器并进一步扩展列表，**所有过滤器都使用一个通用接口**
* 通过此接口，可以给跟踪滤波器和运动模型提供函数句柄，以便与工具箱库中的跟踪器集成

### 3.3 其它

* 除了滤波器之外，还提供基于历史轨迹和轨迹分数的确认和删除逻辑
* 还可以作为构建块提供一整套数据关联算法，用于解决 2D 和 S-D 关联以获得 1-best 或 k-best 的解决方案
* 多对象跟踪器的输出由轨迹列表组成，这些轨迹是对真实对象的估计，分为已确认的轨迹和暂定的轨迹



## 四、跟踪自动驾驶车辆周围的物体

### 4.1 多目标跟踪器

#### 4.1.1 单假设跟踪器

##### a. multiObjectTracker

* 作用
  * 创建一个多传感器、多对象跟踪器

* 传感器数据关联方法

  * 数据关联

    * 应用背景
      * 多目标跟踪问题的难点在于密集环境中的目标跟踪，在这种情况下**难以区分相近目标的轨迹**
      * 需要对多传感器的数据进行融合来提高跟踪效果
    * 概念
      * 分配和计算与观察值或轨迹相关的权重实现从**一组轨迹的观测值映射到另一组数据**的过程

    * 基本步骤
      * 建立 Gating ，确定 Gating 阈值
        * Gating——跟踪门 / 关联门
          * 作用
            * 数据关联时，通常采用 Gating 相关的方法实现目标数据的关联
          * 方法
            * 以前一采样周期预测点为中心，设置一个关联门
          * 筛选准则
            * 尽可能的让预测值-观测值对相关度超过阈值
      *  Gating 阈值过滤
      * 确定相似性度量方法
      * 建立关联矩阵
      * 确定关联判定准则
      * 形成关联对

  * 最近邻关联 (Nearest Neighbour, NN)

    * 方法

      * **最小距离规则**

        * 假设在第 K 次扫描之前，已经建立了 N 条轨迹

        * 第 K 次新观测为 Z~j~ (k), j = 1, 2, ..., N

        * 在第 i 条轨迹的 Gating 内，观测 j 和轨迹 i 的差矢量定义为测量值和预测值之间的差，即滤波器残差

          * $$
            e_{ij}\left(k\right)=Z_j\left(k\right)-H\hat{X}_i\left(k|k-1\right)
            $$

            * H 为观测矩阵

            * 状态估计的下一步预测方程

              * $$
                \hat{X}_i\left(k|k-1\right)=AX_i\left(k-1|k-1\right)
                $$

              * A 为状态转移矩阵

        * 设 S (k) 为 e~ij~ (k) 的协方差矩阵，则统计距离（平方）为

          * $$
            d_{ij}=e_{ij}\left(k\right)S\left(k\right)^{-1}e_{ij}\left(k\right)^T
            $$

        * 该距离在**阀值范围内**且**距离最近**则认为为成功关联

      * 关联思路

        * **从第 0 行开始**按照最小距离优先规则关联
        * **关联成功后，划掉此行和列**
        * 依次按行进行

    * 特点
      * 一个目标最多只与 Gating 中一个测量相关
      * 以总关联代价（或总距离）作为关联评价标准，取总关联代价或总距离最小的关联对为正确关联对
    * 优点
      * 运算量小
      * 易于硬件的实现
    * 缺点
      * 只能适用于稀疏目标和杂波环境的目标跟踪系统
      * 当在**目标或者杂波密度较大时**，很容易出现**误跟和漏跟**现象，从而导致算法跟踪性能不高

  * **全局最近邻关联 (Global Nearest Neighbour, GNN)**——函数采取的方法

    * 与 NN 的联系与区别

      * 联系
        * 距离计算方法相同
        * 已经匹配过的点不再参与下次匹配
      * 区别
        * 使**总的距离或关联代价达到最小**，实现最优分配

    * 优化目标

      * $$
        min\{\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{n}c_{ij}x_{ij}\}\\
        s.t.\quad\sum\limits_{i=1}^{n}x_{ij}=1,\ \sum\limits_{j=1}^{n}x_{ij}=1
        $$

      * x~ij~ 的值仅为  0 或 1

* 参数
  * TrackerIndex 
    * 跟踪器的唯一标识符
  * FilterInitializationFcn
    * 一个函数句柄，用于根据检测初始化跟踪过滤器
      * 函数句柄
        * 概念
          * 类似于函数的指针，函数的唯一标识
        * 使用方式
          * khandle = @kan
            * 创建了 kan 的句柄，输入 khandle(x) 就是调用了 kan(x) 的功能
        * 优点
          * 避免调用函数时反复查找所有地址，从而**提高运行速度**
          * 可以和变量一样**方便使用**
    * 必须是**函数句柄**或**字符矢量**
  * AssignmentThreshold
    * 控制将检测分配给轨迹的阈值
    * 必须是正**有限实数标量**，或者是 **2 元素数组 [a b]**，其中 a 是有限的，b 可以是无限的
  * MaxNumTracks
    * 定义最大轨迹数
    * 必须是一个**实正整数**
  * MaxNumDetections
    * 定义最大检测数
  * MaxNumSensors
    * 定义最大传感器数
  * OOSMHandling
    * 处理无序测量 (Out-Of-Sequence Measurement, OOSM)
  * ConfirmationThreshold
    * 如果在自轨迹初始化以来的前 N 个更新中至少有 M 个检测被分配给轨道，则轨道被确认，其中 M＜＝N；反之如果在 N 个更新中将少于 M 个检测分配给该轨道，则该轨迹被删除
    * 必须是一个**双元素向量 [M N]**，且 M 和 N 都必须是实正整数，同时 M ≤ N
  * DeletionThreshold
    * 如果在之前 Q 次跟踪器更新中，确认的轨迹没有被分配任何检测 P 次，则删除该轨迹
    * 必须是一个**双元素向量 [P Q]**，且 P 和 Q 都必须是实正整数，同时 Q ≤ Q
  * HasCostMatrixInput
    * 提供代价矩阵作为输入
    * 必须是一个**逻辑标量**
  * HasDetectableTrackIDsInput
    * 提供可检测的轨迹 ID 作为输入
  * StateParameters
    * 定义轨迹状态的参数
  * NumTracks
    * 轨迹总数（只读）
  * NumConfirmedTracks
    * 已确认的轨迹数量（只读）
  
* Methods
  * updateTracks
    * 作用
      * 创建、更新和删除轨迹
    * 输入输出
      * 输入
        * tracker
          * 多目标跟踪器
        * detections
          * 包含检测对象的元胞数组
        * time
          * 所有轨迹将被更新和预测的时间
          * 实数标量，且必须大于上一次调用中的值
        * costMatrix
          * 将检测分配给轨迹的代价
          * T \* D 矩阵
            * T 是上一次调用中所有轨迹的数量，D是当前调用中的检测数量
          * 代价越高，分配的可能性越低
        * detectableTrackIDs
          * 传感器期望检测的轨道的ID
          * M \* 1 或 M \* 2矩阵
            * 第一列是轨迹 ID，由轨迹输出的 TrackID 字段报告
            * 第二列是可选的，允许添加每条轨迹的检测概率
      * 输出
        * 一个已确认的、对时间实例进行校正和预测的轨迹数组
    * 基本步骤
      * 尝试将检测**分配给现有轨迹**
      * 基于**未分配的检测创建新的轨迹**
      * 更新并**确认分配给检测的轨迹**
      * 未分配给检测的轨迹被将被**标记**，一直未分配给检测的轨迹将被**删除**
  * predictTracksToTime
    * 将轨迹预测为时间戳
  * getTrackFilterProperties
    * 返回滤波器属性的值
  * setTrackFilterProperties
    * 设置滤波器特性的值
  * initializeTrack
    * 初始化一条新的轨迹
  * confirmTrack
    * 确认一条轨迹
  * deleteTrack
    * 删除一条已存在的轨迹
  * release
    * 允许更改属性值和输入特性
  * clone
    * 创建 multiObjectTracker 的副本
  * isLocked
    * 锁定状态（逻辑）
  * reset
    * 重置 multiObjectTracker 的状态
  * createBus
    * 创建 Simulink 输出总线（仅限Simulink）



tracker=multiObjectTracker创建了一个多传感器、多对象跟踪器，该跟踪器使用全局最近邻居（GNN）分配来维护关于其跟踪的对象的单一假设。multiObjectTracker初始化、确认、校正、预测（执行滑行）和删除轨迹。创建的轨迹状态为“暂定”，这意味着multiObjectTracker没有足够的证据来确定该轨迹是物理对象的。如果为暂定轨道分配了足够多的附加检测，其状态将更改为“已确认”（请参阅ConfirmationThreshold）。或者，如果将ObjectClassID值为非零值的检测分配给轨迹，则将确认轨迹，因为这意味着传感器能够对物理对象进行分类



系统对象可以像函数一样直接调用，而不是使用step方法。例如，y=step（obj）和y=obj（）是等价的。





#### 4.1.2 多假设跟踪器

#### 4.1.3 联合概率被检测关联 (Joint Probabilistic Detection Association, JPDA) 跟踪器

#### 4.1.4 基于随机有限集 (Random Finite Sets, RFS) 的方法

#### 4.1.5 不同类型的概率假设密度 (Probability Hypothesis Density, PHD) 跟踪器

### 4.2 传统跟踪器

#### 4.2.1 跟踪器

* trackerGNN
* trackerJPDA
* trackerTOMHT

#### 4.2.2 主要思想

* 假设每次扫描对每个对象进行一次检测
* **在传感器针对每个对象返回多个检测的情况下，必须先将这些检测结果聚类**
* 可以提供单点和更大的协方差，或者可以通过 bounding box 获得检测到的对象的形状
* 跟踪器将对象建模为点目标并跟踪其运动状态
* 对于激光雷达等高分辨率传感器，对扩展物体的多次检测可以转换为单个参数化形状检测
  * 形状检测
    * 形状检测包括物体的运动状态及其长度、宽度和高度等范围参数
    * 形状检测也可以通过传统跟踪器进行处理，该跟踪器将对象建模为 bounding box 并跟踪对象的运动状态及其尺寸
  * ego vehicle——被自主控制的车辆
    * ego vehicle 周围车辆的激光雷达检测被转换为具有定义的长度、宽度和高度的边界框形式的长方体检测
    *  通过使用每个维度中点的坐标的最小值和最大值，将边界框拟合到每个簇上
    * 检测器将点云分割成对象后将边界框包裹在点云周围



## 五、跟踪扩展对象

### 5.1 聚类产生的问题

#### 5.1.1 丢失信息

* 可能丢失有关轨迹下物体的大小和方向等信息
  * 导致**转动中心定位不一致**——该转动中心定位会**根据传感器相对于被跟踪物体的方位角而变化**

#### 5.1.2 轨迹错误

* 可能会导致错误的轨迹或丢失轨迹
  * 导致感知系统混乱
  * 可能导致下游算法出现问题——例如执行紧急制动

### 5.2 转向扩展对象跟踪器 (extended object tracker)

#### 5.2.1 主要思想

* 扩展对象每次扫描可以对每个对象进行多次检测——例如 2D 雷达
* 扩展对象以椭圆或矩形的形式映射到二维平面
*  通过使用多次检测，跟踪器估计每个物体的位置、速度、尺寸和方向

#### 5.2.1 Point object 和 Extended object 区别

* ![Point object and Extended object](./Point object and Extended object.png)

* Point object
  * 通过单个点表示距离
  * **每次扫描对每个对象获得一个检测**
* Extended object
  * 高分辨率传感器**每次扫描对每个物体产生多次检测**

#### 5.2.2 分类

* 使用椭圆作为扩展对象形状
  * GGIW-PHD Tracker
    * GGIW-PHD: Gamma Gaussian Inverse-Wishart Probability Hypothesis Density（伽玛高斯逆威沙特概率假设密度）
    * 假设测量分布在物体范围周围——轨迹中心位于车辆可观察部分上
* 使用矩形作为扩展对象形状
  * Rectangular GGIW-PHD Tracker

#### 5.2.3 方法对比

##### a. 数据关联对比

* ![data association](./data association.png)
* Point Target Tracker 的数据关联存在问题

##### b. 位置精度对比

* ![positional accuracy](./positional accuracy.png)
  * Point Target Tracker 不估计对象的偏航和尺寸
* Point Target Tracker
  * 能够**以合理的精度估计对象的运动学**
  * **ego vehicle 后面的车辆位置误差较高**
    * 当过往车辆超越 ego vehicle 时，ego vehicle 被拖向左侧——当对象彼此靠近时，聚类产生的问题
* GGIW-PHD Tracker
  * 对于 ego vehicle 前后方的车辆，跟踪器能够以约 0.3 米的精度估计物体的尺寸
  * 当经过的车辆**相对于 ego vehicle 进行机动**时，**偏航估计的误差会更高**

* Rectangular GGIW-PHD Tracker
  * 可以**更准确地估计形状和方向**
  * **计算成本更高**

##### c. 运行时间对比

* ![run times](./run times.png)



## 六、仿真示例

### 6.1 基于雷达和视觉传感器的数据融合的车辆跟踪

#### 6.1.1 主程序

##### a. 生成场景

* ```matlab
  % 定义一个空场景
  scenario = drivingScenario; % 创建驾驶场景
  scenario.SampleTime = 0.01;
  
  % 将道路添加到驾驶场景
  roadCenters = [0 0; 50 0; 100 0; 250 20; 500 40];
  mainRoad = road(scenario, roadCenters, 'lanes', lanespec(2));
  barrier(scenario, mainRoad);
  
  % 创建以25m/s的速度沿道路行驶的 ego vehicle
  egoCar = vehicle(scenario, 'ClassID', 1);
  trajectory(egoCar, roadCenters(2:end, :) - [0 1.8], 25); % 在右侧车道上
  
  % 在 ego vehicle 前面添加一辆车
  leadCar = vehicle(scenario, 'ClassID', 1);
  trajectory(leadCar, [70 0; roadCenters(3:end, :)] - [0 1.8], 25); % 在左侧车道上
  
  % 添加一辆以35m/s的速度沿道路行驶并经过 ego vehicle 的汽车
  passingCar = vehicle(scenario, 'ClassID', 1);
  waypoints = [0 -1.8; 50 1.8; 100 1.8; 250 21.8; 400 32.2; 500 38.2];
  trajectory(passingCar, waypoints, 35);
  
  % 在 ego vehicle 后面添加一辆车
  chaseCar = vehicle(scenario, 'ClassID', 1);
  trajectory(chaseCar, [25 0; roadCenters(2:end, :)] - [0 1.8], 25); % 在左侧车道上

##### b. 定义雷达和相机

* ```matlab
  sensors = cell(8, 1);
  
  % 汽车前保险杠中央的前置远程雷达传感器
  sensors{1} = drivingRadarDataGenerator('SensorIndex', 1, 'RangeLimits', [0 174], ...
      'MountingLocation', [egoCar.Wheelbase + egoCar.FrontOverhang, 0, 0.2], 'FieldOfView', [20, 5]);
  
  % 汽车后保险杠中心的后向远程雷达传感器
  sensors{2} = drivingRadarDataGenerator('SensorIndex', 2, 'MountingAngles', [180 0 0], ...
      'MountingLocation', [-egoCar.RearOverhang, 0, 0.2], 'RangeLimits', [0 30], 'FieldOfView', [20, 5]);
  
  % 汽车左后轮罩处的左后短程雷达传感器
  sensors{3} = drivingRadarDataGenerator('SensorIndex', 3, 'MountingAngles', [120 0 0], ...
      'MountingLocation', [0, egoCar.Width/2, 0.2], 'RangeLimits', [0 30], 'ReferenceRange', 50, ...
      'FieldOfView', [90, 5], 'AzimuthResolution', 10, 'RangeResolution', 1.25);
  
  % 位于汽车右后轮罩的右后短程雷达传感器
  sensors{4} = drivingRadarDataGenerator('SensorIndex', 4, 'MountingAngles', [-120 0 0], ...
      'MountingLocation', [0, -egoCar.Width/2, 0.2], 'RangeLimits', [0 30], 'ReferenceRange', 50, ...
      'FieldOfView', [90, 5], 'AzimuthResolution', 10, 'RangeResolution', 1.25);
  
  % 汽车左前轮罩左前短程雷达传感器
  sensors{5} = drivingRadarDataGenerator('SensorIndex', 5, 'MountingAngles', [60 0 0], ...
      'MountingLocation', [egoCar.Wheelbase, egoCar.Width/2, 0.2], 'RangeLimits', [0 30], ...
      'ReferenceRange', 50, 'FieldOfView', [90, 5], 'AzimuthResolution', 10, ...
      'RangeResolution', 1.25);
  
  % 汽车右前轮罩右前短程雷达传感器
  sensors{6} = drivingRadarDataGenerator('SensorIndex', 6, 'MountingAngles', [-60 0 0], ...
      'MountingLocation', [egoCar.Wheelbase, -egoCar.Width / 2, 0.2], 'RangeLimits', [0 30], ...
      'ReferenceRange', 50, 'FieldOfView', [90, 5], 'AzimuthResolution', 10, ...
      'RangeResolution', 1.25);
  
  % 前挡风玻璃上的前置摄像头
  sensors{7} = visionDetectionGenerator('SensorIndex', 7, 'FalsePositivesPerImage', 0.1, ...
      'SensorLocation', [0.75 * egoCar.Wheelbase 0], 'Height', 1.1);
  
  % 后挡风玻璃上的后置摄像头
  sensors{8} = visionDetectionGenerator('SensorIndex', 8, 'FalsePositivesPerImage', 0.1, ...
      'SensorLocation', [0.2 * egoCar.Wheelbase 0], 'Height', 1.1, 'Yaw', 180);
  
  % 为传感器注册 actor 配置文件
  profiles = actorProfiles(scenario);
  for m = 1:numel(sensors)
      if isa(sensors{m}, 'drivingRadarDataGenerator')
          sensors{m}.Profiles = profiles;
      else
          sensors{m}.ActorProfiles = profiles;
      end
  end

##### c. 创建跟踪器

* ```matlab
  tracker = multiObjectTracker('FilterInitializationFcn', @initSimDemoFilter, ...
      'AssignmentThreshold', 30, 'ConfirmationThreshold', [4 5]);
  positionSelector = [1 0 0 0; 0 0 1 0]; % 位置选择器
  velocitySelector = [0 1 0 0; 0 0 0 1]; % 速度选择器
  
  BEP = createDemoDisplay(egoCar, sensors); % 创建显示并返回鸟瞰图

##### d. 仿真场景

* ```matlab
  toSnap = true;
  while advance(scenario) && ishghandle(BEP.Parent)
      time = scenario.SimulationTime; % 获取场景时间
  
      ta = targetPoses(egoCar); % 获取其他车辆在 ego vehicle 坐标中的位置
  
      % 模拟传感器
      detectionClusters = {};
      isValidTime = false(1, 8);
      for i = 1:8
          [sensorDets, numValidDets, isValidTime(i)] = sensors{i}(ta, time);
          if numValidDets
              for j = 1:numValidDets
  
                  % 视觉检测不报告 SNR
                  % 跟踪器要求它们具有与雷达探测相同的对象属性
                  % 这将 SNR 对象属性添加到视觉检测中，并将其设置为 NaN
                  if ~isfield(sensorDets{j}.ObjectAttributes{1}, 'SNR')
                      sensorDets{j}.ObjectAttributes{1}.SNR = NaN;
                  end
  
                  % 从测量和测量噪声场中去除测量位置和速度的 Z 分量
                  sensorDets{j}.Measurement = sensorDets{j}.Measurement([1 2 4 5]);
                  sensorDets{j}.MeasurementNoise = sensorDets{j}.MeasurementNoise([1 2 4 5], [1 2 4 5]);
              end
              detectionClusters = [detectionClusters; sensorDets];
          end
      end
  
      % 如果有新的检测，更新跟踪器
      if any(isValidTime)
          if isa(sensors{1}, 'drivingRadarDataGenerator')
              vehicleLength = sensors{1}.Profiles.Length;
          else
              vehicleLength = sensors{1}.ActorProfiles.Length;
          end
          confirmedTracks = updateTracks(tracker, detectionClusters, time);
  
          % 更新鸟瞰图
          updateBEP(BEP, egoCar, detectionClusters, confirmedTracks, positionSelector, velocitySelector);
      end
  
      % 当汽车经过 ego vehicle 时，截图
      if ta(1).Position(1) > 0 && toSnap
          toSnap = false;
          snapnow
      end
  end

#### 6.1.2 函数

##### a. initSimDemoFilter.m

* ```matlab
  function filter = initSimDemoFilter(detection)
  % 使用二维等速模型初始化跟踪 KF 滤波器。
  % 状态向量为 [x；vx；y；vy]
  % 检测测量矢量为 [x；y；vx；vy]
  % 因此，测量模型为 H = [1 0 0 0；0 0 1 0；0 1 0 0；00 0 1]
  H = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1];
  filter = trackingKF('MotionModel', '2D Constant Velocity', ...
      'State', H' * detection.Measurement, ...
      'MeasurementModel', H, ...
      'StateCovariance', H' * detection.MeasurementNoise * H, ...
      'MeasurementNoise', detection.MeasurementNoise);
  end

##### b. createDemoDisplay.m

* ```matlab
  function BEP = createDemoDisplay(egoCar, sensors)
      % 创建一张图像
      hFigure = figure('Position', [0, 0, 1200, 640], 'Name', 'Sensor Fusion with Synthetic Data Example');
      movegui(hFigure, [0 -1]); % 将图像向左移动，并从顶部向下移动一点
  
      % 添加一个从后面跟随 ego vehicle 的汽车
      hCarViewPanel = uipanel(hFigure, 'Position', [0 0 0.5 0.5], 'Title', 'Chase Camera View');
      hCarPlot = axes(hCarViewPanel);
      chasePlot(egoCar, 'Parent', hCarPlot);
  
      % 添加一个从俯视图跟随 ego vehicle 的汽车
      hTopViewPanel = uipanel(hFigure, 'Position', [0 0.5 0.5 0.5], 'Title', 'Top View');
      hCarPlot = axes(hTopViewPanel);
      chasePlot(egoCar, 'Parent', hCarPlot, 'ViewHeight', 130, 'ViewLocation', [0 0], 'ViewPitch', 90);
  
      % 为鸟瞰图添加面板
      hBEVPanel = uipanel(hFigure, 'Position', [0.5 0 0.5 1], 'Title', 'Bird''s-Eye Plot');
  
      % 创建 ego vehicle 和传感器覆盖范围的鸟瞰图
      hBEVPlot = axes(hBEVPanel);
      frontBackLim = 60;
      BEP = birdsEyePlot('Parent', hBEVPlot, 'Xlimits', [-frontBackLim frontBackLim], 'Ylimits', [-35 35]);
  
      % 绘制雷达的覆盖区域
      for i = 1:6
          cap = coverageAreaPlotter(BEP, 'FaceColor', 'red', 'EdgeColor', 'red');
          if isa(sensors{i}, 'drivingRadarDataGenerator')
              plotCoverageArea(cap, sensors{i}.MountingLocation(1:2),...
                  sensors{i}.RangeLimits(2), sensors{i}.MountingAngles(1), sensors{i}.FieldOfView(1));
          else
              plotCoverageArea(cap, sensors{i}.SensorLocation, ...
                  sensors{i}.MaxRange, sensors{i}.Yaw, sensors{i}.FieldOfView(1));
          end
      end
  
      % 绘制视觉传感器的覆盖区域
      for i = 7:8
          cap = coverageAreaPlotter(BEP, 'FaceColor', 'blue', 'EdgeColor', 'blue');
          if isa(sensors{i},'drivingRadarDataGenerator')
              plotCoverageArea(cap, sensors{i}.MountingLocation(1:2), ...
                  sensors{i}.RangeLimits(2), sensors{i}.MountingAngles(1), 45);
          else
              plotCoverageArea(cap, sensors{i}.SensorLocation,...
                  sensors{i}.MaxRange, sensors{i}.Yaw, 45);
          end
      end
  
      % 创建视觉检测绘图仪，将其放入结构中以备将来使用
      detectionPlotter(BEP, 'DisplayName', 'vision', 'MarkerEdgeColor', 'blue', 'Marker', '^');
  
      % 将所有雷达探测合并为一个条目，并将其存储以备后续更新
      detectionPlotter(BEP, 'DisplayName', 'radar', 'MarkerEdgeColor', 'red');
  
      % 将道路边界添加到图像中
      laneMarkingPlotter(BEP, 'DisplayName', 'lane markings');
  
      % 将轨迹添加到鸟瞰图中，并显示最近10个 track 更新
      trackPlotter(BEP, 'DisplayName', 'track', 'HistoryDepth', 10);
  
      axis(BEP.Parent, 'equal');
      xlim(BEP.Parent, [-frontBackLim frontBackLim]);
      ylim(BEP.Parent, [-40 40]);
  
      % 为 GT 添加轮廓绘图仪
      outlinePlotter(BEP, 'Tag', 'Ground truth');
  end

##### c. updateBEP.m

* ```matlab
  function updateBEP(BEP, egoCar, detections, confirmedTracks, psel, vsel)
      % 更新道路边界及其显示
      [lmv, lmf] = laneMarkingVertices(egoCar);
      plotLaneMarking(findPlotter(BEP, 'DisplayName','lane markings'), lmv, lmf);
  
      % 更新地面实况数据
      [position, yaw, length, width, originOffset, color] = targetOutlines(egoCar);
      plotOutline(findPlotter(BEP,'Tag','Ground truth'), position, yaw, length, width, 'OriginOffset', originOffset, 'Color', color);
  
      % 更新障碍数据
      [bPosition, bYaw, bLength, bWidth, bOriginOffset, bColor, numBarrierSegments] = targetOutlines(egoCar, 'Barriers');
      plotBarrierOutline(findPlotter(BEP, 'Tag', 'Ground truth'), numBarrierSegments, bPosition, bYaw, bLength, bWidth, ...
                         'OriginOffset', bOriginOffset, 'Color', bColor);
  
      % 准备和更新检测显示
      N = numel(detections);
      detPos = zeros(N, 2);
      isRadar = true(N, 1);
      for i = 1:N
          detPos(i, :) = detections{i}.Measurement(1:2)';
          if detections{i}.SensorIndex > 6 % 视觉检测
              isRadar(i) = false;
          end
      end
      plotDetection(findPlotter(BEP, 'DisplayName','vision'), detPos(~isRadar, :));
      plotDetection(findPlotter(BEP, 'DisplayName','radar'), detPos(isRadar, :));
  
      % 在更新轨迹显示之前，删除视觉检测生成器未识别的所有对象轨迹
      % 这些对象的 ObjectClassID 参数值为 0，并包括障碍等对象
      isNotBarrier = arrayfun(@(t)t.ObjectClassID, confirmedTracks) > 0;
      confirmedTracks = confirmedTracks(isNotBarrier);
  
      % 准备和更新轨迹显示
      trackIDs = {confirmedTracks.TrackID};
      labels = cellfun(@num2str, trackIDs, 'UniformOutput', false);
      [tracksPos, tracksCov] = getTrackPositions(confirmedTracks, psel);
      tracksVel = getTrackVelocities(confirmedTracks, vsel);
      plotTrack(findPlotter(BEP, 'DisplayName', 'track'), tracksPos, tracksVel, tracksCov, labels);
  end