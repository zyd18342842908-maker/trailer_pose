# trailer_pose：拖挂/托盘位姿估计

多车型流程已接通：在标注器中填写车型，按车型划分数据并联合训练，部署时自动识别或指定车型。详见 [多车型标注、训练与推理](docs/multi_type_training.md)。现有 `best_pose_model.pth` 兼容保留，多车型能力需要重新训练。

`trailer_pose` 是一个 ROS1 Catkin 功能包，使用激光雷达点云估计拖挂/托盘的：

- 挂钩纵向位置 `hook_x`，单位：米；
- 挂钩横向位置 `hook_y`，单位：米；
- 拖挂朝向角 `angle`，独立脚本中单位为度，ROS 消息中以 yaw 四元数表示。

算法路线为 **点云 → BEV（鸟瞰图）→ CNN 回归**。工程使用 Python 和 PyTorch，不包含 C++ 推理实现；既支持直接读取 PCD 文件，也支持订阅 ROS `PointCloud2` 在线推理。

## 1. 工程结构

```text
catkin_ws/
├── src/
│   ├── CMakeLists.txt                 # Catkin 工作空间入口
│   └── trailer_pose/
│       ├── CMakeLists.txt             # 功能包构建与 Python 脚本安装规则
│       ├── package.xml                # ROS/Python 运行依赖
│       ├── README.md
│       ├── launch/
│       │   ├── offline_test.launch    # PCD 发布节点 + ROS 推理节点
│       │   └── trailer_pose.launch    # 默认在线推理节点
│       ├── models/
│       │   └── best_pose_model.pth    # 当前模型权重
│       ├── scripts/
│       │   ├── analyze_sessions.py    # 从点云生成段级伪标签
│       │   ├── annotate_sessions.py   # 人工查看、修正段级标签
│       │   ├── annotate_pose_gui.py   # BEV 拖拽标注位置和朝向，保存后可断点继续
│       │   ├── prepare_dataset.py     # 标签展开与 train/val/test 划分
│       │   ├── preprocess_bags.py     # bag 抽帧、去地面、ROI 裁剪并导出 PCD
│       │   ├── train_bev.py           # BEV、数据集、网络、训练与验证
│       │   ├── inference.py           # 不依赖 ROS 的 PCD 推理
│       │   ├── pcd_publisher.py       # PCD 转 PointCloud2，供离线联调
│       │   └── trailer_pose_node.py   # ROS 在线推理节点
│       ├── data/
│       │   ├── bag_combined/          # record*/ 目录下的原始 PCD 数据
│       │   ├── session_analysis.csv   # 段级伪标签或人工修正后的标签
│       │   ├── dataset_labels.csv     # 帧级训练标签及数据划分
│       │   ├── session_labels.csv     # 已有的段级分箱统计结果
│       │   └── test_predictions.csv   # 已有的测试集预测结果
│       └── docs/
│           ├── 采集数据.jpg
│           └── images/                # 数据覆盖、点云和测试结果图
├── build/                             # Catkin 构建目录
└── devel/                             # Catkin 开发环境
```

`session_labels.csv`、`test_predictions.csv` 和 `docs/images/` 是已有分析产物；当前 Python 脚本不会自动重新生成这些文件。

## 2. 整体流程

```text
ROS1 bag 的 PointCloud2 topic
                    │
       preprocess_bags.py（抽 20 帧、去地面、ROI 裁剪）
                    │
                    ▼
每个 record* 目录表示一个静态/近似静态采集段
                    │
                    ▼
       analyze_sessions.py（可选，生成伪标签）
                    │
                    ▼
          session_analysis.csv
                    │
          annotate_sessions.py（人工修正）
                    │
                    ▼
          prepare_dataset.py
       按“段”划分 train/val/test，再展开到帧
                    │
                    ▼
           dataset_labels.csv
                    │
                    ▼
              train_bev.py
       点云转 BEV → CNN 训练 → 验证集选优
                    │
                    ▼
          best_pose_model.pth
             │              │
             ▼              ▼
       inference.py   trailer_pose_node.py
       单帧/目录推理   PointCloud2 在线推理
                            │
                            ▼
                 /trailer_pose（PoseStamped）
```

训练数据按采集段标注：同一段内所有帧共用同一组 `hook_x`、`hook_y`、`angle`。因此，采集时应保证同一段内目标静止或近似静止。

## 3. 环境准备与编译

### 3.1 环境

推荐环境：

- Ubuntu 20.04；
- ROS1 Noetic；
- Python 3.8 及以上；
- NumPy、SciPy、Matplotlib、PyTorch；
- 可选 NVIDIA GPU 和可用的 CUDA 版 PyTorch。未检测到 CUDA 时自动使用 CPU。

当前工程已在 Python 3.8.10、PyTorch 2.4.1 CPU 环境完成编译、短训练和推理验证。仓库没有锁定依赖版本的 `requirements.txt`。

安装 Python 依赖：

```bash
pip3 install torch numpy scipy matplotlib
```

标注工具使用 Matplotlib 的 `TkAgg` 后端；若提示缺少 Tkinter，在 Ubuntu 上安装：

```bash
sudo apt install python3-tk
```

### 3.2 编译 Catkin 工作空间

```bash
cd /home/zyd/tuogua/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

新终端中使用 ROS 节点前，需要加载工作空间：

```bash
source /opt/ros/noetic/setup.bash
source /home/zyd/tuogua/catkin_ws/devel/setup.bash
```

纯 PCD 推理不需要启动 ROS，也不需要 `source devel/setup.bash`。

## 4. 数据格式与坐标约定

### 4.1 目录和文件

原始数据放在：

```text
src/trailer_pose/data/bag_combined/
├── record20260818_103645/
│   ├── record20260818_103645_0001.pcd
│   ├── record20260818_103645_0002.pcd
│   └── ...
├── record20260818_103745/
└── ...
```

要求：

- 子目录以 `record` 开头；
- 每段建议包含 10～30 帧连续点云；
- PCD 为 ASCII 格式，当前读取器不支持二进制 PCD；
- 现有点云字段为 `x y z intensity`，均为 `float32`；
- 文件名为 `recordYYYYMMDD_HHMMSS_NNNN.pcd`。

当前数据与训练资产如下：

| 项目 | 数量/结果 |
|---|---:|
| 原始采集段 | 647 段 |
| 原始 PCD | 12,664 帧 |
| `session_analysis.csv` 标签段 | 636 段 |
| 通过筛选并参与划分 | 505 段、10,077 帧 |
| train | 353 段、7,040 帧 |
| val | 75 段、1,565 帧 |
| test | 77 段、1,472 帧 |

### 4.2 坐标和 ROI

当前工程约定：

| 轴 | 范围 | 含义 |
|---|---|---|
| x | `[-5.2, 0.0]` m | 负 x 指向牵引车后方的拖挂区域 |
| y | `[-1.2, 1.2]` m | 横向偏移，当前数据约定为左负右正 |
| z | `[-0.18, 0.20]` m | 目标高度层，过滤地面和车顶 |

角度由拖挂尾部指向挂钩的方向计算，使用 `atan2(hook_y-back_y, hook_x-back_x)`；独立推理输出角度单位为度。

**训练和推理必须使用一致的安装位置、坐标变换和 ROI。** 当前 `points_to_bev()` 只按 x/y 范围落栅格，不会主动裁剪 z 范围；在线输入前必须在外部完成坐标变换及 x/y/z ROI 滤波。网络实际只使用 x、y、z，intensity 不参与 BEV 特征计算，但 ROS 节点默认要求输入消息包含 `intensity` 字段。

采集方案可参考 `docs/采集数据.jpg`。数据应覆盖实际使用中的距离、横向偏移和角度范围，避免只采中间位置或样本分布严重失衡。

### 4.3 从 ROS bag 预处理生成 PCD

`scripts/preprocess_bags.py` 读取 **ROS1 `.bag` 中的 `sensor_msgs/PointCloud2` topic**，默认对每个 bag 均匀抽取 20 帧，在原始雷达坐标系中保留 **`-6 ≤ x ≤ 0 m`、`-1 ≤ y ≤ 1 m`**，去除地面，然后逐帧保存为 ASCII PCD。只需要加载 ROS 环境，不需要启动 ROS Master 或播放 bag。

以下命令在**工作空间根目录**执行，将路径和 topic 替换为实际值：

```bash
source /opt/ros/noetic/setup.bash

# 查看实际 topic 及其类型
rosbag info /path/to/example.bag

# 单个 bag；--input 也可传多个 bag
python3 src/trailer_pose/scripts/preprocess_bags.py \
  --input /path/to/example.bag \
  --topic /lidar/points \
  --output_dir src/trailer_pose/data/bag_preprocessed

# 批量处理目录下的所有 bag，包含子目录
python3 src/trailer_pose/scripts/preprocess_bags.py \
  --input /path/to/bags \
  --topic /lidar/points \
  --output_dir src/trailer_pose/data/bag_preprocessed
```

输出示例：

```text
data/bag_preprocessed/
└── record_example/
    ├── record_example_0001.pcd
    ├── ...
    ├── record_example_0020.pcd
    └── manifest.json
```

一个 bag 对应一个采集段，20 帧分别保存，不合并点云。PCD 字段顺序为 `x y z intensity`；原 topic 没有 intensity 时补零，非有限 intensity 也补零；非有限 XYZ 点直接丢弃。`manifest.json` 保存输入 bag、topic、参数、每帧原始索引（从 0 开始）、bag/消息时间戳、原始 frame_id、地面平面和过滤点数，便于检查结果。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--frames` | `20` | 每个 bag 抽取的帧数 |
| `--sampling` | `uniform` | `uniform` 在可用帧中均匀取样；`first` 取连续前 N 帧 |
| `--start_frame` | `0` | 从指定 topic 的该帧起取样，索引从 0 开始 |
| `--x_range` | `-6 0` | 原始坐标系 x 范围，包含边界 |
| `--y_range` | `-1 1` | 原始坐标系 y 范围，包含边界 |
| `--ground_threshold` | `0.05` | 地面法向距离阈值，单位米 |
| `--max_ground_tilt` | `15` | 地面相对 XY 平面的最大倾角，单位度 |
| `--ground_z` | 自动估计 | 已知地面在雷达坐标原点处的 z，单位米 |
| `--ground_band` | `0.3` | 指定 ground_z 时的候选高度带半宽，单位米 |
| `--min_ground_points` | `30` | 地面拟合最少支持点数 |
| `--ransac_iterations` | `200` | 每帧 RANSAC 拟合次数 |
| `--rotate_z_deg` | `0` | 裁剪及去地面后绕 z 轴旋转的角度，单位度 |
| `--seed` | `42` | 地面拟合随机种子 |

去地面假设 **z 轴朝上，地面近似平面且出现在 ROI 中**。默认从 ROI 中 z 最低的 30% 点中用 RANSAC 拟合近水平地面，再以最小二乘精修；删除地面及其下方的点，仅保留高于平面法向距离阈值的点。地面回波很少、目标底板较大或存在低处异常回波时，自动拟合可能把目标表面当成地面，建议指定实际 `--ground_z`。例如雷达坐标原点距离地面约 0.8 米且 z 轴朝上时，可加 `--ground_z -0.8 --ground_band 0.2`。候选高度带应覆盖 ROI 内的地面起伏。

不足 20 帧、指定 topic 不存在、消息不是 PointCloud2、没有有效 ROI 点、无法拟合地面或过滤后为空时，该 bag 报错，不留下部分 PCD；批量任务继续处理其他 bag，最终以非零状态报告失败。已存在的采集段不会被覆盖；不同 bag 生成同名采集段时会要求先重命名。原始雷达数据包或厂商自定义消息需要先由对应驱动解码成 PointCloud2。

默认保留原始坐标，不需要绕 z 轴旋转。预处理保留 x 的 `[-6, 0]` 米范围，当前网络 BEV 仍按 `[-5.2, 0]` 米构造特征，不会随预处理范围自动扩展；若需要网络覆盖完整 6 米，应另行统一训练与推理的 BEV 配置并重新训练。新旧数据混用前仍需核对雷达安装位置、完整坐标和高度约定。

每个 bag 内目标应静止或近似静止，以便沿用“一段一组位姿标签”的标注流程。若只需某个静止片段，例如从索引 100 起连续 20 帧，可加 `--sampling first --start_frame 100`。处理完成后，在标注器中打开输出根目录：

```bash
python3 src/trailer_pose/scripts/annotate_pose_gui.py \
  --data_dir src/trailer_pose/data/bag_preprocessed \
  --output src/trailer_pose/data/preprocessed_manual_labels.csv
```

## 5. 训练

以下命令均在功能包目录执行：

```bash
cd /home/zyd/tuogua/catkin_ws/src/trailer_pose
```

完整流程为：**生成伪标签 → 人工修正 → 划分数据集 → 训练模型**。如果已有可靠的外部段级真值，可跳过前两步，直接提供符合格式要求的 `session_analysis.csv`。

### 5.1 生成段级伪标签（可选）

```bash
python3 scripts/analyze_sessions.py \
  --data_dir data/bag_combined \
  --output data/session_analysis.csv
```

处理逻辑：

1. 每段读取排序后的前 5 帧并合并点云；
2. 保留 `x < -0.8 m` 的点，在 x/y 平面以 0.05 m 分辨率栅格化；
3. 提取最大连通点簇；
4. 以前端点估计挂钩、后端点估计拖挂尾部；
5. 输出挂钩位置、方向角和拖挂长度等字段。

输出示例：

```csv
session,n_frames,n_points,hook_x,hook_y,back_x,back_y,centroid_x,centroid_y,angle,length,main_points
record20260818_103645,19,12417,-1.216,0.179,-3.053,0.172,-2.444,0.215,0.213,1.836,11704
```

这是几何算法生成的**伪标签**，不能默认视为准确真值。命令会覆盖指定输出文件，已有修正标签时应先备份。

### 5.2 人工检查和修正标签

推荐使用 [x、y、heading 可视化标注器](docs/manual_pose_annotation.md)，可直接拖动挂钩中心和朝向手柄，并支持保存、撤销和断点继续：

```bash
python3 scripts/annotate_pose_gui.py \
  --data_dir data/bag_combined \
  --initial_csv data/session_analysis.csv \
  --output data/session_manual.csv
```

窗口中可直接打开 PCD 或选择数据目录，左侧列表支持按名称搜索和按确认状态筛选。默认快捷标注无需切换模式：左键位置、右键或蓝点调整朝向、`Shift+左键拖框` 放大、滚轮缩放、中键拖动平移、双击中键恢复全图。右键调整朝向后可立即继续左键移动挂钩。`Ctrl+滚轮` 微调 heading，`F` 恢复全图、`V` 上一视图；顶部保留可选的独立操作模式。`Enter` 保存并进入下一段，`S` 仅保存，`B` 返回，`N` 跳过。只有人工保存确认的段会写入输出；heading 以度为单位保存到 `angle` 列。

也可使用 Open3D 在真实点云中点选挂钩和方向参考点，完整操作及已生成的演示结果见 [Open3D 数据标注示例](docs/data_annotation_demo.md)。

以下为旧版终端输入标注工具的用法，保存后生成的仍是 `session_analysis.csv`：

```bash
python3 scripts/annotate_sessions.py \
  --data_dir data/bag_combined \
  --output data/session_analysis.csv
```

工具逐段显示前 5 帧叠加的 BEV 散点、挂钩位置和方向线。需要本地图形界面或 X11 转发。

| 输入 | 操作 |
|---|---|
| `Enter` | 保留当前值并进入下一段 |
| `x,y,angle` | 修改位姿，例如 `-1.2,0.15,5.0` |
| `x,y,angle,length` | 同时修改位姿和长度 |
| `s` | 跳过当前段 |
| `b` | 返回上一段 |
| `q` | 保存并退出 |

退出或遍历完成后，工具会覆盖保存输出 CSV。也可以直接用 CloudCompare、RViz 等工具核对点云后编辑 CSV。

使用外部真值时，至少提供以下列，其他统计列可以省略：

```csv
session,hook_x,hook_y,angle,length
record20260818_103645,-1.216,0.179,0.213,1.836
```

其中 `session` 必须和 `data/bag_combined/` 下的段目录名完全一致，`angle` 单位为度。

### 5.3 生成帧级标签并划分数据集

```bash
python3 scripts/prepare_dataset.py \
  --data_dir data/bag_combined \
  --session_csv data/session_manual.csv \
  --output data/dataset_labels.csv \
  --seed 42
```

段级筛选条件为：

```text
-3.5 < hook_x < -0.9
abs(hook_y) < 0.9
length > 0.5
```

通过筛选的段在每种车型内按约 **70% train、15% val、15% test** 划分，随后把段标签展开到该段的所有 PCD 帧。每种车型至少 3 段才能覆盖三个集合。`--seed 42` 保证相同输入下的划分可复现。

上述命令使用新版可视化标注器输出的 `session_manual.csv`；如果使用旧版工具或外部标签，将 `--session_csv` 改为相应文件路径。

同一绝对目录 `group_id` 的所有帧只会进入一个集合，避免相邻帧同时出现在训练集和测试集造成数据泄漏。`--session_csv` 支持多个 CSV；优先使用行内 `session_dir`，不同根目录的同名采集段独立处理。缺少 `trailer_type` 的旧标签按 `default` 读取，多车型训练前应补齐真实车型。

注意：

- 命令会覆盖 `dataset_labels.csv`；
- CSV 中保存的是 PCD **绝对路径**；移动工作空间或数据后，需要重新运行本步骤；
- 没有有效段时会无法生成输出，应先检查标签和 PCD 目录。

### 5.4 训练模型

```bash
python3 scripts/train_bev.py \
  --label_csv data/dataset_labels.csv \
  --model_path models/best_pose_model.pth \
  --epochs 40 \
  --batch_size 32 \
  --lr 0.001
```

各参数默认值与上面一致。实验时建议输出到新文件，避免覆盖当前最佳模型：

```bash
python3 scripts/train_bev.py \
  --label_csv data/dataset_labels.csv \
  --model_path models/experiment_pose_model.pth \
  --epochs 40
```

#### 输入表示

每帧点云被转换为 `2×128×256` 的 BEV 张量：

- 通道 0：每个栅格的点数，经 `log1p(count) / log1p(20)` 归一化；
- 通道 1：每个非空栅格的平均 z；
- x/y 范围分别为 `[-5.2,0]` 和 `[-1.2,1.2]`。

训练集默认进行随机增强：±5° 旋转、y 轴镜像、随机丢弃 20% 点和 0.01 m 高斯噪声；验证集不做增强。

#### 网络和损失

`PoseNet` 由 4 组卷积/BatchNorm/ReLU、全局平均池化和全连接回归头组成，输出 4 个量：

```text
normalized_hook_x, normalized_hook_y, sin(angle), cos(angle)
```

使用正弦/余弦表示角度，避免角度在 ±180° 处不连续。训练损失为：

```text
SmoothL1(hook_x, hook_y) + 0.5 × MSE(sin(angle), cos(angle))
```

优化器为 Adam，初始学习率由 `--lr` 控制，每 20 轮学习率减半。每轮在验证集计算：

- `hx_mae`：挂钩 x 平均绝对误差，米；
- `hy_mae`：挂钩 y 平均绝对误差，米；
- `ang_mae`：角度平均绝对误差，度。

单车型模型选择分数为 `hx_mae + hy_mae + 0.02 × ang_mae`，分数降低时覆盖保存 `--model_path`。训练脚本只评估 train/val，不会自动运行 test 或生成 `test_predictions.csv`。

多车型训练自动使用 `MultiTypePoseNet`：共享 BEV 特征、车型分类头、各车型独立位姿头，损失增加 `0.2 × 车型交叉熵`。默认均衡采样，验证时报告各车型的位姿误差和分类准确率，最佳模型按各车型自动推理分数的平均值选择。权重文件包含车型词表，仍支持加载原版纯 `state_dict`。完整命令见 [多车型说明](docs/multi_type_training.md)。

### 5.5 当前模型的已有测试结果

根据仓库现有 `data/test_predictions.csv` 的 1,472 条记录重新汇总：

| 指标 | MAE |
|---|---:|
| `hook_x` | 0.208 m |
| `hook_y` | 0.177 m |
| `angle` | 10.124° |

这些数值是已有结果的统计，不代表重新训练后一定得到相同精度。当前工程没有一键重算完整测试集指标的脚本。

## 6. 独立 PCD 推理

独立推理不需要 ROS，直接使用当前权重：

```bash
cd /home/zyd/tuogua/catkin_ws/src/trailer_pose

# 单帧
python3 scripts/inference.py \
  data/bag_combined/record20260818_103645/record20260818_103645_0001.pcd
```

当前模型实测输出：

```text
hook_x: -1.143 m
hook_y: 0.053 m
angle:  0.28 deg
```

逐帧处理一个目录：

```bash
python3 scripts/inference.py \
  data/bag_combined/record20260818_103645
```

对目录内**全部 PCD**做平均：

```bash
python3 scripts/inference.py \
  data/bag_combined/record20260818_103645 \
  --multi
```

`--multi` 对 x/y 做算术平均，对角度通过平均正弦/余弦后求 `atan2`，以正确处理角度环绕。

指定其他权重：

```bash
python3 scripts/inference.py INPUT.pcd \
  --model models/experiment_pose_model.pth
```

输入可以是文件或目录；目录模式只读取该目录直接包含的 `*.pcd`，不会递归搜索子目录。

## 7. ROS 推理

### 7.1 离线 PCD 联调

加载环境并启动：

```bash
cd /home/zyd/tuogua/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch trailer_pose offline_test.launch
```

`roslaunch` 会在需要时自动启动 ROS Master。该 launch 同时运行：

1. `pcd_publisher.py`：按文件名排序，以 10 Hz 循环发布默认示例段到 `/points_raw`；
2. `trailer_pose_node.py`：加载当前模型，订阅点云并发布 `/trailer_pose`。

指定其他段、频率和滑动窗口：

```bash
roslaunch trailer_pose offline_test.launch \
  pcd_dir:=/home/zyd/tuogua/catkin_ws/src/trailer_pose/data/bag_combined/record20260818_103645 \
  rate:=10 \
  multi_frame:=5
```

在另一个终端查看输出：

```bash
source /opt/ros/noetic/setup.bash
source /home/zyd/tuogua/catkin_ws/devel/setup.bash
rostopic echo /trailer_pose
```

输出示例：

```yaml
header:
  stamp:
    secs: 1794364338
    nsecs: 413646221
  frame_id: "lidar"
pose:
  position:
    x: -1.1711285448074342
    y: 0.08226366490125656
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.006367709295045814
    w: 0.999979725933648
```

### 7.2 在线推理

先启动雷达驱动和预处理节点，将处理后的点云发布为 `/points_raw`，然后运行：

```bash
source /opt/ros/noetic/setup.bash
source /home/zyd/tuogua/catkin_ws/devel/setup.bash
roslaunch trailer_pose trailer_pose.launch
```

默认配置为：

| 配置 | 默认值 |
|---|---|
| 输入 topic | `/points_raw` |
| 输出 topic | `/trailer_pose` |
| 模型 | `$(find trailer_pose)/models/best_pose_model.pth` |
| 备用 frame_id | `lidar` |
| 滑动窗口 | 5 帧 |

`trailer_pose.launch` 支持 `model_path / trailer_type / type_topic / multi_frame` 参数；`trailer_type` 留空时自动识别，填写车型名时使用对应位姿头。需要修改输入 topic 等其他配置时，可直接通过私有参数启动节点：

```bash
rosrun trailer_pose trailer_pose_node.py \
  _model_path:=/home/zyd/tuogua/catkin_ws/src/trailer_pose/models/best_pose_model.pth \
  _cloud_topic:=/points_filtered \
  _pose_topic:=/trailer_pose \
  _frame_id:=lidar \
  _multi_frame:=5
```

本包提供上述离线 bag 预处理脚本；在线雷达驱动和坐标/ROI 预处理节点仍需根据实际雷达型号和车体坐标系另外配置。

### 7.3 ROS 节点处理细节

- 输入类型：`sensor_msgs/PointCloud2`；
- 默认读取字段：`x y z intensity`；
- 少于 10 个有效点时，只输出警告，不发布位姿；
- BEV 转换和输出解码与独立 PCD 推理完全一致；
- 自动选择 CUDA，不可用时使用 CPU；
- `multi_frame` 是最近 N 个有效推理结果的滑动窗口，启动初期不足 N 帧时按现有帧平均；
- 位置使用算术平均，角度使用正弦/余弦平均；
- 车型变化时清空平滑窗口，避免混合不同车型的位姿；
- 输出时间戳沿用输入点云时间戳；
- 输入 `frame_id` 非空时沿用输入值，否则使用 `~frame_id` 参数。

## 8. 输出接口与控制端解码

| Topic | 类型 | 方向 | 内容 |
|---|---|---|---|
| `/points_raw` | `sensor_msgs/PointCloud2` | 订阅 | 与训练数据同坐标系、同 ROI 的点云 |
| `/trailer_pose` | `geometry_msgs/PoseStamped` | 发布 | 挂钩位置和拖挂朝向 |
| `/trailer_type` | `std_msgs/String` | 发布 | 当前所选车型；旧模型为 `default` |

`pose.position.x/y` 是**挂钩位置**，不是拖挂质心；`pose.position.z` 固定为 0。朝向只包含 yaw，转换为：

```text
orientation.z = sin(yaw / 2)
orientation.w = cos(yaw / 2)
orientation.x = orientation.y = 0
```

Python 控制节点解码示例：

```python
import math
from tf.transformations import euler_from_quaternion

q = msg.pose.orientation
_, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
hook_x = msg.pose.position.x
hook_y = msg.pose.position.y
angle_deg = math.degrees(yaw)
```

接入控制前，应在实车上再次确认 x/y 方向、角度正方向和期望对接位置。离线多车型推理的分类概率不代表位姿精度；模型没有位姿置信度或目标存在概率。控制系统需要额外处理点云为空、点云过少、长时间无更新和位姿越界等异常情况。

## 9. 常见问题

### 9.1 点云没有 intensity 字段

独立 PCD 推理的网络特征不使用 intensity，但 `trailer_pose_node.py` 当前显式读取该字段。在线点云缺少它时，应在预处理阶段补充 intensity，或修改节点为只读取 `x y z`。修改读取字段后，构造点数组时仍需保持与 `points_to_bev()` 兼容。

### 9.2 有输入点云，但 `/trailer_pose` 没有输出

依次检查：

```bash
rosnode list
rostopic hz /points_raw
rostopic echo /points_raw -n 1
```

确认：

- 节点实际订阅的 topic 与雷达预处理输出一致；
- 消息包含 `x y z intensity`；
- 每帧至少有 10 个非 NaN 点；
- 模型路径存在且权重与 `PoseNet` 结构匹配。

### 9.3 移动工程后训练提示找不到 PCD

`dataset_labels.csv` 保存绝对路径。段级 CSV 如果含有 `session_dir`，先更新这些目录，再运行 `prepare_dataset.py`。

### 9.4 输出抖动

- 增大在线节点的 `multi_frame`，例如 5～10；
- 检查雷达安装和点云是否稳定；
- 确认在线 ROI 与训练数据一致；
- 必要时在控制端增加低通滤波。

增大窗口会降低响应速度，应根据控制频率和车辆运动速度权衡。

### 9.5 精度不足或新场景误差大

优先检查标签质量、坐标/ROI 一致性和数据覆盖范围，再调整训练轮数或网络。CNN 可以在训练分布内插值，但不能保证对未覆盖距离、角度、横向偏移或不同雷达安装方式可靠外推。

## 10. 常用命令速查

```bash
# 编译
cd /home/zyd/tuogua/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

# 训练
cd /home/zyd/tuogua/catkin_ws/src/trailer_pose
python3 scripts/analyze_sessions.py --data_dir data/bag_combined --output data/session_analysis.csv
python3 scripts/annotate_pose_gui.py --data_dir data/bag_combined --initial_csv data/session_analysis.csv --output data/session_manual.csv
python3 scripts/prepare_dataset.py --data_dir data/bag_combined --session_csv data/session_manual.csv --output data/dataset_labels.csv
python3 scripts/train_bev.py --label_csv data/dataset_labels.csv --model_path models/best_pose_model.pth --epochs 40

# 单帧独立推理
python3 scripts/inference.py data/bag_combined/record20260818_103645/record20260818_103645_0001.pcd

# 目录多帧平均
python3 scripts/inference.py data/bag_combined/record20260818_103645 --multi

# ROS 离线联调
cd /home/zyd/tuogua/catkin_ws
source devel/setup.bash
roslaunch trailer_pose offline_test.launch

# ROS 在线默认配置
roslaunch trailer_pose trailer_pose.launch

# 查看结果
rostopic echo /trailer_pose
```
