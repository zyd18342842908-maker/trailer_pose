# 多车型标注、训练与推理

工程支持一个模型处理多种拖挂车型：自动识别车型后输出 `x / y / heading`，也可在推理时指定已知车型。每帧仍只估计一个拖挂目标。

**已有 `best_pose_model.pth` 是旧版单车型权重，仍可使用，但不会自动获得多车型识别能力。** 下述流程需要为实际车型补充标签，再训练新的模型；`cart_A`、`cart_B` 只是车型名称示例，不代表已训练的车型。

## 1. 标注车型和位姿

在工作空间根目录运行：

```bash
python3 src/trailer_pose/scripts/annotate_pose_gui.py --font_size 20
```

1. 加载采集数据，在右上角“车型”框输入实际车型名称，例如 `cart_A`，按回车提交。
2. 标注该场景的挂钩中心 x、y 和 heading，然后点击“保存并下一段”。车型和位姿一起保存。
3. 连续标注相同车型时，新打开的未标注场景沿用最近设置或保存的车型；遇到另一车型时改为 `cart_B`。已有车型标签和本次会话内的草稿优先，不会被批量改写。

车型框中的回车只提交车型，不保存场景。可通过左侧搜索框按车型名称查找场景；右侧状态会提示车型或位姿是否有未保存修改。启动时可用 `--trailer_type cart_A` 设置未标注场景的默认车型。

车型框使用 Tk 原生输入控件，支持中文输入、`Ctrl+A` 全选和 `Ctrl+V` 粘贴；直接点击保存也会提交当前输入。若此前打开的窗口无法输入，先保存现有标注，关闭旧窗口并重新运行脚本。

各车型都应使用统一的位姿定义：x、y 为实际机械连接中心，heading 为拖挂尾部指向挂钩的纵向方向。不要把某车型的标签改为车架中心。每个固定场景只标注一次，同段所有帧共用标签。

新增的段级字段：

| 字段 | 含义 |
| --- | --- |
| `trailer_type` | 人工确认的车型名称，区分大小写 |
| `session_dir` | 采集段绝对目录，GUI 保存时自动填写 |

旧 CSV 缺少车型时按 `default` 兼容读取。**`default` 表示未补充车型，不能与实际车型名称混在一起进行多车型训练。** 重新打开旧场景、填写车型并保存即可补齐，不必重新调整正确的位姿。

如果不同车型放在不同数据根目录，可分别保存标签。以下路径需要替换为实际位置：

```bash
python3 src/trailer_pose/scripts/annotate_pose_gui.py \
  --data_dir /data/trailers/cart_A \
  --output src/trailer_pose/data/manual_cart_A.csv \
  --trailer_type cart_A --font_size 20

python3 src/trailer_pose/scripts/annotate_pose_gui.py \
  --data_dir /data/trailers/cart_B \
  --output src/trailer_pose/data/manual_cart_B.csv \
  --trailer_type cart_B --font_size 20
```

不同数据集各用一个输出 CSV，数据准备时再合并。这样两个目录下即使都有 `record001`，也不会覆盖标签。GUI 会拒绝用另一个绝对目录覆盖已保存的同名采集段。

## 2. 生成训练数据

如果所有场景已在同一个标注文件中：

```bash
python3 src/trailer_pose/scripts/prepare_dataset.py \
  --data_dir src/trailer_pose/data/bag_combined \
  --session_csv src/trailer_pose/data/manual_pose_labels.csv \
  --output src/trailer_pose/data/dataset_labels_multitype.csv \
  --seed 42
```

如果分别保存为多个标注文件：

```bash
python3 src/trailer_pose/scripts/prepare_dataset.py \
  --data_dir src/trailer_pose/data/bag_combined \
  --session_csv src/trailer_pose/data/manual_cart_A.csv src/trailer_pose/data/manual_cart_B.csv \
  --output src/trailer_pose/data/dataset_labels_multitype.csv \
  --seed 42
```

脚本优先使用每行的 `session_dir`；缺少该列时才使用 `--data_dir/session`。相对的 `session_dir` 按输入 CSV 所在目录解析。移动数据后需要更新这列或在 GUI 中重新打开并保存，再生成帧级 CSV。

脚本在**每种车型内部按采集段**划分约 70% / 15% / 15%，再展开到帧。绝对目录作为 `group_id`，同段全部帧只进入一个集合；同名但路径不同的采集段独立处理。同一目录重复输入会去重，车型或位姿冲突会报错。

每种车型至少需要 3 个不同采集段，才可分别进入 train / val / test；这只是完成划分的最低数量，不代表数据已经足够训练。应覆盖每种车型的实际距离、横向偏移和角度。筛选范围仍为 `-3.5 < hook_x < -0.9`、`abs(hook_y) < 0.9`、`length > 0.5`，`length / n_frames / n_points` 不作为预测目标。

## 3. 训练新模型

```bash
python3 src/trailer_pose/scripts/train_bev.py \
  --label_csv src/trailer_pose/data/dataset_labels_multitype.csv \
  --model_path src/trailer_pose/models/multitype_pose_model.pth \
  --epochs 40 --batch_size 32 --lr 0.001
```

训练脚本自动从训练集生成车型词表。单一车型仍使用原 `PoseNet`；多种车型使用 `MultiTypePoseNet`，由共享 BEV 特征、车型分类头和各车型独立的位姿回归头组成。训练时由真实车型选择对应位姿头，自动推理时由分类结果选择。

多车型损失为原位姿损失加 `0.2 × CrossEntropy(车型)`。默认按车型均衡采样，减少帧数较多车型的主导作用。可使用 `--no_balance_types` 关闭，使用 `--classification_weight` 调整分类损失权重。`--num_workers 0` 可关闭数据加载子进程。

每轮输出整体和各车型的 x/y/heading 误差、车型识别准确率，并单独输出已知车型时的位姿误差。验证集必须包含每种训练车型。最佳模型按**各车型自动推理分数的平均值**选择，单类分数为 `hx_mae + hy_mae + 0.02 × ang_mae`。

新模型文件同时保存网络结构版本、车型名称及权重，推理时直接读取，无需另配车型编号。新增车型后需要使用覆盖各车型的数据重新训练。不要覆盖仍在使用的旧模型。

## 4. 离线推理

自动识别车型：

```bash
python3 src/trailer_pose/scripts/inference.py \
  src/trailer_pose/data/bag_combined/record20260818_103645 \
  --model src/trailer_pose/models/multitype_pose_model.pth --multi
```

已知车型时直接指定：

```bash
python3 src/trailer_pose/scripts/inference.py \
  src/trailer_pose/data/bag_combined/record20260818_103645 \
  --model src/trailer_pose/models/multitype_pose_model.pth \
  --trailer_type cart_A --multi
```

输出包含 `trailer_type / hook_x / hook_y / angle`。Python API 的 `predicted_type` 保留分类器原始判断；指定车型时，位姿始终来自指定车型的回归头。不在模型词表中的车型会报错并列出可选名称。

`type_confidence` 是所选车型的原始分类概率，不是位姿精度或目标存在置信度。旧单车型模型返回 `default`，不输出分类概率。自动多帧模式按多数票确定车型，只平均属于该车型的帧，同时报告总帧数 `n_frames` 和实际平均帧数 `n_frames_used`。

如果两种车型在当前点云视角下难以区分，而连接点几何偏移不同，应指定已知车型。自动分类只在训练过的车型词表中选择，不能据此判断未见车型或空场景。

## 5. ROS 推理

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash

# 自动识别
roslaunch trailer_pose trailer_pose.launch \
  model_path:=$(pwd)/src/trailer_pose/models/multitype_pose_model.pth

# 指定已知车型
roslaunch trailer_pose trailer_pose.launch \
  model_path:=$(pwd)/src/trailer_pose/models/multitype_pose_model.pth \
  trailer_type:=cart_A multi_frame:=5
```

`offline_test.launch` 同样支持 `model_path / trailer_type / type_topic` 参数。节点增加 `~trailer_type`（空字符串为自动）和 `~type_topic`（默认 `/trailer_type`）；车型通过 `std_msgs/String` 发布，位姿继续通过 `/trailer_pose` 发布。更换所选车型时，节点清空历史平滑窗口。

```bash
rostopic echo /trailer_type
rostopic echo /trailer_pose
```

## 验证

```bash
source /opt/ros/noetic/setup.bash
MPLCONFIGDIR=/tmp/trailer_pose_mpl python3 -m unittest discover -s src/trailer_pose/tests -v
```

测试覆盖车型保存恢复、按车型分组划分、同名目录隔离、自动与指定车型的分支选择、新旧权重加载、多车型临时训练和 ROS 平滑窗口切换。测试使用临时标签和临时模型，不覆盖业务数据；点云交互通过 Agg 模拟，原生车型框用模拟 Tk 控件验证提交、焦点及保存流程。实际输入法和系统剪贴板需在本地图形桌面验证。
