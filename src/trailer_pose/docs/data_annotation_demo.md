# 用 Open3D 标注一段训练数据

本示例只处理训练数据的 `hook_x`、`hook_y`、`angle` 和 `length` 标签，直接使用已有 PCD 坐标。示例段为 `record20260818_103645`，包含 19 帧；显示时叠加按文件名排序的前 5 帧，共 12,637 个有效点。

已生成的点位示意：

![真实点云中的挂钩和方向参考点](../data/annotation_demo/annotation_demo.png)

图中点位为演示选点，未经过现场尺寸和挂钩中心真值核验。当前会话无法连接 X11 桌面，因此生成结果采用指定点索引回放；这张图是 Matplotlib 导出的点位说明图，不是 Open3D 窗口截图。可在本地图形桌面中执行下列命令，完成实际鼠标点选。

## 1. 打开指定采集段

工程自带的 `.venv-labelcloud` 已安装 Open3D 0.19.0。以下命令在工作空间根目录执行，无需启动 ROS：

```bash
cd /home/zyd/tuogua/catkin_ws

.venv-labelcloud/bin/python src/trailer_pose/scripts/annotate_open3d.py \
  src/trailer_pose/data/bag_combined/record20260818_103645 \
  --length 1.836424831630237 \
  --source-csv src/trailer_pose/data/session_analysis.csv \
  --output src/trailer_pose/data/annotation_manual/session_labels_manual.csv \
  --preview src/trailer_pose/data/annotation_manual/annotation_preview.png
```

`--length` 的示例值沿用该段已有几何标签，仅用于演示保存流程。正式标注时应填入挂钩连接中心到所定义尾部参考中心的实测距离，单位为米；它不是车宽，也不是任选的 A、B 两点间距。当前网络不回归长度，但数据筛选和方向线显示使用该值。

`--source-csv` 复制已有段级标签，仅更新本次选择的段。工具要求输出路径与源 CSV 不同，并拒绝覆盖已存在的输出文件。需要重新操作时请使用新的输出文件名。

其他机器可单独安装 Open3D：

```bash
python3 -m venv .venv-open3d
.venv-open3d/bin/python -m pip install "open3d==0.19.0" matplotlib
```

然后将运行命令中的 `.venv-labelcloud/bin/python` 替换为 `.venv-open3d/bin/python`。

## 2. 按顺序选择四个点

默认从俯视方向打开点云。拖动鼠标可调整视角，滚轮缩放；改变观察视角不改变点云坐标。

按住 `Shift` 并依次单击：

| 顺序 | 点 | 选择位置 |
| --- | --- | --- |
| 1 | H1 | 挂钩孔轮廓的一侧 |
| 2 | H2 | 同一轮廓、同一直径上与 H1 相对的另一侧 |
| 3 | A | 拖挂纵向直边上靠近挂钩的一点 |
| 4 | B | 同一条直边上靠近尾部的一点 |

H1、H2 应来自同一圆形轮廓相对的两侧，中点才能代表孔中心。圆心处通常没有回波，直接点选只能得到表面点。

A、B 所在直线需要与拖挂纵轴平行。图中选择车架上方的直边，**不要沿 V 字形牵引杆的斜边确定拖挂朝向**。尽量选间隔较远的清晰点，正式高精度标注可使用同一直边的多个点拟合方向。

选错时按 `Shift + 右键` 撤销最近一点；四点选完后按 `Q` 关闭窗口并保存。工具不会将只有默认值、没有点选结果的段作为本次人工标注写入。

如果连接中心本身能通过可靠特征直接定位，可加 `--direct-hook`，改为依次选择 H、A、B 共三点。

## 3. 计算标签

默认四点模式使用：

```text
H = (H1 + H2) / 2
hook_x = H.x
hook_y = H.y
angle = atan2(A.y - B.y, A.x - B.x) × 180 / π
```

角度方向是由尾部向挂钩，单位为度。脚本再沿此方向的反方向，按照给定 `length` 计算用于显示的尾部参考中心。

本次演示使用真实点云中的以下点，索引对应前 5 帧排序拼接并去除非有限点后的点数组：

| 点 | 索引 | x / m | y / m |
| --- | ---: | ---: | ---: |
| H1 | 10779 | -1.256421 | 0.251768 |
| H2 | 5488 | -1.249081 | 0.153571 |
| A | 9796 | -2.597292 | 0.803401 |
| B | 9903 | -2.949415 | 0.803075 |

实际计算输出：

```text
hook_x: -1.252751
hook_y: 0.202669
angle: 0.053032
length: 1.836425
```

角度接近 0°，表示该示例中拖挂纵向轴基本平行于点云的 x 轴。数值用于演示点选到标签的计算，不作为精度评估真值。

## 4. 查看保存文件

本次实际生成的文件位于 `src/trailer_pose/data/annotation_demo/`：

| 文件 | 内容 |
| --- | --- |
| [session_labels_demo.csv](../data/annotation_demo/session_labels_demo.csv) | 复制原有 636 段标签，仅更新演示段 |
| [session_labels_demo.picks.json](../data/annotation_demo/session_labels_demo.picks.json) | 输入帧、点索引、点坐标、计算结果及演示标记 |
| [annotation_demo.png](../data/annotation_demo/annotation_demo.png) | 点位与计算结果示意图 |
| [dataset_labels_demo.csv](../data/annotation_demo/dataset_labels_demo.csv) | 基于上述段级标签重新展开的帧级标签 |

CSV 中该段的核心字段为：

```csv
session,hook_x,hook_y,angle,length
record20260818_103645,-1.2527507,0.202669155,0.05303200131049815,1.836424831630237
```

原有 `data/session_analysis.csv`、`data/dataset_labels.csv` 和模型权重保持原样。其余段在演示 CSV 中沿用原始标签，并不表示它们经过了本次人工复核。

## 5. 展开为帧级标签

本次已实际执行：

```bash
python3 src/trailer_pose/scripts/prepare_dataset.py \
  --data_dir src/trailer_pose/data/bag_combined \
  --session_csv src/trailer_pose/data/annotation_demo/session_labels_demo.csv \
  --output src/trailer_pose/data/annotation_demo/dataset_labels_demo.csv \
  --seed 42
```

输出为 505 个有效段、10,077 帧，其中 train 7,040 帧、val 1,565 帧、test 1,472 帧。示例段的全部 19 帧均继承新标签，且仍位于同一集合中。

正式使用时，将 `--session_csv` 改为人工标注输出，`--output` 改为新的帧级标签文件。完成真值核对后再用于训练，演示文件保留为操作示例。

## 6. 无图形界面时复现计算

`--indices` 可以重放已记录的选点，跳过窗口。以下命令使用新文件名，复现本次演示：

```bash
.venv-labelcloud/bin/python src/trailer_pose/scripts/annotate_open3d.py \
  src/trailer_pose/data/bag_combined/record20260818_103645 \
  --length 1.836424831630237 \
  --source-csv src/trailer_pose/data/session_analysis.csv \
  --output src/trailer_pose/data/annotation_replay/session_labels_replay.csv \
  --preview src/trailer_pose/data/annotation_replay/annotation_replay.png \
  --indices 10779 5488 9796 9903 --demo
```

重放索引要求使用相同点云、相同帧排序和相同 `--frames` 值。此工具是独立 Python 工具，未加入 Catkin 节点安装列表。

点选 API 和快捷键参考 [Open3D 官方交互教程](https://www.open3d.org/docs/release/tutorial/visualization/interactive_visualization.html)。
