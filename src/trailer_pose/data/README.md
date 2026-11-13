# 本地点云和标注数据

该目录的数据由本地采集、标注和数据准备脚本生成，GitHub 版本默认只保留本说明。现有数据不会因 `.gitignore` 被删除。

数据布局：

```text
data/
├── bag_combined/
│   └── record*/                 # 每个固定场景的 ASCII PCD 帧
├── manual_pose_labels.csv       # GUI 人工确认的位姿和车型标签
├── session_analysis.csv         # 可选几何伪标签初值
├── dataset_labels*.csv          # 数据划分和帧级标签
└── annotation_demo/             # 基于本地点云生成的标注演示结果
```

从 GitHub 克隆源码后，将已有点云和段级标注复制回对应目录；如果数据绝对路径改变，先更新段级 CSV 的 `session_dir`，再重新运行 `prepare_dataset.py`。标注器会在首次保存时创建输出 CSV。

操作说明：[手工标注](../docs/manual_pose_annotation.md)、[多车型训练](../docs/multi_type_training.md)。
