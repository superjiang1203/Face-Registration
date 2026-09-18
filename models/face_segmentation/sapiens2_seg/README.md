# Sapiens2 Segmentation

## 模型文件

```text
sapiens2_seg_0.4b_fp32.onnx
sapiens2_seg_0.4b_fp32.onnx.data
```

模型使用 ONNX external data 格式，`.onnx` 与 `.onnx.data` 必须位于同一目录且文件名保持不变。权重数据约 1.55 GB，两个文件均使用 Git LFS 跟踪；克隆仓库后执行 `git lfs install` 和 `git lfs pull`，即可取得完整文件。

## 流水线用途

模型输入为 `image`、`float32`、`[1, 3, 1024, 768]`，输出为 `logits`、`float32`、`[1, 29, 1024, 768]`。模型本身只处理 RGB；对齐深度和点云筛选属于 C++ 应用层。

完整语义输出包括 `face_neck`、耳朵、头发和遮挡物类别；当前 `faceMask` 合并类别 3 和 24–28，并不把所有头发/遮挡类别都视为脸。集成流水线把各个连通的人脸区域作为候选，结合对齐深度和点云执行物理门限；Seg 候选的语义置信度相同时，优先选择接近 `face_selection.preferred_depth_mm` 且点云质量更好的区域。

选择 `pipeline.target_locator: sapiens_seg` 时，Top-1 mask 用于约束后续配准点云。关键点 provider 仍由 `pipeline.keypoint_model` 独立选择。

独立的 Sapiens2 Seg/Pose 调用示例见 [`C++/SAPIENS2.md`](../../../C++/SAPIENS2.md)。
