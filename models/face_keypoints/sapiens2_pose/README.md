# Sapiens2 Pose

模型文件：

```text
sapiens2_pose_0.4b_fp32.onnx
```

用途：输出 Sapiens2 定义的 308 个姿态关键点；人脸处理使用索引 `70–307`，再结合 Seg mask 筛除遮挡点。

输入形状：`[1, 3, 1024, 768]`。输出为 308 通道关键点热图。

模型约 1.62GB，不存入 Git。请从根目录 README 中的 Google Drive“模型参数包”下载并解压到本目录。
