# YOLO Face Detection

模型文件：

```text
yolov12n-face.onnx
```

用途：多帧人脸检测、候选框置信度排序和全局 Top-1 选择。

输入为 RGB/BGR 图像预处理后的 ONNX 张量，输出由 `FaceDetector` 解码并执行 NMS。

该模型体积小于 GitHub 单文件限制，已直接保存在 Git 仓库中。
