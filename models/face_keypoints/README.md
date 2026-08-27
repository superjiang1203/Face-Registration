# 人脸关键点 ONNX 模型

## 文件

```text
models/face_keypoints/hrnet/hrnetv2_w18_wflw_256x256_heatmap.onnx
```

## 输入输出

- 输入：`input`，`float32`，形状 `[1, 3, 256, 256]`；
- 输出：`heatmaps`，`float32`，形状 `[1, 98, 64, 64]`；
- 输出由 `FaceKeypointService` 解码为 WFLW 98 点关键点。

## 验证

在仓库根目录执行：

```powershell
.\build\C++\Release\face_model_validation.exe `
  .\models\face_detection\yolo_face\yolov12n-face.onnx `
  .\models\face_keypoints\hrnet\hrnetv2_w18_wflw_256x256_heatmap.onnx
```

灰色测试图验证的是模型加载、张量形状、推理与热图解码通路，不代表真实关键点精度；真实效果使用 `face_camera_pipeline_demo.exe` 生成的 `detected_face.png` 验证。

## 运行环境

模型由命令行显式传入，也可由 `FaceKeypointService` 的构造函数指定。默认路径就是本目录中的上述文件。CPU 推理无需 CUDA。
