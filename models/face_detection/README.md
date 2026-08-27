# 人脸检测 ONNX 模型

## 文件

```text
models/face_detection/yolo_face/yolov12n-face.onnx
```

模型来源：<https://github.com/akanametov/yolo-face>

## 输入输出

- 输入：`images`，`float32`，形状 `[1, 3, 640, 640]`；
- 输出：`output0`，`float32`，形状 `[1, 300, 6]`；
- 输出由 `FaceDetector` 按 `x1, y1, x2, y2, score, class` 解码并执行 NMS。

## 验证

在仓库根目录执行：

```powershell
.\build\C++\Release\face_model_validation.exe `
  .\models\face_detection\yolo_face\yolov12n-face.onnx `
  .\models\face_keypoints\hrnet\hrnetv2_w18_wflw_256x256_heatmap.onnx
```

灰色测试图验证的是模型加载、张量形状、推理与解码通路，不代表真实人脸检测精度；真实效果使用 `face_camera_pipeline_demo.exe` 验证。

## 运行环境

ONNX Runtime 已包含在 `thirdparty/packages/windows/onnxruntime-win-x64-gpu-1.24.4.zip`。CPU 推理无需 CUDA；只有启用 CUDA provider 时才需要与该 ONNX Runtime 包匹配的 NVIDIA CUDA/cuDNN 环境。
