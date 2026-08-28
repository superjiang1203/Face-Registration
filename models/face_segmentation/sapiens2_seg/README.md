# Sapiens2 Segmentation

模型文件：

```text
sapiens2_seg_0.4b_fp32.onnx
sapiens2_seg_0.4b_fp32.onnx.data
```

用途：输出人体语义分割结果，包括 `face_neck`、耳朵、头发和遮挡物类别；结合对齐深度筛选目标头部。

模型使用 ONNX external data 格式，`.onnx` 与 `.onnx.data` 必须放在同一目录且文件名保持不变。

权重数据约 1.55GB，不存入 Git。请从根目录 README 中的 Google Drive“模型参数包”下载并解压到本目录。
