# HRNet-WFLW Face Keypoints

模型文件：

```text
hrnetv2_w18_wflw_256x256_heatmap.onnx
```

用途：对 Face Detection 全局 Top-1 的人脸 ROI 输出 WFLW 98 点热图。当前配准流程选择其中 6 个命名点建立三维对应。

输入形状：`[1, 3, 256, 256]`。输出形状：`[1, 98, 64, 64]`。

该模型体积小于 GitHub 单文件限制，已直接保存在 Git 仓库中。
