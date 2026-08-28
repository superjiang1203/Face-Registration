# Sapiens2 ONNX C++ 调用

作用：用 Orbbec 拍摄并运行 Sapiens2 Seg。

```powershell
.\build\C++\Release\sapiens_seg.exe --model ".\models\face_segmentation\sapiens2_seg\sapiens2_seg_0.4b_fp32.onnx" --camera-sn CP2AB53000CK --min-depth-mm 500 --max-depth-mm 600 --warmup 15 --no-hrnet --output ".\output\sapiens2_pose_live\capture"
```

作用：运行 Sapiens2 Pose 并结合 Seg mask 筛除遮挡点。

```powershell
.\build\C++\Release\sapiens_pose.exe ".\models\face_keypoints\sapiens2_pose\sapiens2_pose_0.4b_fp32.onnx" ".\output\sapiens2_pose_live\capture\color.png" ".\output\sapiens2_pose_live\capture\target_face_mask.png" ".\output\sapiens2_pose_live\pose" 0.25 0.20 5
```
