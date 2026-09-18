# Sapiens2 Pose

## 模型文件

```text
sapiens2_pose_0.4b_fp32.onnx
```

该模型约 1.62 GB，使用 Git LFS 跟踪。克隆仓库后执行 `git lfs install` 和 `git lfs pull`，即可取得完整权重文件。

## 接口与配准用途

- 输入：`float32`，`[1, 3, 1024, 768]`
- 输出：308 通道关键点热图
- 人脸点：索引 70–307

`camera_keypoints.pose_solver: triplet_vote` 仅使用固定的 60 个鼻梁、鼻部和双侧眼眶/眼周刚性点；`overdetermined_svd` 使用全部有效人脸点，并按配置门限剔除离群对应后重新估计 SVD。Seg mask 用于约束配准点云；在当前集成流水线中，关键点深度从完整有序 ROI 点云查询，不能把 `detected_face.png` 或 Seg 定位图当作关键点结果。

集成配置、验证命令和输出位置见[上级 README](../README.md)。独立的 `sapiens_pose.exe` 调用示例见 [`C++/SAPIENS2.md`](../../../C++/SAPIENS2.md)。
