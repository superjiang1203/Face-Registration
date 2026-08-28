# Face Registration C++

## 当前算法流程（适用于所有分支）

运行时由 `C++/config/runtime.yml` 选择定位器、关键点模型和粗配准求解器，不再按旧的 1.1～1.3 分开理解：

```text
STL/模型点云预处理完成（不计入正式定位计时）
→ 采集 RGB、深度和有序点云（自动分支采集配置的 N 帧）
→ 目标定位：YOLO Face / Sapiens2 Seg / --manual-roi
→ 多帧按 detection score 或 mask 质量选 Top-1
→ 仅对目标运行 HRNet 或 Sapiens2 Pose
→ 关键点结合深度生成 3D 点，过滤低置信度、无效深度和遮挡点
→ triplet_vote 或 overdetermined_svd 求粗变换
→ 粗匹配失败时 FPFH + RANSAC，再执行 ICP 精配准
→ 结果写入 output/时间戳/，并输出各阶段及总耗时
```

可用组合如下：

| `target_locator` | `keypoint_model` | 流程 |
|---|---|---|
| `face_detection` | `hrnet` | YOLO Top-1 → HRNet 98 点 → SVD/ICP |
| `face_detection` | `sapiens_pose` | YOLO Top-1 → Sapiens2 Pose 脸部点 → SVD/ICP |
| `sapiens_seg` | `hrnet` | Seg mask → mask 内 HRNet 点 → SVD/ICP |
| `sapiens_seg` | `sapiens_pose` | Seg mask → mask 内、未遮挡的 Sapiens2 脸部点 → SVD/ICP |

使用 `--manual-roi` 时，ROI 窗口结束后自动保存 `output/时间戳/roi/manual_roi.txt`，并直接进入相同的关键点、粗配准和 ICP 流程，不再运行 YOLO/Seg 定位。

旧版 1.1～1.3 说明仅作历史记录，以上统一流程和 `runtime.yml` 是当前实现的准则。

纯 C++17 人脸/头模点云配准工程。推理统一使用 ONNX Runtime；图像处理使用 OpenCV；点云粗配准和精配准使用 Open3D；相机支持 Orbbec，Windows 额外支持 Vcamera。

## 1. 算法流程

### 1.1 Face Detection + Face Keypoints 配准

```text
采集 N 帧 RGB、深度和有序点云
→ N 帧并行执行 YOLO Face Detection
→ 深度、尺寸、点数和点密度门控
→ 汇总有效候选并按 Detection score 排序
→ 选择全局 Top-1
→ 只对 Top-1 执行 HRNet-WFLW 关键点检测
→ RGB 关键点结合深度生成同名三维点
→ triplet_vote 或 overdetermined_svd 求初始刚体变换
→ 多尺度 ICP
→ 关键点位姿未通过门控时回退到 FPFH + RANSAC + ICP
```

### 1.2 手动 ROI 配准

```text
采集一帧对齐的 RGB、深度和有序点云
→ 弹出 RGB 窗口并交互选择 ROI
→ 写入 output/时间戳/roi/manual_roi.txt
→ 裁剪 ROI 三维点云
→ 按 runtime.yml 选择 HRNet 或 Sapiens Pose
→ 相机/STL 同名或同索引关键点求粗变换
→ 多尺度 ICP
→ 关键点不足时回退 FPFH + RANSAC + ICP
```

该模式只用手动框替代目标定位，关键点、粗配准和 ICP 仍属于同一条流水线。

### 1.3 Sapiens2 Seg + Pose

```text
Orbbec 采集并对齐 RGB 与深度
→ Sapiens2 Seg 生成人脸语义 mask
→ 深度范围定位目标头部
→ Sapiens2 Pose 输出 308 个关键点
→ 选择 Sapiens2 脸部关键点
→ 使用 Pose score 和 Seg mask 邻域覆盖率筛除遮挡点
```

该流程不使用 HRNet 98 点。

## 2. 模型目录

```text
models/
├── face_detection/yolo_face/yolov12n-face.onnx
├── face_keypoints/hrnet/hrnetv2_w18_wflw_256x256_heatmap.onnx
├── face_keypoints/sapiens2_pose/sapiens2_pose_0.4b_fp32.onnx
└── face_segmentation/sapiens2_seg/
    ├── sapiens2_seg_0.4b_fp32.onnx
    └── sapiens2_seg_0.4b_fp32.onnx.data
```

## 3. Windows 编译

以下命令均在仓库根目录执行，每个代码块是一条可直接复制的完整命令。

作用：确认当前目录是工程根目录。

```powershell
Test-Path .\CMakeLists.txt
```

作用：安装 CMake。

```powershell
winget install --id Kitware.CMake --exact --accept-source-agreements --accept-package-agreements
```

作用：安装 Visual Studio 2022 C++ Build Tools。

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-source-agreements --accept-package-agreements
```

作用：解压第三方 C++ 依赖、配置 CMake 并编译 Release。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\C++\scripts\windows\build_windows.ps1
```

作用：使用纯 CMake 命令重新配置 Visual Studio 2022 x64 工程。

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

作用：使用纯 CMake 命令编译 Release。

```powershell
cmake --build build --config Release --parallel
```

Release 可执行文件位于 `build/C++/Release/`：

```text
face_camera_pipeline.exe
face_registration_cli.exe
manual_roi.exe
sapiens_seg.exe
sapiens_pose.exe
```

## 4. 调用算法 1.1

配置文件为 `C++/config/runtime.yml`：

```yaml
runtime:
  onnx_provider: cuda

pipeline:
  target_locator: face_detection
  keypoint_model: hrnet
  detection_frames: 4

camera_keypoints:
  pose_solver: triplet_vote
```

主配准流水线支持四种组合。命令行参数会临时覆盖 YAML：

正常运行只需要配置文件；模型路径、输出目录和算法分支均从 YAML 读取：

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml"
```

手动 ROI 只增加一个开关：

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --manual-roi
```

相机后端、SN、分辨率和 FPS 默认读取 `C++/config/camera.yml`。其余命令行参数只作为临时覆盖保留。

| 目标定位 | 关键点 | 参数组合 |
|---|---|---|
| YOLO Face Detection | HRNet-WFLW | `--target-locator face_detection --keypoint-provider hrnet` |
| YOLO Face Detection | Sapiens Pose | `--target-locator face_detection --keypoint-provider sapiens_pose` |
| Sapiens Seg | HRNet-WFLW | `--target-locator sapiens_seg --keypoint-provider hrnet` |
| Sapiens Seg | Sapiens Pose | `--target-locator sapiens_seg --keypoint-provider sapiens_pose` |

作用：使用 Face Detection 定位目标，使用 HRNet 关键点初始化 SVD/ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --target-locator face_detection --keypoint-provider hrnet --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK ".\models\face_detection\yolo_face\yolov12n-face.onnx" ".\models\face_keypoints\hrnet\hrnetv2_w18_wflw_256x256_heatmap.onnx" ".\output"
```

作用：使用 Face Detection 定位目标，使用 Sapiens Pose 稠密脸部点执行超定 SVD/ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --target-locator face_detection --keypoint-provider sapiens_pose --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK ".\models\face_detection\yolo_face\yolov12n-face.onnx" - ".\output"
```

作用：使用 Sapiens Seg 定位目标 mask，使用 HRNet 关键点初始化 SVD/ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --target-locator sapiens_seg --keypoint-provider hrnet --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK - ".\models\face_keypoints\hrnet\hrnetv2_w18_wflw_256x256_heatmap.onnx" ".\output"
```

作用：使用 Sapiens Seg 定位目标 mask，使用 Sapiens Pose 稠密脸部点执行超定 SVD/ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --target-locator sapiens_seg --keypoint-provider sapiens_pose --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK - - ".\output"
```

`pose_solver` 可设为 `triplet_vote` 或 `overdetermined_svd`，修改配置后不需要重新编译。

作用：枚举 Orbbec 相机，不执行推理。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --camera-backend orbbec --list-cameras
```

作用：采集 4 帧并行执行 Face Detection，选择全局 Top-1，然后执行 HRNet、SVD 和配准。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK ".\models\face_detection\yolo_face\yolov12n-face.onnx" ".\models\face_keypoints\hrnet\hrnetv2_w18_wflw_256x256_heatmap.onnx" ".\output"
```

作用：使用 Vcamera 执行相同流程。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --threads 4 --camera-backend vcamera --camera-sn 207000167813 --laser-auto off --laser-power 25 ".\models\face_detection\yolo_face\yolov12n-face.onnx" ".\models\face_keypoints\hrnet\hrnetv2_w18_wflw_256x256_heatmap.onnx" ".\output"
```

相机 SN 应替换为枚举得到的实际序列号。

## 5. 调用算法 1.2

作用：单独验证 ROI 窗口和 txt 输出，不执行配准。

```powershell
.\build\C++\Release\manual_roi.exe --camera-sn CP2AB53000CK --output ".\output\manual_roi.txt"
```

作用：在完整配准程序中弹窗选择 ROI，自动写入当前时间戳目录，然后按照 `runtime.yml` 的 `keypoint_model` 执行关键点、粗配准和 ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --manual-roi --camera-backend orbbec --camera-sn CP2AB53000CK - - ".\output"
```

窗口中拖动鼠标选择人脸，按 Enter 或空格确认，按 ESC 取消（C 可清除当前框）。确认后才开始总计时；默认先按 `camera.yml` 的分辨率和 FPS 采集 8 帧，再并发处理。每帧默认只执行 1 次 FPFH/RANSAC + ICP，最后优先按 fitness、再按 RMSE/P95 选择最佳结果。帧数和单帧全局尝试数分别由 `pipeline.manual_roi_frames`、`pipeline.manual_roi_global_attempts` 控制。

### Windows/Linux 算法脚本

`C++/scripts/windows` 和 `C++/scripts/linux` 均只保留以下三个同名入口：

- `run_face_detection`：Face Detection 定位，关键点模型读取配置。
- `run_face_keypoints`：弹出手动 ROI，单独验证配置指定的关键点分支并继续配准。
- `run_face_segmentation`：Sapiens Seg 定位，关键点模型读取配置。

Windows 示例：

```powershell
.\C++\scripts\windows\run_face_detection.ps1 -CameraSn CP2AB53000CK
.\C++\scripts\windows\run_face_keypoints.ps1 -CameraSn CP2AB53000CK
.\C++\scripts\windows\run_face_segmentation.ps1 -CameraSn CP2AB53000CK
```

Linux 示例：

```bash
CAMERA_SN=CP2AB53000CK bash C++/scripts/linux/run_face_detection.sh
CAMERA_SN=CP2AB53000CK bash C++/scripts/linux/run_face_keypoints.sh
CAMERA_SN=CP2AB53000CK bash C++/scripts/linux/run_face_segmentation.sh
```

## 6. 调用算法 1.3

算法 1.3 由两个纯 C++ 可执行程序顺序执行。

作用：从 Orbbec 拍摄 RGB/深度，运行 Sapiens2 Seg，在 500–600 mm 中定位目标并输出 RGB 和 `target_face_mask.png`。

```powershell
.\build\C++\Release\sapiens_seg.exe --model ".\models\face_segmentation\sapiens2_seg\sapiens2_seg_0.4b_fp32.onnx" --camera-sn CP2AB53000CK --min-depth-mm 500 --max-depth-mm 600 --warmup 15 --no-hrnet --output ".\output\sapiens2_pose_live\capture"
```

作用：读取上一条命令产生的 RGB 和 mask，运行 Sapiens2 Pose，并输出全部脸部关键点和未遮挡脸部关键点。

```powershell
.\build\C++\Release\sapiens_pose.exe ".\models\face_keypoints\sapiens2_pose\sapiens2_pose_0.4b_fp32.onnx" ".\output\sapiens2_pose_live\capture\color.png" ".\output\sapiens2_pose_live\capture\target_face_mask.png" ".\output\sapiens2_pose_live\pose" 0.25 0.20 5
```

三个末尾参数依次表示 Pose score 阈值、mask 邻域覆盖率阈值和邻域半径像素数。

## 7. 输出

算法 1.1 和 1.2 每次建立独立会话目录：

```text
output/YYYY-MM-DD_HH-MM-SS-ms/
├── face_detection/
├── face_keypoints_detection/
├── camera/
├── STL/
├── logs/
└── aligned_camera_face.ready
```

- `camera/aligned_camera_face.ply`：配准后的相机点云。
- `STL/camera_to_stl_transformation.txt`：相机坐标到 STL 坐标的 4×4 变换。
- `STL/pose_stl_to_camera.txt`：STL 坐标到相机坐标的 4×4 变换。
- `logs/registration_timing.txt`：输出完整墙钟阶段以及配准内部明细。自动分支可用“目标定位 + registration_batch_wall + pipeline_control_overhead”核对总时间；手动分支可用“capture_batch + registration_batch_wall + pipeline_control_overhead”核对总时间。`detail_batch_max_*` 是并发帧的单阶段最大值，属于 registration_batch_wall 内部，不能再次与总时间相加。

总耗时是墙钟时间，因此还包含相机采集、点云裁剪/滤波/降采样、法向估计、FPFH、RANSAC 和质量评估；这些过程按输出约定不单独打印，所以分项之和可以小于总耗时。手动 ROI 的交互等待不计入总耗时。
- `registration_total_excluding_initial_model_reconstruction`：不包含首次 STL 表面点云重建的总时间。

算法 1.3 主要输出：

```text
output/sapiens2_pose_live/
├── capture/
│   ├── color.png
│   ├── target_face_mask.png
│   └── overlay.png
└── pose/
    ├── sapiens2_pose_keypoints_onnx.json
    └── sapiens2_visible_face_keypoints.json
```

## 8. C++ 库接口

构建后可链接以下静态库：

```text
face_camera
face_detection
face_segmentation
face_pose
face_registration
```

对应公共头文件位于：

```text
C++/hpp/camera/
C++/hpp/detection/
C++/hpp/segmentation/
C++/hpp/pose/
C++/hpp/registration/
```

作用：安装 C++ 静态库、可执行程序、公共头文件和配置文件。

```powershell
cmake --install build --config Release --prefix ".\install"
```

## 9. Windows 到 Linux 迁移

作用：在 Windows 工程根目录创建迁移压缩包。

```powershell
tar.exe -czf ..\face_registration-linux-transfer.tar.gz --exclude=.git --exclude=build --exclude=thirdparty/build --exclude=data --exclude=output .
```

作用：在 Linux 中解压工程。

```bash
mkdir -p face_registration && tar -xzf face_registration-linux-transfer.tar.gz -C face_registration
```

作用：进入工程并确认根目录。

```bash
cd face_registration && test -f CMakeLists.txt && echo "project root OK"
```

## 10. Linux 编译

作用：安装 Ubuntu 22.04 编译工具和运行库。

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build pkg-config unzip libgl1 libglib2.0-0
```

作用：为依赖解包脚本添加执行权限。

```bash
chmod +x C++/scripts/linux/*.sh thirdparty/scripts/linux/*.sh
```

作用：解压 Linux 第三方 C++ 依赖并编译 Release。

```bash
C++/scripts/linux/build_linux.sh
```

作用：不使用构建脚本，直接用 CMake 和 Ninja 配置 Release。

```bash
cmake -S . -B build/linux-Release -G Ninja -DCMAKE_BUILD_TYPE=Release
```

作用：编译 Linux Release。

```bash
cmake --build build/linux-Release --parallel
```

作用：枚举 Linux 下的 Orbbec 相机。

```bash
build/linux-Release/C++/face_camera_pipeline --camera-backend orbbec --list-cameras
```

Vcamera SDK 当前只有 Windows 库，Linux 支持 Orbbec 和离线配准。
