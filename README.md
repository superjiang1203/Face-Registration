# Face Registration C++

## 1. 算法流程（适用于所有分支）

运行时由 `C++/config/runtime.yml` 选择定位器、关键点模型和粗配准求解器：

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

以上统一流程和 `runtime.yml` 是当前实现的准则。

纯 C++17 人脸/头模点云配准工程。推理统一使用 ONNX Runtime；图像处理使用 OpenCV；点云粗配准和精配准使用 Open3D；相机支持 Orbbec，Windows 额外支持 Vcamera。

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

确认当前目录是工程根目录。

```powershell
Test-Path .\CMakeLists.txt
```

安装 CMake。

```powershell
winget install --id Kitware.CMake --exact --accept-source-agreements --accept-package-agreements
```

安装 Visual Studio 2022 C++ Build Tools。

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-source-agreements --accept-package-agreements
```

解压第三方 C++ 依赖、配置 CMake 并编译 Release。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\C++\scripts\windows\build_windows.ps1
```

使用纯 CMake 命令重新配置 Visual Studio 2022 x64 工程。

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

使用纯 CMake 命令编译 Release。

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

## 4. 调用算法

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

使用 Face Detection 定位目标，使用 HRNet 关键点初始化 SVD/ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --target-locator face_detection --keypoint-provider hrnet --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK ".\models\face_detection\yolo_face\yolov12n-face.onnx" ".\models\face_keypoints\hrnet\hrnetv2_w18_wflw_256x256_heatmap.onnx" ".\output"
```

使用 Face Detection 定位目标，使用 Sapiens Pose 稠密脸部点执行超定 SVD/ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --target-locator face_detection --keypoint-provider sapiens_pose --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK ".\models\face_detection\yolo_face\yolov12n-face.onnx" - ".\output"
```

使用 Sapiens Seg 定位目标 mask，使用 HRNet 关键点初始化 SVD/ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --target-locator sapiens_seg --keypoint-provider hrnet --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK - ".\models\face_keypoints\hrnet\hrnetv2_w18_wflw_256x256_heatmap.onnx" ".\output"
```

使用 Sapiens Seg 定位目标 mask，使用 Sapiens Pose 稠密脸部点执行超定 SVD/ICP。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --config ".\C++\config\runtime.yml" --target-locator sapiens_seg --keypoint-provider sapiens_pose --threads 4 --camera-backend orbbec --camera-sn CP2AB53000CK - - ".\output"
```

`pose_solver` 可设为 `triplet_vote` 或 `overdetermined_svd`，修改配置后不需要重新编译。

枚举 Orbbec 相机，不执行推理。

```powershell
.\build\C++\Release\face_camera_pipeline.exe --camera-backend orbbec --list-cameras
```

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

## 5. C++ 库接口

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

安装 C++ 静态库、可执行程序、公共头文件和配置文件。

```powershell
cmake --install build --config Release --prefix ".\install"
```

## 6. Windows 到 Linux 迁移

在 Windows 工程根目录创建迁移压缩包。

```powershell
tar.exe -czf ..\face_registration-linux-transfer.tar.gz --exclude=.git --exclude=build --exclude=thirdparty/build --exclude=data --exclude=output .
```

在 Linux 中解压工程。

```bash
mkdir -p face_registration && tar -xzf face_registration-linux-transfer.tar.gz -C face_registration
```

进入工程并确认根目录。

```bash
cd face_registration && test -f CMakeLists.txt && echo "project root OK"
```

## 7. Linux 编译

安装 Ubuntu 22.04 编译工具和运行库。

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build pkg-config unzip libgl1 libglib2.0-0
```

为依赖解包脚本添加执行权限。

```bash
chmod +x C++/scripts/linux/*.sh thirdparty/scripts/linux/*.sh
```

解压 Linux 第三方 C++ 依赖并编译 Release。

```bash
C++/scripts/linux/build_linux.sh
```

不使用构建脚本，直接用 CMake 和 Ninja 配置 Release。

```bash
cmake -S . -B build/linux-Release -G Ninja -DCMAKE_BUILD_TYPE=Release
```

编译 Linux Release。

```bash
cmake --build build/linux-Release --parallel
``` 