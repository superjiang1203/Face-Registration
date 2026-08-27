[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [switch]$RunCamera,
    [switch]$SkipVcamera,
    [ValidateSet("orbbec", "vcamera")]
    [string]$CameraBackend = "orbbec",
    [string]$CameraSn = "",
    [string]$CameraIp = "",
    [ValidateSet("on", "off")]
    [string]$LaserAuto = "off",
    [int]$LaserPower = 25,
    [ValidateRange(1, 64)]
    [int]$Threads = 8,
    [switch]$ForceUnpack
)

$ErrorActionPreference = "Stop"

# 1. Resolve paths relative to this script so it works from any current directory.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Packages = Join-Path $ProjectRoot "thirdparty/scripts/windows/packages.ps1"
$BuildDir = Join-Path $ProjectRoot "build"
$VcameraBuild = Join-Path $ProjectRoot "thirdparty/scripts/windows/build_vcamera.ps1"

# 2. Unpack only when a required dependency is absent, unless -ForceUnpack is used.
$RequiredFiles = @(
    (Join-Path $ProjectRoot "thirdparty/opencv/x64/vc17/lib/OpenCVConfig.cmake"),
    (Join-Path $ProjectRoot "thirdparty/opencv/x64/vc17/bin/opencv_core4120.dll"),
    (Join-Path $ProjectRoot "thirdparty/open3d/CMake/Open3DConfig.cmake"),
    (Join-Path $ProjectRoot "thirdparty/vtk/lib/cmake/vtk-9.4/vtk-config.cmake"),
    (Join-Path $ProjectRoot "thirdparty/vtk/bin/vtkRenderingOpenGL2-9.4.dll"),
    (Join-Path $ProjectRoot "thirdparty/OrbbecSDK/lib/OrbbecSDKConfig.cmake"),
    (Join-Path $ProjectRoot "thirdparty/onnxruntime-win-x64-gpu-1.24.4/lib/onnxruntime.dll"),
    (Join-Path $ProjectRoot "thirdparty/src/VcameraSDK-26.1.10/cpp/example/CMakeLists.txt")
)
$DependenciesReady = ($RequiredFiles | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0
if ($ForceUnpack -or -not $DependenciesReady) {
    & powershell -ExecutionPolicy Bypass -File $Packages -Action unpack
    if ($LASTEXITCODE -ne 0) { throw "Dependency unpack failed" }
} else {
    Write-Host "Third-party dependencies are already unpacked; skipping extraction."
}

# 3. Normalize the vendor binary SDK and compile its C++ examples. This also
# verifies that headers, import library and runtime DLLs form a usable package.
if (-not $SkipVcamera) {
    & powershell -ExecutionPolicy Bypass -File $VcameraBuild -Config $Config
    if ($LASTEXITCODE -ne 0) { throw "Vcamera SDK build failed" }
}

# 4. Generate a Visual Studio 2022 x64 build tree from the root CMakeLists.txt.
cmake -S $ProjectRoot -B $BuildDir -G "Visual Studio 17 2022" -A x64
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
# 5. Compile the selected Release/Debug configuration using parallel jobs.
cmake --build $BuildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "C++ build failed" }

$BinDir = Join-Path $BuildDir "C++/$Config"

if ($RunCamera) {
    # 6. Optional hardware path: camera -> physical gates -> nearest-first
    # candidate lock -> keypoint/geometry initialization -> model registration.
    $CameraApp = Join-Path $BinDir "face_camera_pipeline.exe"
    $CameraArgs = @("--camera-backend", $CameraBackend)
    if ($Threads -gt 0) { $CameraArgs += @("--threads", $Threads) }
    if ($CameraSn) { $CameraArgs += @("--camera-sn", $CameraSn) }
    if ($CameraIp) { $CameraArgs += @("--camera-ip", $CameraIp) }
    if ($CameraBackend -eq "vcamera") {
        $CameraArgs += @("--laser-auto", $LaserAuto, "--laser-power", $LaserPower)
    }
    $CameraArgs += @(
        (Join-Path $ProjectRoot "models/face_detection/yolo_face/yolov12n-face.onnx"),
        (Join-Path $ProjectRoot "models/face_keypoints/hrnet/hrnetv2_w18_wflw_256x256_heatmap.onnx"),
        (Join-Path $ProjectRoot "output")
    )
    Push-Location $ProjectRoot
    try {
        & $CameraApp @CameraArgs
        $CameraExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($CameraExitCode -ne 0) { throw "Depth-camera pipeline failed" }
}

Write-Host "Build completed successfully." -ForegroundColor Green
