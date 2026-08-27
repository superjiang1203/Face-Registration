# OpenCV 4.x Build Script for Windows with Qt 6.8.3
# Run this script in PowerShell

if ($args -contains "-h" -or $args -contains "--help") {
    Write-Host "Usage: .\build_opencv.ps1"
    Write-Host ""
    Write-Host "Environment:"
    Write-Host "  Qt6_DIR  (optional) Enable Qt backend for HighGUI if provided; must contain Qt6Config.cmake"
    Write-Host "          Example: setx Qt6_DIR C:\Qt\6.8.3\msvc2022_64\lib\cmake\Qt6"
    Write-Host "  (If Qt6_DIR is not set, OpenCV will be built with WITH_QT=OFF.)"
    exit 0
}

# Determine Project Root (robust to nested script directories)
function Resolve-ProjectRoot {
    param([string]$StartDir)

    $dir = Resolve-Path -LiteralPath $StartDir
    while ($true) {
        $cand = $dir.Path
        if ((Test-Path -LiteralPath (Join-Path $cand "CMakeLists.txt")) -and (Test-Path -LiteralPath (Join-Path $cand "thirdparty"))) {
            return ($cand -replace '\\', '/')
        }
        $parent = Split-Path -Parent $cand
        if (-not $parent -or $parent -eq $cand) {
            break
        }
        $dir = Resolve-Path -LiteralPath $parent
    }

    throw ("Cannot locate project root from: " + $StartDir)
}

$ProjectRoot = Resolve-ProjectRoot -StartDir $PSScriptRoot
$InstallDir = "$ProjectRoot/thirdparty/opencv"

$SourceDir = "$ProjectRoot/thirdparty/src/opencv-4.12.0"
$ContribDir = "$ProjectRoot/thirdparty/src/opencv_contrib-4.12.0/modules"
$BuildDir = "$ProjectRoot/thirdparty/build/opencv-4.12.0"

$QtDir = $env:Qt6_DIR
$WithQt = "OFF"
$QtArgs = @()
if ($QtDir) {
    $QtConfig = Join-Path $QtDir "Qt6Config.cmake"
    if (-not (Test-Path -LiteralPath $QtConfig)) {
        Write-Error "Invalid Qt6_DIR: missing Qt6Config.cmake under: $QtDir"
        exit 1
    }
    $WithQt = "ON"
    $QtArgs += "-DQt6_DIR=$QtDir"
} else {
    Write-Host "Qt disabled for OpenCV (set Qt6_DIR to enable)." -ForegroundColor Yellow
}

# Clean previous build if requested
if (Test-Path -Path $BuildDir) {
    Write-Host "Cleaning previous build directory: $BuildDir" -ForegroundColor Yellow
    Remove-Item -Path $BuildDir -Recurse -Force
}

Write-Host "Checking Source Directory..." -ForegroundColor Cyan
if (-not (Test-Path -Path $SourceDir)) {
    Write-Error "OpenCV Source directory not found at: $SourceDir"
    exit 1
}

# Check for Contrib modules (Required per user request)
Write-Host "Checking Contrib Directory..." -ForegroundColor Cyan
if (-not (Test-Path -Path $ContribDir)) {
    Write-Error "OpenCV Contrib modules not found at: $ContribDir"
    Write-Error "Please download opencv_contrib matching your OpenCV version (4.12.0) and extract it to D:/third_project/opencv_contrib"
    Write-Error "Download Link: https://github.com/opencv/opencv_contrib/tags"
    exit 1
}

Write-Host "Configuring OpenCV with Contrib..." -ForegroundColor Cyan

# Create directories if they don't exist
if (-not (Test-Path -Path $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

# Configure CMake
# Key Options:
# - BUILD_opencv_world=ON: Builds a single huge DLL (easier to manage) instead of many small ones
# - WITH_QT=ON: Enables Qt highgui backend (cv::imshow uses Qt)
# - CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL: Matches your VTK/ITK build
cmake -S "$SourceDir" -B "$BuildDir" -G "Visual Studio 17 2022" `
    -DCMAKE_CONFIGURATION_TYPES="Debug;Release" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DBUILD_SHARED_LIBS=ON `
    # Note: BUILD_opencv_world=ON often causes build failures with contrib modules (especially stereo)
    # Disabling it ensures better compatibility and stability.
    -DBUILD_opencv_world=OFF `
    -DOPENCV_EXTRA_MODULES_PATH="$ContribDir" `
    -DOPENCV_ENABLE_NONFREE=ON `
    -DWITH_IPP=ON `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL `
    -DCMAKE_DEBUG_POSTFIX="d" `
    -DWITH_QT=$WithQt `
    -DWITH_OPENGL=ON `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_TESTS=OFF `
    -DBUILD_PERF_TESTS=OFF `
    -DBUILD_JAVA=OFF `
    -DBUILD_PYTHON=OFF `
    -DWITH_VTK=OFF `
    -DWITH_LAPACK=OFF `
    -DWITH_ADE=OFF `
    @QtArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit 1
}

# --- Build & Install Debug ---
Write-Host "Building OpenCV (Debug)..." -ForegroundColor Cyan
cmake --build $BuildDir --config Debug --parallel 8
if ($LASTEXITCODE -ne 0) { Write-Error "OpenCV Debug build failed."; exit 1 }

Write-Host "Installing OpenCV (Debug)..." -ForegroundColor Cyan
cmake --install $BuildDir --config Debug
if ($LASTEXITCODE -ne 0) { Write-Error "OpenCV Debug install failed."; exit 1 }

# --- Build & Install Release ---
Write-Host "Building OpenCV (Release)..." -ForegroundColor Cyan
cmake --build $BuildDir --config Release --parallel 8
if ($LASTEXITCODE -ne 0) { Write-Error "OpenCV Release build failed."; exit 1 }

Write-Host "Installing OpenCV (Release)..." -ForegroundColor Cyan
cmake --install $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Error "OpenCV Release install failed."; exit 1 }

Write-Host "Success! OpenCV installed to: $InstallDir" -ForegroundColor Green
Write-Host "Please update your CMakeUserPresets.json to set OpenCV_DIR to: $InstallDir" -ForegroundColor Yellow
