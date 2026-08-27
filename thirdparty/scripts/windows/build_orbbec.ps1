# OrbbecSDK v2 Build Script for Windows
# Run this script in PowerShell

# Add CMake to PATH (VS 2022 Community)
# $env:Path = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$env:Path"

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
$InstallDir = "$ProjectRoot/thirdparty/OrbbecSDK"

$SourceDir = "$ProjectRoot/thirdparty/src/OrbbecSDK_v2-2.7.2-rc"
$BuildDir = "$ProjectRoot/thirdparty/build/OrbbecSDK_v2-2.7.2-rc"

# Clean previous build if requested
if (Test-Path -Path $BuildDir) {
    Write-Host "Cleaning previous build directory: $BuildDir" -ForegroundColor Yellow
    Remove-Item -Path $BuildDir -Recurse -Force
}

Write-Host "Configuring OrbbecSDK..." -ForegroundColor Cyan

# Create directories if they don't exist
if (-not (Test-Path -Path $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

# Configure CMake
# Note: For Visual Studio generator, CMAKE_BUILD_TYPE is ignored at configuration time.
# We set CMAKE_CONFIGURATION_TYPES to ensure both Debug and Release are available.
# CRITICAL: Set SPDLOG_BUILD_SHARED=OFF to force spdlog to be static INSIDE the OrbbecSDK DLL.
# This prevents spdlog.dll conflicts with the main project.
cmake -S "$SourceDir" -B "$BuildDir" -G "Visual Studio 17 2022" `
    -DCMAKE_CONFIGURATION_TYPES="Debug;Release" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DBUILD_SHARED_LIBS=OFF `
    -DSPDLOG_BUILD_SHARED=OFF `
    -DOB_BUILD_EXAMPLES=OFF `
    -DOB_BUILD_TESTS=OFF `
    -DOB_BUILD_DOCS=OFF `
    -DOB_BUILD_TOOLS=OFF `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit 1
}

# --- Build & Install Debug ---
Write-Host "Building OrbbecSDK (Debug)... This may take a while." -ForegroundColor Cyan
cmake --build $BuildDir --config Debug --parallel 8
if ($LASTEXITCODE -ne 0) { Write-Error "OrbbecSDK Debug build failed."; exit 1 }

Write-Host "Installing OrbbecSDK (Debug)..." -ForegroundColor Cyan
cmake --install $BuildDir --config Debug
if ($LASTEXITCODE -ne 0) { Write-Error "OrbbecSDK Debug install failed."; exit 1 }

# --- Build & Install Release ---
Write-Host "Building OrbbecSDK (Release)... This may take a while." -ForegroundColor Cyan
cmake --build $BuildDir --config Release --parallel 8
if ($LASTEXITCODE -ne 0) { Write-Error "OrbbecSDK Release build failed."; exit 1 }

Write-Host "Installing OrbbecSDK (Release)..." -ForegroundColor Cyan
cmake --install $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Error "OrbbecSDK Release install failed."; exit 1 }

Write-Host "Success! OrbbecSDK (Debug & Release) installed to: $InstallDir" -ForegroundColor Green
Write-Host "Please update your CMakeUserPresets.json to set OrbbecSDK_DIR to: $InstallDir" -ForegroundColor Yellow
