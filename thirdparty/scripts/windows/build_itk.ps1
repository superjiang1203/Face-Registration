# ITK 5.4.5 Build Script for Windows (Linked with VTK 9.4.2)
# Run this script in PowerShell

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

# 1. PLEASE DOWNLOAD ITK SOURCE CODE FIRST
# Download from: https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.4.5/InsightToolkit-5.4.5.zip
# Extract to: <project_root>/thirdparty/src/InsightToolkit-5.4.5
# (Or update $SourceDir below to match your path)

$ProjectRoot = Resolve-ProjectRoot -StartDir $PSScriptRoot
$InstallDir = "$ProjectRoot/thirdparty/itk"

$SourceDir = "$ProjectRoot/thirdparty/src/InsightToolkit-5.4.5"
$BuildDir = "$ProjectRoot/thirdparty/build/itk-5.4.5"

# Point to your manually built VTK
$VtkDir = "$ProjectRoot/thirdparty/vtk/lib/cmake/vtk-9.4"

# Clean previous build if requested
if (Test-Path -Path $BuildDir) {
    Write-Host "Cleaning previous build directory: $BuildDir" -ForegroundColor Yellow
    Remove-Item -Path $BuildDir -Recurse -Force
}

Write-Host "Checking Source Directory..." -ForegroundColor Cyan
if (-not (Test-Path -Path $SourceDir)) {
    Write-Error "ITK Source directory not found at: $SourceDir"
    Write-Error "Please download ITK and extract it there, or update `$SourceDir in this script."
    exit 1
}

Write-Host "Configuring ITK..." -ForegroundColor Cyan

# Create directories if they don't exist
if (-not (Test-Path -Path $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

# Configure CMake
# Key Options:
# - BUILD_SHARED_LIBS=ON: Must match VTK/Qt
# - Module_ITKVtkGlue=ON: Enables ITK->VTK bridge (ImageToVTKImageFilter etc.)
# - VTK_DIR: Points to your custom VTK build
cmake -S "$SourceDir" -B "$BuildDir" -G "Visual Studio 17 2022" `
    -DCMAKE_CONFIGURATION_TYPES="Debug;Release" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DBUILD_SHARED_LIBS=ON `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL `
    -DCMAKE_DEBUG_POSTFIX="d" `
    -DITK_BUILD_DEFAULT_MODULES=ON `
    -DModule_ITKVtkGlue=ON `
    -DVTK_DIR="$VtkDir" `
    -DITK_SKIP_PATH_LENGTH_CHECKS=ON `
    -DBUILD_TESTING=OFF

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit 1
}

# --- Build & Install Debug ---
Write-Host "Building ITK (Debug)..." -ForegroundColor Cyan
cmake --build $BuildDir --config Debug --parallel 8
if ($LASTEXITCODE -ne 0) { Write-Error "ITK Debug build failed."; exit 1 }

Write-Host "Installing ITK (Debug)..." -ForegroundColor Cyan
cmake --install $BuildDir --config Debug
if ($LASTEXITCODE -ne 0) { Write-Error "ITK Debug install failed."; exit 1 }

# --- Build & Install Release ---
Write-Host "Building ITK (Release)..." -ForegroundColor Cyan
cmake --build $BuildDir --config Release --parallel 8
if ($LASTEXITCODE -ne 0) { Write-Error "ITK Release build failed."; exit 1 }

Write-Host "Installing ITK (Release)..." -ForegroundColor Cyan
cmake --install $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Error "ITK Release install failed."; exit 1 }

# --- Post-Install: Sanitize ITKVtkGlue.cmake ---
# Remove hardcoded VTK_DIR to avoid absolute paths on specific machines.
# The consuming project (neuro_navi) will provide VTK_DIR via CMakePresets.
$GlueConfigFile = "$InstallDir/lib/cmake/ITK-5.4/Modules/ITKVtkGlue.cmake"
if (Test-Path $GlueConfigFile) {
    Write-Host "Sanitizing ITKVtkGlue.cmake (removing hardcoded VTK_DIR)..." -ForegroundColor Cyan
    $Content = Get-Content $GlueConfigFile
    # Remove lines starting with set(VTK_DIR or similar
    $NewContent = $Content | Where-Object { $_ -notmatch '^\s*set\(\s*VTK_DIR\s+' }
    $NewContent | Set-Content $GlueConfigFile
    Write-Host "Removed hardcoded VTK_DIR from $GlueConfigFile" -ForegroundColor Green
}

Write-Host "Success! ITK installed to: $InstallDir" -ForegroundColor Green
Write-Host "Please update your CMakeUserPresets.json to set ITK_DIR to: $InstallDir/lib/cmake/ITK-5.4" -ForegroundColor Yellow
