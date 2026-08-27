# VTK 9.4.2 Build Script for Windows with Qt 6.8.3
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

$ProjectRoot = Resolve-ProjectRoot -StartDir $PSScriptRoot
$InstallDir = "$ProjectRoot/thirdparty/vtk"
$SourceDir = "$ProjectRoot/thirdparty/src/VTK-9.4.2"
$BuildDir = "$ProjectRoot/thirdparty/build/vtk-9.4.2"

$QtDir = "C:/Qt/6.8.3/msvc2022_64/lib/cmake/Qt6"

# Clean previous build if requested
if (Test-Path -Path $BuildDir) {
    Write-Host "Cleaning previous build directory: $BuildDir" -ForegroundColor Yellow
    Remove-Item -Path $BuildDir -Recurse -Force
}

Write-Host "Configuring VTK..." -ForegroundColor Cyan

# Create directories if they don't exist
if (-not (Test-Path -Path $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

cmake -S "$SourceDir" -B "$BuildDir" -G "Visual Studio 17 2022" `
    -DCMAKE_CONFIGURATION_TYPES="Debug;Release" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DBUILD_SHARED_LIBS=ON `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL `
    -DCMAKE_DEBUG_POSTFIX="d" `
    -DVTK_GROUP_ENABLE_Qt=YES `
    -DVTK_QT_VERSION=6 `
    -DQt6_DIR="$QtDir" `
    -DVTK_MODULE_USE_EXTERNAL_VTK_qt=ON `
    -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES `
    -DVTK_MODULE_ENABLE_VTK_RenderingQt=YES `
    -DVTK_MODULE_ENABLE_VTK_ViewsQt=YES `
    -DVTK_MODULE_ENABLE_VTK_CommonExecutionModel=YES `
    -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2=YES `
    -DVTK_MODULE_ENABLE_VTK_RenderingCore=YES `
    -DVTK_MODULE_ENABLE_VTK_RenderingContextOpenGL2=YES `
    -DVTK_MODULE_ENABLE_VTK_RenderingAnnotation=YES `
    -DVTK_MODULE_ENABLE_VTK_InteractionStyle=YES `
    -DVTK_MODULE_ENABLE_VTK_InteractionWidgets=YES `
    -DVTK_MODULE_ENABLE_VTK_CommonCore=YES `
    -DVTK_MODULE_ENABLE_VTK_CommonDataModel=YES `
    -DVTK_MODULE_ENABLE_VTK_FiltersCore=YES `
    -DVTK_MODULE_ENABLE_VTK_FiltersGeneral=YES `
    -DVTK_MODULE_ENABLE_VTK_FiltersExtraction=YES `
    -DVTK_MODULE_ENABLE_VTK_FiltersSources=YES `
    -DVTK_MODULE_ENABLE_VTK_FiltersHybrid=YES `
    -DVTK_MODULE_ENABLE_VTK_IOImage=YES `
    -DVTK_MODULE_ENABLE_VTK_ImagingCore=YES `
    -DVTK_MODULE_ENABLE_VTK_ImagingGeneral=YES `
    -DVTK_MODULE_ENABLE_VTK_IOGeometry=YES `
    -DVTK_MODULE_ENABLE_VTK_IOLegacy=YES `
    -DVTK_MODULE_ENABLE_VTK_IOPLY=YES `
    -DVTK_MODULE_ENABLE_VTK_ViewsCore=YES `
    -DVTK_MODULE_ENABLE_VTK_RenderingImage=YES `
    -DVTK_MODULE_ENABLE_VTK_RenderingVolumeOpenGL2=YES `
    -DVTK_MODULE_ENABLE_VTK_InteractionImage=YES `
    -DVTK_BUILD_TESTING=OFF `
    -DVTK_BUILD_EXAMPLES=OFF

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit 1
}

# --- Build & Install Debug ---
Write-Host "Building VTK (Debug)... This may take a while." -ForegroundColor Cyan
cmake --build $BuildDir --config Debug --parallel 8
if ($LASTEXITCODE -ne 0) { Write-Error "VTK Debug build failed."; exit 1 }

Write-Host "Installing VTK (Debug)..." -ForegroundColor Cyan
cmake --install $BuildDir --config Debug
if ($LASTEXITCODE -ne 0) { Write-Error "VTK Debug install failed."; exit 1 }

# --- Build & Install Release ---
Write-Host "Building VTK (Release)... This may take a while." -ForegroundColor Cyan
cmake --build $BuildDir --config Release --parallel 8
if ($LASTEXITCODE -ne 0) { Write-Error "VTK Release build failed."; exit 1 }

Write-Host "Installing VTK (Release)..." -ForegroundColor Cyan
cmake --install $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Error "VTK Release install failed."; exit 1 }

Write-Host "Success! VTK (Debug & Release) installed to: $InstallDir" -ForegroundColor Green
Write-Host "Please update your CMakeUserPresets.json to set VTK_DIR to: $InstallDir/lib/cmake/vtk-9.4" -ForegroundColor Yellow
