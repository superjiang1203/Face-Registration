# Open3D Build Script for Windows (C++ library only)
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
$InstallDir = "$ProjectRoot/thirdparty/open3d"
$BuildDir = "$ProjectRoot/thirdparty/build/open3d"

$SourceDir = "$ProjectRoot/thirdparty/src/Open3D-main"
$SourceZip = "$ProjectRoot/thirdparty/src/Open3D-main.zip"

if (-not (Test-Path -Path $SourceDir)) {
    if (Test-Path -Path $SourceZip) {
        Write-Host "Extracting Open3D sources from: $SourceZip" -ForegroundColor Cyan
        Expand-Archive -Path $SourceZip -DestinationPath "$ProjectRoot/thirdparty/src" -Force
    }

    if (-not (Test-Path -Path $SourceDir)) {
        $Candidate = Get-ChildItem -Path "$ProjectRoot/thirdparty/src" -Directory `
            | Where-Object { $_.Name -like "Open3D-main*" } `
            | Sort-Object LastWriteTime -Descending `
            | Select-Object -First 1
        if ($null -ne $Candidate) {
            $SourceDir = $Candidate.FullName -replace '\\', '/'
        }
    }
}

Write-Host "Checking Source Directory..." -ForegroundColor Cyan
if (-not (Test-Path -Path $SourceDir)) {
    Write-Error "Open3D source directory not found at: $SourceDir"
    Write-Error "Please extract Open3D to: $ProjectRoot/thirdparty/src/Open3D-main (or keep Open3D-main.zip there)."
    exit 1
}

$EigenCmake = "$SourceDir/3rdparty/eigen/eigen.cmake"
if (Test-Path -Path $EigenCmake) {
    $EigenUrl = "https://github.com/eigenteam/eigen-git-mirror/archive/da7909592376c893dabbc4b6453a8ffe46b1eb8e.tar.gz;https://gitlab.com/libeigen/eigen/-/archive/da7909592376c893dabbc4b6453a8ffe46b1eb8e/eigen-da7909592376c893dabbc4b6453a8ffe46b1eb8e.tar.gz"
    $EigenText = Get-Content -Path $EigenCmake -Raw
    $EigenText2 = $EigenText `
        -replace 'URL\s+https://gitlab\.com/libeigen/eigen/-/archive/\S+/eigen-\S+\.tar\.gz', "URL $EigenUrl" `
        -replace '\r?\n\s*URL_HASH[^\r\n]*', ''
    if ($EigenText2 -ne $EigenText) {
        Write-Host "Patching Eigen download URL (GitLab 403 workaround)..." -ForegroundColor Yellow
        Set-Content -Path $EigenCmake -Value $EigenText2 -NoNewline
    }
}

$MaxJobs = 24
$Jobs = [Environment]::ProcessorCount
if ($Jobs -lt 1) { $Jobs = 1 }
if ($Jobs -gt $MaxJobs) { $Jobs = $MaxJobs }

# Clean previous build if requested
if (Test-Path -Path $BuildDir) {
    Write-Host "Cleaning previous build directory: $BuildDir" -ForegroundColor Yellow
    Remove-Item -Path $BuildDir -Recurse -Force
}

Write-Host "Configuring Open3D..." -ForegroundColor Cyan

if (-not (Test-Path -Path $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

cmake -S "$SourceDir" -B "$BuildDir" -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_CONFIGURATION_TYPES="Debug;Release" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DBUILD_SHARED_LIBS=ON `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL `
    -DCMAKE_DEBUG_POSTFIX="d" `
    -DBUILD_PYTHON_MODULE=OFF `
    -DBUILD_GUI=OFF `
    -DBUILD_WEBRTC=OFF `
    -DBUILD_UNIT_TESTS=OFF `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_BENCHMARKS=OFF `
    -DBUILD_CUDA_MODULE=OFF `
    -DBUILD_TENSORFLOW_OPS=OFF `
    -DBUILD_PYTORCH_OPS=OFF `
    -DWITH_OPENMP=ON

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit 1
}

# --- Build & Install Debug ---
Write-Host "Building+Installing Open3D (Debug)..." -ForegroundColor Cyan
cmake --build $BuildDir --config Debug --target INSTALL --parallel $Jobs
if ($LASTEXITCODE -ne 0) { Write-Error "Open3D Debug build failed."; exit 1 }

# --- Build & Install Release ---
Write-Host "Building+Installing Open3D (Release)..." -ForegroundColor Cyan
cmake --build $BuildDir --config Release --target INSTALL --parallel $Jobs
if ($LASTEXITCODE -ne 0) { Write-Error "Open3D Release build failed."; exit 1 }

Write-Host "Success! Open3D (Debug & Release) installed to: $InstallDir" -ForegroundColor Green
Write-Host "Please update your CMakeUserPresets.json to set Open3D_DIR to: $InstallDir/lib/cmake/Open3D" -ForegroundColor Yellow
