<#
Usage:
  PowerShell (建议在 "x64 Native Tools Command Prompt for VS 2022" 或 VS Developer PowerShell 中执行)
    .\thirdparty\scripts\windows\build_grpc.ps1

Options:
  - 无

Notes:
  - 默认使用源码目录：thirdparty/src/grpc
  - 若源码目录不存在，会按 Tag 自动 git clone（包含 submodules）
  - 安装到：thirdparty/grpc
  - 下载源码：git clone --depth 1 -b v1.78.1 --recurse-submodules --shallow-submodules https://github.com/grpc/grpc.git 
  - 本脚本按 gRPC 官方 Quickstart 的方式：在 gRPC 仓库内同时编译并安装 gRPC + Protobuf
  - 默认安装 Debug + Release 两个版本
#>
param(
    [string]$Generator = "Visual Studio 17 2022",
    [string]$Arch = "x64",
    [string]$Jobs = ""
)

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

function Require-Cmd([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "Missing command: $Name"
        exit 1
    }
}

$ProjectRoot = Resolve-ProjectRoot -StartDir $PSScriptRoot
$ThirdpartyDir = "$ProjectRoot/thirdparty"
$SrcDir = "$ThirdpartyDir/src/grpc"
$BuildDir = "$ThirdpartyDir/build/grpc"
$InstallDir = "$ThirdpartyDir/grpc"

$GrpcTag = "v1.78.1"

Require-Cmd git
Require-Cmd cmake

if (-not $Jobs) {
    $Jobs = $env:JOBS
}
if (-not $Jobs) {
    $Jobs = "8"
}

$RequestedConfigs = @("Debug", "Release")
$RequestedConfigsText = ($RequestedConfigs -join ", ")

if (-not (Test-Path -LiteralPath $SrcDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SrcDir) | Out-Null
    Write-Host "Cloning gRPC $GrpcTag -> $SrcDir" -ForegroundColor Cyan
    git clone --depth 1 --branch $GrpcTag --recurse-submodules --shallow-submodules https://github.com/grpc/grpc.git $SrcDir
    if ($LASTEXITCODE -ne 0) { Write-Error "git clone failed"; exit 1 }
} else {
    if (Test-Path -LiteralPath (Join-Path $SrcDir ".git")) {
        Push-Location $SrcDir
        try {
            git submodule update --init --recursive
        } finally {
            Pop-Location
        }
    }
}

if (Test-Path -LiteralPath $BuildDir) {
    Write-Host "Cleaning: $BuildDir" -ForegroundColor Yellow
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Write-Host "Configuring gRPC ($RequestedConfigsText)..." -ForegroundColor Cyan
cmake -S "$SrcDir" -B "$BuildDir" -G "$Generator" -A "$Arch" `
    -DCMAKE_CXX_STANDARD=17 `
    -DCMAKE_CONFIGURATION_TYPES="Debug;Release" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DgRPC_INSTALL=ON `
    -DgRPC_BUILD_CODEGEN=ON `
    -DgRPC_BUILD_TESTS=OFF `
    -DgRPC_PROTOBUF_PROVIDER=module `
    -DgRPC_ZLIB_PROVIDER=module `
    -DgRPC_SSL_PROVIDER=module `
    -DgRPC_ABSL_PROVIDER=module `
    -DgRPC_RE2_PROVIDER=module `
    -DgRPC_CARES_PROVIDER=module
if ($LASTEXITCODE -ne 0) { Write-Error "CMake configuration failed"; exit 1 }

foreach ($CurrentConfig in $RequestedConfigs) {
    Write-Host "Building & installing gRPC ($CurrentConfig)..." -ForegroundColor Cyan
    cmake --build "$BuildDir" --config "$CurrentConfig" --target install --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { Write-Error "gRPC build/install failed ($CurrentConfig)"; exit 1 }
}

Write-Host "Success! gRPC ($RequestedConfigsText) installed to: $InstallDir" -ForegroundColor Green
Write-Host "Set gRPC_DIR to: $InstallDir/lib/cmake/grpc" -ForegroundColor Yellow
Write-Host "Set Protobuf_DIR to: $InstallDir/lib/cmake/protobuf" -ForegroundColor Yellow
