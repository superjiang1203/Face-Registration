<#
Usage:
  PowerShell (建议在 "x64 Native Tools Command Prompt for VS 2022" 或 VS Developer PowerShell 中执行)
    .\thirdparty\scripts\windows\build_protobuf.ps1

Options:
  -Config Release|Debug

Notes:
  - 默认使用源码目录：thirdparty/src/protobuf
  - Protobuf CMake 导出可能依赖 absl/utf8_range，本脚本会先编译安装：
      thirdparty/absl
      thirdparty/utf8_range
  - 安装到：thirdparty/protobuf
#>
param(
    [string]$Generator = "Visual Studio 17 2022",
    [string]$Arch = "x64",
    [string]$Config = "Release",
    [string]$Jobs = "",
    [bool]$BuildShared = $false,
    [bool]$WithZlib = $false,
    [bool]$RecloneIfIncomplete = $true
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
$SrcDir = "$ThirdpartyDir/src/protobuf"
$BuildDir = "$ThirdpartyDir/build/protobuf"
$InstallDir = "$ThirdpartyDir/protobuf"

$ProtobufTag = "v34.2"

Require-Cmd git
Require-Cmd cmake

if (-not $Jobs) {
    $Jobs = $env:JOBS
}
if (-not $Jobs) {
    $Jobs = "8"
}

function Get-UniquePath {
    param([string]$BasePath)

    $candidate = $BasePath
    $i = 1
    while (Test-Path -LiteralPath $candidate) {
        $candidate = "$BasePath.$i"
        $i++
    }
    return $candidate
}

function Get-ProtobufProtocVersionFromSource {
    param([string]$SourceDir)

    $versionJson = Join-Path $SourceDir "version.json"
    if (-not (Test-Path -LiteralPath $versionJson)) {
        return ""
    }
    try {
        $raw = Get-Content -LiteralPath $versionJson -Raw
        $m = [regex]::Match($raw, '"protoc_version"\s*:\s*"([^"]+)"')
        if ($m.Success) { return $m.Groups[1].Value }
    } catch {
    }
    return ""
}

function Ensure-ProtobufSourcesReady {
    param(
        [Parameter(Mandatory = $true)][string]$SrcDir,
        [Parameter(Mandatory = $true)][bool]$RecloneIfIncomplete
    )

    $utf8Rel = "third_party/utf8_range"
    $expectedProtocVer = $ProtobufTag
    if ($expectedProtocVer.StartsWith("v")) { $expectedProtocVer = $expectedProtocVer.Substring(1) }
    $m = [regex]::Match($expectedProtocVer, '^3\.(\d+)\.(\d+)$')
    if ($m.Success) { $expectedProtocVer = ($m.Groups[1].Value + "." + $m.Groups[2].Value) }

    if (Test-Path -LiteralPath $SrcDir) {
        $actualProtocVer = Get-ProtobufProtocVersionFromSource -SourceDir $SrcDir
        if ($actualProtocVer -and $expectedProtocVer -and ($actualProtocVer -ne $expectedProtocVer)) {
            if (-not $RecloneIfIncomplete) {
                Write-Error "Protobuf source version mismatch: expected $expectedProtocVer but found $actualProtocVer under $SrcDir"
                exit 1
            }
            $backup = Get-UniquePath -BasePath "$SrcDir.mismatch"
            Write-Host "Protobuf source version mismatch, moving aside: $SrcDir -> $backup" -ForegroundColor Yellow
            Move-Item -LiteralPath $SrcDir -Destination $backup
        }
    }

    if ((Test-Path -LiteralPath $SrcDir) -and (Test-Path -LiteralPath (Join-Path $SrcDir $utf8Rel))) {
        return $SrcDir
    }

    if (-not (Test-Path -LiteralPath $SrcDir)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SrcDir) | Out-Null
        Write-Host "Cloning protobuf $ProtobufTag -> $SrcDir" -ForegroundColor Cyan
        git clone --depth 1 --branch $ProtobufTag --recurse-submodules --shallow-submodules https://github.com/protocolbuffers/protobuf.git $SrcDir
        if ($LASTEXITCODE -ne 0) { Write-Error "git clone failed"; exit 1 }
    }

    $gitDir = Join-Path $SrcDir ".git"
    if (Test-Path -LiteralPath $gitDir) {
        Push-Location $SrcDir
        try {
            git submodule update --init --recursive --depth 1
        } finally {
            Pop-Location
        }
    }

    $utf8SrcDir = Join-Path $SrcDir $utf8Rel
    $missingThirdParty = (-not (Test-Path -LiteralPath $utf8SrcDir))

    if ($missingThirdParty -and (Test-Path -LiteralPath $gitDir)) {
        Push-Location $SrcDir
        try {
            git submodule update --init --recursive --depth 1
        } finally {
            Pop-Location
        }
        $missingThirdParty = (-not (Test-Path -LiteralPath $utf8SrcDir))
    }

    if ($missingThirdParty -and $RecloneIfIncomplete) {
        $backup = Get-UniquePath -BasePath "$SrcDir.incomplete"
        Write-Host "Protobuf source tree seems incomplete, moving aside: $SrcDir -> $backup" -ForegroundColor Yellow
        Move-Item -LiteralPath $SrcDir -Destination $backup

        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SrcDir) | Out-Null
        Write-Host "Cloning protobuf $ProtobufTag -> $SrcDir" -ForegroundColor Cyan
        git clone --depth 1 --branch $ProtobufTag --recurse-submodules --shallow-submodules https://github.com/protocolbuffers/protobuf.git $SrcDir
        if ($LASTEXITCODE -ne 0) { Write-Error "git clone failed"; exit 1 }

        Push-Location $SrcDir
        try {
            git submodule update --init --recursive --depth 1
        } finally {
            Pop-Location
        }

        $missingThirdParty = (-not (Test-Path -LiteralPath $utf8SrcDir))
    }

    if ($missingThirdParty) {
        Write-Error "Protobuf third_party dependencies are missing under: $SrcDir/third_party. Please ensure sources are complete, or delete $SrcDir and rerun."
        exit 1
    }

    return $SrcDir
}

function Ensure-CMakeProjectBuilt {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$SourceDir,
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)][string]$InstallDir,
        [Parameter(Mandatory = $true)][string[]]$ConfigureArgs
    )

    if (-not (Test-Path -LiteralPath $SourceDir)) {
        Write-Error "$Name source directory not found: $SourceDir"
        exit 1
    }

    if (Test-Path -LiteralPath $BuildDir) {
        Write-Host "Cleaning: $BuildDir" -ForegroundColor Yellow
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

    Write-Host "Configuring $Name ($Config)..." -ForegroundColor Cyan
    cmake -S "$SourceDir" -B "$BuildDir" -G "$Generator" -A "$Arch" `
        -DCMAKE_CONFIGURATION_TYPES="Debug;Release" `
        -DCMAKE_INSTALL_PREFIX="$InstallDir" `
        @ConfigureArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "CMake configuration failed: $Name"; exit 1 }

    Write-Host "Building $Name ($Config)..." -ForegroundColor Cyan
    cmake --build "$BuildDir" --config "$Config" --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { Write-Error "$Name build failed"; exit 1 }

    Write-Host "Installing $Name ($Config)..." -ForegroundColor Cyan
    cmake --install "$BuildDir" --config "$Config"
    if ($LASTEXITCODE -ne 0) { Write-Error "$Name install failed"; exit 1 }
}

$EffectiveSrcDir = Ensure-ProtobufSourcesReady -SrcDir $SrcDir -RecloneIfIncomplete $RecloneIfIncomplete

$BuildSharedOpt = if ($BuildShared) { "ON" } else { "OFF" }
$WithZlibOpt = if ($WithZlib) { "ON" } else { "OFF" }

if (Test-Path -LiteralPath $BuildDir) {
    Write-Host "Cleaning: $BuildDir" -ForegroundColor Yellow
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Write-Host "Configuring protobuf ($Config)..." -ForegroundColor Cyan
cmake -S "$EffectiveSrcDir" -B "$BuildDir" -G "$Generator" -A "$Arch" `
    -DCMAKE_CONFIGURATION_TYPES="Debug;Release" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -Dprotobuf_BUILD_SHARED_LIBS="$BuildSharedOpt" `
    -DBUILD_SHARED_LIBS="$BuildSharedOpt" `
    -Dprotobuf_BUILD_TESTS=OFF `
    -Dprotobuf_BUILD_EXAMPLES=OFF `
    -Dprotobuf_WITH_ZLIB="$WithZlibOpt" `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
if ($LASTEXITCODE -ne 0) { Write-Error "CMake configuration failed"; exit 1 }

Write-Host "Building protobuf ($Config)..." -ForegroundColor Cyan
cmake --build "$BuildDir" --config "$Config" --parallel $Jobs
if ($LASTEXITCODE -ne 0) { Write-Error "protobuf build failed"; exit 1 }

Write-Host "Installing protobuf ($Config)..." -ForegroundColor Cyan
cmake --install "$BuildDir" --config "$Config"
if ($LASTEXITCODE -ne 0) { Write-Error "protobuf install failed"; exit 1 }

Write-Host "Success! Protobuf installed to: $InstallDir" -ForegroundColor Green
