[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [string]$Version = "26.1.10",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot([string]$StartDir) {
    $dir = Resolve-Path -LiteralPath $StartDir
    while ($true) {
        if ((Test-Path -LiteralPath (Join-Path $dir "CMakeLists.txt")) -and
            (Test-Path -LiteralPath (Join-Path $dir "thirdparty"))) {
            return $dir.Path
        }
        $parent = Split-Path -Parent $dir
        if (-not $parent -or $parent -eq $dir) { break }
        $dir = Resolve-Path -LiteralPath $parent
    }
    throw "Cannot locate project root from: $StartDir"
}

$ProjectRoot = Resolve-ProjectRoot $PSScriptRoot
$ThirdpartyDir = Join-Path $ProjectRoot "thirdparty"
$SourceDir = Join-Path $ThirdpartyDir "src/VcameraSDK-$Version"
$InstallDir = Join-Path $ThirdpartyDir "VcameraSDK"
$BuildDir = Join-Path $ThirdpartyDir "build/VcameraSDK-$Version-examples"

if (-not (Test-Path -LiteralPath (Join-Path $SourceDir "cpp/example/CMakeLists.txt"))) {
    throw "Vcamera SDK source is missing: $SourceDir. Put VcameraSDK-$Version.zip in thirdparty/packages/windows or extract it under thirdparty/src."
}

# Vcamera is distributed as a binary SDK. "Install" means normalizing its
# headers, import libraries, DLLs, configuration files and original examples.
if ($Clean -and (Test-Path -LiteralPath $InstallDir)) {
    $resolvedThirdparty = (Resolve-Path -LiteralPath $ThirdpartyDir).Path
    $resolvedInstall = (Resolve-Path -LiteralPath $InstallDir).Path
    if (-not $resolvedInstall.StartsWith($resolvedThirdparty + [IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing to remove a directory outside thirdparty: $resolvedInstall"
    }
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -LiteralPath (Join-Path $SourceDir "cpp") -Destination $InstallDir -Recurse -Force
Copy-Item -LiteralPath (Join-Path $SourceDir "doc") -Destination $InstallDir -Recurse -Force
Copy-Item -LiteralPath (Join-Path $SourceDir "README.md") -Destination $InstallDir -Force
Copy-Item -LiteralPath (Join-Path $SourceDir "README_zh.md") -Destination $InstallDir -Force

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
    $resolvedBuild = (Resolve-Path -LiteralPath $BuildDir).Path
    $resolvedThirdparty = (Resolve-Path -LiteralPath $ThirdpartyDir).Path
    if (-not $resolvedBuild.StartsWith($resolvedThirdparty + [IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing to remove a build directory outside thirdparty: $resolvedBuild"
    }
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

$OpenCvDir = Join-Path $ThirdpartyDir "opencv/x64/vc17/lib"
$ConfigureArgs = @(
    "-S", (Join-Path $InstallDir "cpp/example"),
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64"
)
if (Test-Path -LiteralPath (Join-Path $OpenCvDir "OpenCVConfig.cmake")) {
    $ConfigureArgs += "-DOpenCV_DIR=$OpenCvDir"
}

Write-Host "Configuring Vcamera SDK examples..." -ForegroundColor Cyan
& cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) { throw "Vcamera SDK example configuration failed" }

Write-Host "Building Vcamera SDK examples ($Config)..." -ForegroundColor Cyan
cmake --build $BuildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "Vcamera SDK example build failed" }

$OutputDir = Join-Path $InstallDir "examples/$Config"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Copy-Item -Path (Join-Path $BuildDir "$Config/*.exe") -Destination $OutputDir -Force
Copy-Item -Path (Join-Path $InstallDir "cpp/win/$Config/bin/*") -Destination $OutputDir -Recurse -Force

Write-Host "Vcamera SDK installed to: $InstallDir" -ForegroundColor Green
Write-Host "Compiled examples and runtime: $OutputDir" -ForegroundColor Green
