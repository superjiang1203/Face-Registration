[CmdletBinding()]
param(
    [string]$ConfigFile = "C++/config/runtime.yml",
    [string]$CameraSn = "",
    [ValidateRange(1, 64)][int]$Threads = 4
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Exe = Join-Path $ProjectRoot "build/C++/Release/face_camera_pipeline.exe"
$Arguments = @("--config", $ConfigFile, "--target-locator", "face_detection", "--threads", $Threads, "--camera-backend", "orbbec")
if ($CameraSn) { $Arguments += @("--camera-sn", $CameraSn) }
Push-Location $ProjectRoot
try { & $Exe @Arguments; exit $LASTEXITCODE } finally { Pop-Location }
