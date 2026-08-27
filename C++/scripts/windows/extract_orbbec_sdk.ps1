[CmdletBinding()]
param(
    [ValidateSet("package", "source")]
    [string]$Mode = "package",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path

if ($Mode -eq "package") {
    $Archive = Join-Path $ProjectRoot "thirdparty/packages/windows/OrbbecSDK.zip"
    $Destination = Join-Path $ProjectRoot "thirdparty/OrbbecSDK"
    $RootName = "OrbbecSDK"
} else {
    $Archive = Join-Path $ProjectRoot "thirdparty/src/OrbbecSDK_v2-2.7.2-rc.zip"
    $Destination = Join-Path $ProjectRoot "thirdparty/src/OrbbecSDK_v2-2.7.2-rc"
    $RootName = "OrbbecSDK_v2-2.7.2-rc"
}

if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
    throw "Archive not found: $Archive"
}
if ((Test-Path -LiteralPath $Destination) -and -not $Force) {
    throw "Destination already exists: $Destination (use -Force to replace it)"
}

$Temp = Join-Path ([IO.Path]::GetTempPath()) ("face-registration-orbbec-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $Temp | Out-Null
try {
    Expand-Archive -LiteralPath $Archive -DestinationPath $Temp
    $Extracted = Join-Path $Temp $RootName
    if (-not (Test-Path -LiteralPath $Extracted -PathType Container)) {
        throw "Expected archive root '$RootName' was not found"
    }
    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Move-Item -LiteralPath $Extracted -Destination $Destination
    Write-Host "Orbbec SDK extracted to: $Destination"
} finally {
    if (Test-Path -LiteralPath $Temp) {
        Remove-Item -LiteralPath $Temp -Recurse -Force
    }
}
