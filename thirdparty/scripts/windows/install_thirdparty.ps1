# Script to manually install pre-built third-party libraries to the project directory
# Run this script in PowerShell

param(
    [string]$VtkVersion = "9.4.2",
    [string]$ItkVersion = "5.4.5",
    [string]$OpenCvVersion = "4.12.0",
    [string]$BuildRoot = "",
    [string]$VtkBuildDir = "",
    [string]$ItkBuildDir = "",
    [string]$OpenCvBuildDir = ""
)

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
$DestDir = "$ProjectRoot/thirdparty"

$ResolvedBuildRoot = $BuildRoot
if (-not $ResolvedBuildRoot) {
    $ResolvedBuildRoot = "$ProjectRoot/thirdparty/build"
}
if (-not $VtkBuildDir) {
    $VtkBuildDir = "$ResolvedBuildRoot/vtk-$VtkVersion"
}
if (-not $ItkBuildDir) {
    $ItkBuildDir = "$ResolvedBuildRoot/itk-$ItkVersion"
}
if (-not $OpenCvBuildDir) {
    $OpenCvBuildDir = "$ResolvedBuildRoot/opencv-$OpenCvVersion"
}

# --- VTK ---
Write-Host "Installing VTK to $DestDir/vtk ..." -ForegroundColor Cyan
if (Test-Path $VtkBuildDir) {
    cmake --install "$VtkBuildDir" --prefix "$DestDir/vtk" --config Release
    cmake --install "$VtkBuildDir" --prefix "$DestDir/vtk" --config Debug
} else {
    Write-Error "VTK Build directory not found: $VtkBuildDir"
}

# --- ITK ---
Write-Host "Installing ITK to $DestDir/itk ..." -ForegroundColor Cyan
if (Test-Path $ItkBuildDir) {
    cmake --install "$ItkBuildDir" --prefix "$DestDir/itk" --config Release
    cmake --install "$ItkBuildDir" --prefix "$DestDir/itk" --config Debug
} else {
    Write-Error "ITK Build directory not found: $ItkBuildDir"
}

# --- OpenCV ---
Write-Host "Installing OpenCV to $DestDir/opencv ..." -ForegroundColor Cyan
if (Test-Path $OpenCvBuildDir) {
    cmake --install "$OpenCvBuildDir" --prefix "$DestDir/opencv" --config Release
    cmake --install "$OpenCvBuildDir" --prefix "$DestDir/opencv" --config Debug
} else {
    Write-Error "OpenCV Build directory not found: $OpenCvBuildDir"
}

Write-Host "All third-party libraries installed successfully!" -ForegroundColor Green
