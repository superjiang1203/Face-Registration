param(
    [ValidateSet("list", "unpack", "pack")]
    [string]$Action = "list",
    [string]$OnnxRuntimeWinPackage = "onnxruntime-win-x64-gpu-1.24.4.zip",
    [string]$CudnnWinPackage = "cudnn-windows-x86_64-9.5.1.17_cuda12-archive.zip",
    [string]$CudaCublasWinPackage = "libcublas-windows-x86_64-12.8.4.1-archive.zip",
    [string]$CudaCufftWinPackage = "libcufft-windows-x86_64-11.3.3.83-archive.zip",
    [string]$CudaCudartWinPackage = "cuda_cudart-windows-x86_64-12.8.90-archive.zip",
    [string]$VcameraPackage = "VcameraSDK-26.1.10.zip"
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

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Remove-IfExists([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Force
    }
}

function Expand-ZipToThirdparty([string]$ZipPath, [string]$ThirdpartyDir) {
    if (-not (Test-Path -LiteralPath $ZipPath)) {
        Write-Warning "Package not found: $ZipPath"
        return
    }
    Write-Host "Extracting: $ZipPath -> $ThirdpartyDir"
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $ThirdpartyDir -Force
}

function Expand-ZipSingleRoot([string]$ZipPath, [string]$DestinationDir) {
    if (-not (Test-Path -LiteralPath $ZipPath)) {
        Write-Warning "Package not found: $ZipPath"
        return
    }

    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("face-registration-" + [guid]::NewGuid())
    try {
        Write-Host "Extracting: $ZipPath -> $DestinationDir"
        Expand-Archive -LiteralPath $ZipPath -DestinationPath $tempDir -Force
        $roots = @(Get-ChildItem -LiteralPath $tempDir -Directory)
        if ($roots.Count -ne 1) {
            throw "Expected one top-level directory in package: $ZipPath"
        }
        if (Test-Path -LiteralPath $DestinationDir) {
            Remove-Item -LiteralPath $DestinationDir -Recurse -Force
        }
        Move-Item -LiteralPath $roots[0].FullName -Destination $DestinationDir
    } finally {
        if (Test-Path -LiteralPath $tempDir) {
            Remove-Item -LiteralPath $tempDir -Recurse -Force
        }
    }
}

function Expand-ZipSingleRootMerge([string]$ZipPath, [string]$DestinationDir) {
    if (-not (Test-Path -LiteralPath $ZipPath)) {
        Write-Warning "Package not found: $ZipPath"
        return
    }

    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("face-registration-" + [guid]::NewGuid())
    try {
        Write-Host "Extracting: $ZipPath -> $DestinationDir"
        Expand-Archive -LiteralPath $ZipPath -DestinationPath $tempDir -Force
        $roots = @(Get-ChildItem -LiteralPath $tempDir -Directory)
        if ($roots.Count -ne 1) {
            throw "Expected one top-level directory in package: $ZipPath"
        }
        Ensure-Dir $DestinationDir
        Copy-Item -Path (Join-Path $roots[0].FullName "*") -Destination $DestinationDir -Recurse -Force
    } finally {
        if (Test-Path -LiteralPath $tempDir) {
            Remove-Item -LiteralPath $tempDir -Recurse -Force
        }
    }
}

$ProjectRoot = Resolve-ProjectRoot -StartDir $PSScriptRoot
$ThirdpartyDir = Join-Path $ProjectRoot "thirdparty"
$PackagesDir = Join-Path $ThirdpartyDir "packages"
$WindowsDir = Join-Path $PackagesDir "windows"
$LinuxDir = Join-Path $PackagesDir "linux"

Ensure-Dir $WindowsDir
Ensure-Dir $LinuxDir

if ($Action -eq "list") {
    Write-Host "Windows packages: $WindowsDir"
    Get-ChildItem -LiteralPath $WindowsDir -File | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    Write-Host ""
    Write-Host "Linux packages: $LinuxDir"
    Get-ChildItem -LiteralPath $LinuxDir -File | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    exit 0
}

if ($Action -eq "unpack") {
    # PostgreSQL is intentionally archive-only: postgresql.zip is preserved in
    # packages/windows but is not required or extracted by this algorithm project.
    Expand-ZipToThirdparty (Join-Path $WindowsDir "opencv_itk_vtk.zip") $ThirdpartyDir
    Expand-ZipToThirdparty (Join-Path $WindowsDir "open3d.zip") $ThirdpartyDir
    Expand-ZipToThirdparty (Join-Path $WindowsDir "OrbbecSDK.zip") $ThirdpartyDir
    Expand-ZipToThirdparty (Join-Path $WindowsDir $OnnxRuntimeWinPackage) $ThirdpartyDir
    Expand-ZipSingleRoot (Join-Path $WindowsDir $CudnnWinPackage) (Join-Path $ThirdpartyDir "cudnn")
    $CudaRuntimeDir = Join-Path $ThirdpartyDir "cuda_runtime"
    Expand-ZipSingleRootMerge (Join-Path $WindowsDir $CudaCublasWinPackage) $CudaRuntimeDir
    Expand-ZipSingleRootMerge (Join-Path $WindowsDir $CudaCufftWinPackage) $CudaRuntimeDir
    Expand-ZipSingleRootMerge (Join-Path $WindowsDir $CudaCudartWinPackage) $CudaRuntimeDir
    $VcameraArchive = Join-Path $WindowsDir $VcameraPackage
    if (Test-Path -LiteralPath $VcameraArchive) {
        # The vendor archive must contain the top-level VcameraSDK-26.1.10 folder.
        Expand-ZipToThirdparty $VcameraArchive (Join-Path $ThirdpartyDir "src")
    } elseif (-not (Test-Path -LiteralPath (Join-Path $ThirdpartyDir "src/VcameraSDK-26.1.10"))) {
        Write-Warning "Vcamera package not found: $VcameraArchive"
    }
    Write-Host "Done."
    exit 0
}

if ($Action -eq "pack") {
    Ensure-Dir $WindowsDir

    $bundle = Join-Path $WindowsDir "opencv_itk_vtk.zip"
    Remove-IfExists $bundle
    Push-Location $ThirdpartyDir
    try {
        Compress-Archive -Path @("opencv", "itk", "vtk") -DestinationPath $bundle -Force
    } finally {
        Pop-Location
    }

    $o3d = Join-Path $WindowsDir "open3d.zip"
    Remove-IfExists $o3d
    Push-Location $ThirdpartyDir
    try {
        if (Test-Path -LiteralPath (Join-Path $ThirdpartyDir "open3d")) {
            Compress-Archive -Path @("open3d") -DestinationPath $o3d -Force
        }
    } finally {
        Pop-Location
    }

    $ob = Join-Path $WindowsDir "OrbbecSDK.zip"
    Remove-IfExists $ob
    Push-Location $ThirdpartyDir
    try {
        if (Test-Path -LiteralPath (Join-Path $ThirdpartyDir "OrbbecSDK")) {
            Compress-Archive -Path @("OrbbecSDK") -DestinationPath $ob -Force
        }
    } finally {
        Pop-Location
    }

    $vcamera = Join-Path $WindowsDir $VcameraPackage
    Remove-IfExists $vcamera
    Push-Location (Join-Path $ThirdpartyDir "src")
    try {
        if (Test-Path -LiteralPath (Join-Path $ThirdpartyDir "src/VcameraSDK-26.1.10")) {
            Compress-Archive -Path @("VcameraSDK-26.1.10") -DestinationPath $vcamera -Force
        }
    } finally {
        Pop-Location
    }

    Write-Host "Done."
    exit 0
}

throw ("Unknown action: " + $Action)
