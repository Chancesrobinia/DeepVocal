$ErrorActionPreference = 'Stop'
$seedRevision = '51383efd921027683c89e5348211d93ff12ac2a8'
$seedRoot = Join-Path $PSScriptRoot 'data/seed-vc'
$seedSource = Join-Path $seedRoot "seed-vc-$seedRevision"
$seedVenv = Join-Path $seedRoot '.venv'
$seedPython = Join-Path $seedVenv 'Scripts/python.exe'
$seedBin = Join-Path $seedRoot 'bin'
New-Item -ItemType Directory -Force -Path $seedRoot, $seedBin | Out-Null
$env:PIP_CACHE_DIR = Join-Path $seedRoot 'pip-cache'

if (-not (Test-Path -LiteralPath (Join-Path $seedSource 'app_svc.py'))) {
    $archive = Join-Path $seedRoot "seed-vc-$seedRevision.zip"
    curl.exe --fail --location --retry 2 --silent --show-error --output $archive "https://codeload.github.com/Plachtaa/seed-vc/zip/$seedRevision"
    if ($LASTEXITCODE -ne 0) { throw 'Seed-VC source download failed.' }
    Expand-Archive -LiteralPath $archive -DestinationPath $seedRoot -Force
}
if (-not (Test-Path -LiteralPath $seedPython)) {
    py -3.11 -m venv $seedVenv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Seed-VC Python 3.11 environment.' }
}

$torchPackages = @('torch==2.4.0+cu118', 'torchvision==0.19.0+cu118', 'torchaudio==2.4.0+cu118')
& $seedPython -m pip install --disable-pip-version-check --timeout 45 --retries 2 @torchPackages --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw 'Could not install the CUDA build of PyTorch.' }

# The PyTorch 2.4.0 Windows wheel links to this runtime without bundling it.
$ompLibrary = 'libomp140.x86_64.dll'
$torchLib = Join-Path $seedVenv 'Lib/site-packages/torch/lib'
$ompTarget = Join-Path $torchLib $ompLibrary
if (-not (Test-Path -LiteralPath $ompTarget) -and -not (Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32/$ompLibrary"))) {
    $runtimePackage = Join-Path $seedRoot 'microsoft.quantum.simulators.0.28.263081.nupkg'
    curl.exe --fail --location --retry 2 --max-time 90 --silent --show-error --output $runtimePackage 'https://api.nuget.org/v3-flatcontainer/microsoft.quantum.simulators/0.28.263081/microsoft.quantum.simulators.0.28.263081.nupkg'
    if ($LASTEXITCODE -ne 0) { throw 'Could not download the Microsoft OpenMP runtime.' }
    $runtimeDll = Join-Path $seedRoot $ompLibrary
    & $seedPython -c "from pathlib import Path; from sys import argv; from zipfile import ZipFile; Path(argv[2]).write_bytes(ZipFile(argv[1]).read('runtimes/win-x64/native/libomp140.x86_64.dll'))" $runtimePackage $runtimeDll
    if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $runtimeDll).Hash -ne '009B3AABFAD6C4CCDD395B1BEFFF744C141E51B2C124CA24C36F6D4631365CEF') {
        throw 'Could not verify the Microsoft OpenMP runtime.'
    }
    Copy-Item -LiteralPath $runtimeDll -Destination $ompTarget
}

& $seedPython -c "import torch; assert torch.cuda.is_available(); values = torch.arange(4, device='cuda'); assert (values * values).sum().item() == 14; print(torch.cuda.get_device_name())"
if ($LASTEXITCODE -ne 0) { throw 'The local PyTorch environment could not run a CUDA operation.' }

& $seedPython -m pip install --disable-pip-version-check --timeout 45 --retries 2 -r (Join-Path $PSScriptRoot 'requirements-seed-vc.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the Seed-VC inference dependencies.' }

$audioExecutables = @('ffmpeg.exe', 'ffprobe.exe')
$missingAudioTools = $audioExecutables | Where-Object { -not (Test-Path -LiteralPath (Join-Path $seedBin $_)) }
if ($missingAudioTools) {
    $ffmpegBuild = 'ffmpeg-9.1-essentials_build'
    $ffmpegArchive = Join-Path $seedRoot "$ffmpegBuild.zip"
    curl.exe --fail --location --retry 2 --max-time 180 --silent --show-error --output $ffmpegArchive "https://github.com/GyanD/codexffmpeg/releases/download/9.1/$ffmpegBuild.zip"
    if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $ffmpegArchive).Hash -ne 'B9BBBB3ACECE000CBEEC994F856649354E52C773807841E78908374B24D658FD') {
        throw 'Could not verify the FFmpeg archive.'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($ffmpegArchive)
    try {
        foreach ($executable in $audioExecutables) {
            $entry = $zip.GetEntry("$ffmpegBuild/bin/$executable")
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $seedBin $executable), $true)
        }
    } finally {
        $zip.Dispose()
    }
}
foreach ($executable in $audioExecutables) {
    & (Join-Path $seedBin $executable) -version | Select-Object -First 1
    if ($LASTEXITCODE -ne 0) { throw "Could not run $executable." }
}
& $seedPython -m pip check
exit $LASTEXITCODE
