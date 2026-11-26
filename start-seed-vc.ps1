param(
    [ValidateRange(1024, 65535)]
    [int] $Port = 7860,
    [switch] $Offline,
    [switch] $FullPrecision
)

$ErrorActionPreference = 'Stop'
$seedRoot = Join-Path $PSScriptRoot 'data/seed-vc'
$seedSource = Join-Path $seedRoot 'seed-vc-51383efd921027683c89e5348211d93ff12ac2a8'
$seedPython = Join-Path $seedRoot '.venv/Scripts/python.exe'
$seedBin = Join-Path $seedRoot 'bin'
foreach ($path in @($seedPython, (Join-Path $seedSource 'app_svc.py'), (Join-Path $seedBin 'ffmpeg.exe'), (Join-Path $seedBin 'ffprobe.exe'))) {
    if (-not (Test-Path -LiteralPath $path)) { throw 'Run setup-seed-vc.ps1 before starting Seed-VC.' }
}

$env:PATH = $seedBin + [IO.Path]::PathSeparator + $env:PATH
$env:GRADIO_SERVER_NAME = '127.0.0.1'
$env:GRADIO_SERVER_PORT = [string] $Port
$env:GRADIO_TEMP_DIR = Join-Path $seedRoot 'gradio-temp'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
if ($Offline) { $env:HF_HUB_OFFLINE = '1' }
$precision = if ($FullPrecision) { 'False' } else { 'True' }

Push-Location $seedSource
try {
    & $seedPython -u app_svc.py --fp16 $precision
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
