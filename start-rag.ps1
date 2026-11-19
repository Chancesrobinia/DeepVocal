param(
    [switch] $Local,
    [switch] $Evaluate,
    [int] $Port = 8000
)

$ErrorActionPreference = 'Stop'
$env:DEEPVOCAL_MILVUS_URI = 'lite'
$env:DEEPVOCAL_LOCAL_EMBEDDING = '1'
$env:DEEPVOCAL_EMBEDDING_PROVIDER = if ($Local) { 'local' } else { 'dashscope' }
$env:DEEPVOCAL_EMBEDDING_MODEL = 'qwen3.7-text-embedding-flash'
$env:DEEPVOCAL_EMBEDDING_DIMENSIONS = '1024'
$env:DEEPVOCAL_EMBEDDING_USE_SAVED_KEY = '1'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

Push-Location $PSScriptRoot
try {
    if ($Evaluate) {
        & $python -m deepvocal.evaluate eval/demo.jsonl --expected-count 10
    } else {
        & $python -m uvicorn deepvocal.main:app --host 127.0.0.1 --port $Port
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
