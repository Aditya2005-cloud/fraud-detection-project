$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project Python interpreter not found at $python"
}

& $python (Join-Path $PSScriptRoot "app.py")
