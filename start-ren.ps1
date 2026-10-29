# ============================================================================
# Ren — Single-Process Local PC Launcher (CPU Mode)
# ============================================================================
# Runs the bundled UI and FastAPI + SQLite + CPU RAG engine on port 8765
# and opens http://127.0.0.1:8765 in your browser.
# ============================================================================

param(
    [int]$Port = 8765,
    [switch]$NoBrowser = $false
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "[!] Virtual environment not found. Running install.ps1 first..." -ForegroundColor Yellow
    & (Join-Path $projectRoot 'install.ps1')
}

$distIndex = Join-Path $projectRoot 'frontend\dist\index.html'
if (-not (Test-Path $distIndex)) {
    Write-Host "[!] Frontend production build missing. Building frontend/dist..." -ForegroundColor Yellow
    & npm --prefix frontend run build
}

$argsList = @('-m', 'backend.launcher', '--port', "$Port")
if ($NoBrowser) {
    $argsList += '--no-browser'
}

& $venvPython @argsList

