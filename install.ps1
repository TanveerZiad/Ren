# ============================================================================
# Ren — Windows Local PC Installer (CPU Mode)
# ============================================================================
# Installs Python virtualenv, builds the production React UI, initializes the
# local SQLite database in %APPDATA%\Ren, and creates Start Menu / Desktop
# launchers so Ren runs locally on your PC CPU with one click.
# ============================================================================

param(
    [switch]$CreateDesktopShortcut = $true,
    [switch]$SkipTests = $false
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Ren Local PC Installer (100% CPU Mode)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Verify Python 3.11+
Write-Host "`n[1/6] Checking Python installation..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    throw "Python 3.11+ is required but was not found on PATH. Install Python from https://www.python.org/downloads/"
}
$pyVersion = & python --version 2>&1
Write-Host "      Found $pyVersion" -ForegroundColor Green

# 2. Verify Node.js & npm
Write-Host "`n[2/6] Checking Node.js & npm..." -ForegroundColor Yellow
$npmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCmd) {
    throw "Node.js (with npm) is required to build the frontend UI. Install from https://nodejs.org/"
}
$nodeVersion = & node --version 2>&1
Write-Host "      Found Node.js $nodeVersion" -ForegroundColor Green

# 3. Create / Update Python Virtual Environment & Install CPU Dependencies
Write-Host "`n[3/6] Setting up local Python environment (.venv)..." -ForegroundColor Yellow
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    & python -m venv .venv
}
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt --quiet
Write-Host "      Python CPU backend dependencies installed." -ForegroundColor Green

# 4. Install Frontend Packages & Build Production Static Bundle
Write-Host "`n[4/6] Building production frontend bundle (frontend/dist)..." -ForegroundColor Yellow
if (-not (Test-Path (Join-Path $projectRoot 'frontend\node_modules'))) {
    & npm --prefix frontend install --no-audit --no-fund
}
& npm --prefix frontend run build
Write-Host "      Production UI built into frontend/dist." -ForegroundColor Green

# 5. Initialize Local SQLite Database & CPU Defaults
Write-Host "`n[5/6] Initializing local SQLite database and CPU settings..." -ForegroundColor Yellow
& $venvPython -c "from backend.app import initialize_database; initialize_database()"
Write-Host "      Database initialized in $env:APPDATA\Ren\ren.sqlite3 (CPU mode)." -ForegroundColor Green

# 6. Create Windows Start Menu & Desktop Shortcuts
Write-Host "`n[6/6] Creating Windows Shortcuts..." -ForegroundColor Yellow
$launcherBat = Join-Path $projectRoot 'start-ren.bat'
$wshShell = New-Object -ComObject WScript.Shell

# Start Menu Shortcut
$startMenuDir = [Environment]::GetFolderPath('Programs')
$startShortcutPath = Join-Path $startMenuDir 'Ren Knowledge Companion.lnk'
$shortcut = $wshShell.CreateShortcut($startShortcutPath)
$shortcut.TargetPath = $launcherBat
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = "Ren Local-First Personal Knowledge Companion (CPU Mode)"
$shortcut.Save()
Write-Host "      Start Menu shortcut created: $startShortcutPath" -ForegroundColor Green

if ($CreateDesktopShortcut) {
    $desktopDir = [Environment]::GetFolderPath('Desktop')
    $desktopShortcutPath = Join-Path $desktopDir 'Ren.lnk'
    $dShortcut = $wshShell.CreateShortcut($desktopShortcutPath)
    $dShortcut.TargetPath = $launcherBat
    $dShortcut.WorkingDirectory = $projectRoot
    $dShortcut.Description = "Ren Local-First Personal Knowledge Companion (CPU Mode)"
    $dShortcut.Save()
    Write-Host "      Desktop shortcut created: $desktopShortcutPath" -ForegroundColor Green
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  To start Ren anytime:"
Write-Host "    - Double-click 'start-ren.bat' (or the 'Ren' Desktop shortcut)"
Write-Host "    - Or run: .\start-ren.ps1"
Write-Host "    - Opens at: http://127.0.0.1:8765"
Write-Host "=================================================================" -ForegroundColor Cyan

