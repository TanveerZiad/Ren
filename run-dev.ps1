$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'backend.app:app', '--port', '8765', '--reload' -WorkingDirectory $projectRoot
Set-Location (Join-Path $projectRoot 'frontend')
npm run dev
