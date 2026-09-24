# Ren

Ren is a local-first personal context engine. Import files or capture a thought; Ren stores an untouched managed copy, derives searchable knowledge, drafts cited topic notes, and answers questions from your own sources.

## Development

Prerequisites: Python 3.11+ and Node 20+. Rust/Cargo is required only to package the Tauri desktop executable.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn backend.app:app --reload --port 8765
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the shown local URL. The development UI expects the API at `http://127.0.0.1:8765`.

Run `./run-dev.ps1` instead to start both processes from the repository root.

## Privacy model

Ren copies imported files into its own application-data library and never moves, renames, modifies, or deletes originals. Text, chunks, notes, and the SQLite database are local. API keys are stored in the operating-system credential vault when available and are never written to SQLite. AI processing is optional; without a key Ren uses local extraction, search, note drafting, and extractive answers.

## Desktop packaging

Install Rust and the Tauri prerequisites, then run `npm run tauri:dev` inside `frontend`. The source tree includes the Tauri wrapper in `src-tauri/`.

For a distributable Windows installer, package the Python service as a PyInstaller sidecar and register it with Tauri. This is intentionally the final packaging step: Ren's interface and service can be developed and personally tested before adding binary-distribution complexity.

