# Ren

Ren is a local-first knowledge inbox: **Dump → Organize → Remind**. Drop a PDF, TXT, or Markdown file; Ren keeps a safe managed copy, turns useful concepts into small source-backed cards, and resurfaces cards you have not reviewed.

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

## What Ren does now

- **Inbox:** import PDF, TXT, and Markdown files only.
- **My Knowledge:** browse and search the small knowledge cards extracted from each source.
- **Remember:** review a small queue. Cards return after 1, 3, 7, 14, then 30 days.
- **Source-backed:** every card opens its local source file. Ren never edits or moves the original.

The first version intentionally has no document chat, agents, cloud sync, browser extension, vector database, or knowledge graph.

## Privacy model

Ren copies imported files into its own application-data library and never moves, renames, modifies, or deletes originals. Extracted text, knowledge cards, review history, and the SQLite database are local. The first version does not require an AI key.

## Desktop packaging

Install Rust and the Tauri prerequisites, then run `npm run tauri:dev` inside `frontend`. The source tree includes the Tauri wrapper in `src-tauri/`.

For a distributable Windows installer, package the Python service as a PyInstaller sidecar and register it with Tauri. This is intentionally the final packaging step: Ren's interface and service can be developed and personally tested before adding binary-distribution complexity.

