# Ren

Ren is a local-first knowledge inbox: **Dump → Organize → Remind**. Drop a PDF, TXT, or Markdown file; Ren keeps a safe managed copy, turns useful concepts into small source-backed cards, and resurfaces cards you have not reviewed.

## Project Run Guide

### Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 20.x or higher (with npm)
- *(Optional)* **Rust & Cargo**: Required only if packaging or running the Tauri desktop app.

---

### Method 1: Quick Start (Single Command)

If you are using Windows PowerShell, you can launch both the backend and frontend development servers simultaneously with:

```powershell
./run-dev.ps1
```

- **Backend API:** [http://127.0.0.1:8765](http://127.0.0.1:8765)
- **Frontend App:** [http://localhost:5173](http://localhost:5173)

---

### Method 2: Step-by-Step Manual Run

Run each service in its own terminal to observe logs independently.

#### 1. Backend (FastAPI + Uvicorn)

Open your first terminal in the repository root:

```powershell
# Create virtual environment (first time only)
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
# .venv\Scripts\activate.bat
# macOS / Linux:
# source .venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Run the backend API server
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --reload
```

- Backend API endpoint: `http://127.0.0.1:8765`
- Interactive API Documentation (Swagger UI): `http://127.0.0.1:8765/docs`

#### 2. Frontend (Vite + React)

Open a second terminal in the repository root:

```powershell
# Navigate to the frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start the Vite development server
npm run dev
```

- Open [http://localhost:5173](http://localhost:5173) in your web browser.

---

### Running the Test Suite

To verify backend endpoints and database logic:

```powershell
# Activate your virtual environment first, then run:
python -m unittest backend/test_app.py
```

---

### Running as Desktop App (Tauri)

Once Rust and the Tauri prerequisites are installed on your machine:

```powershell
cd frontend
npm run tauri:dev
```

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

