from __future__ import annotations

import hashlib
import html
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import threading
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_NAME = "Ren"
DEFAULT_DATA_DIR = Path(os.getenv("APPDATA", Path.home() / ".ren")) / "Ren"
DATA_DIR = Path(os.getenv("REN_DATA_DIR", DEFAULT_DATA_DIR))
LIBRARY_DIR = DATA_DIR / "library"
NOTES_DIR = DATA_DIR / "notes"
DB_PATH = DATA_DIR / "ren.sqlite3"


def now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_directories() -> None:
    for directory in (DATA_DIR, LIBRARY_DIR, NOTES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


@contextmanager
def db_connection():
    ensure_directories()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database() -> None:
    with db_connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
              id TEXT PRIMARY KEY, filename TEXT NOT NULL, source_kind TEXT NOT NULL,
              original_path TEXT, managed_path TEXT, url TEXT, mime_type TEXT,
              content_hash TEXT, extracted_text TEXT DEFAULT '', status TEXT NOT NULL,
              error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash)
              WHERE content_hash IS NOT NULL;
            CREATE TABLE IF NOT EXISTS processing_jobs (
              id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              status TEXT NOT NULL, stage TEXT NOT NULL, error_message TEXT,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              ordinal INTEGER NOT NULL, text TEXT NOT NULL, anchor TEXT, created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text);
            CREATE TABLE IF NOT EXISTS topics (
              id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
              summary TEXT DEFAULT '', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_topics (
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
              confidence REAL NOT NULL, PRIMARY KEY(source_id, topic_id)
            );
            CREATE TABLE IF NOT EXISTS notes (
              id TEXT PRIMARY KEY, topic_id TEXT NOT NULL UNIQUE REFERENCES topics(id) ON DELETE CASCADE,
              status TEXT NOT NULL, current_revision_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS note_revisions (
              id TEXT PRIMARY KEY, note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
              status TEXT NOT NULL, body TEXT NOT NULL, diff_summary TEXT NOT NULL,
              created_at TEXT NOT NULL, reviewed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS citations (
              id TEXT PRIMARY KEY, revision_id TEXT REFERENCES note_revisions(id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL, label TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS questions (
              id TEXT PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            """
        )


class StripHtml(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        elif tag in {"p", "br", "div", "h1", "h2", "h3", "li", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def extract_text(path: Path, source_kind: str) -> str:
    suffix = path.suffix.lower()
    if source_kind == "text" or suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        parser = StripHtml()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return html.unescape(" ".join(parser.parts))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        except Exception as error:
            raise ValueError(f"PDF extraction failed: {error}") from error
    if suffix == ".docx":
        try:
            from docx import Document
            return "\n".join(p.text for p in Document(str(path)).paragraphs)
        except Exception as error:
            raise ValueError(f"DOCX extraction failed: {error}") from error
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def make_chunks(text: str, size: int = 1100, overlap: int = 180) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


STOP_WORDS = {"about", "after", "again", "also", "and", "are", "been", "from", "have", "into", "more", "notes", "that", "this", "with", "your", "the", "for", "how", "what"}


def suggested_topics(filename: str, text: str) -> list[str]:
    candidates = re.findall(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE)
    candidates += re.findall(r"\b[A-Z][A-Za-z0-9+\-]{2,}(?:\s+[A-Z][A-Za-z0-9+\-]{2,}){0,2}\b", text[:8000])
    candidates += re.split(r"[_\-\.]+", Path(filename).stem)
    unique: list[str] = []
    for candidate in candidates:
        candidate = re.sub(r"\s+", " ", candidate).strip(" -_:.")
        if not candidate or candidate.lower() in STOP_WORDS or len(candidate) < 3 or len(candidate) > 60:
            continue
        if candidate.lower() not in {item.lower() for item in unique}:
            unique.append(candidate)
        if len(unique) == 5:
            break
    return unique or [Path(filename).stem[:60] or "Inbox"]


def get_or_create_topic(db: sqlite3.Connection, name: str) -> str:
    row = db.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    topic_id = str(uuid.uuid4())
    db.execute("INSERT INTO topics(id, name, created_at) VALUES (?, ?, ?)", (topic_id, name, now()))
    return topic_id


def update_job(db: sqlite3.Connection, job_id: str, status: str, stage: str, error: str | None = None) -> None:
    db.execute("UPDATE processing_jobs SET status=?, stage=?, error_message=?, updated_at=? WHERE id=?", (status, stage, error, now(), job_id))


def process_source(source_id: str, job_id: str) -> None:
    affected_topics: list[str] = []
    with db_connection() as db:
        source = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        if not source:
            return
        try:
            update_job(db, job_id, "processing", "extracting")
            text = source["extracted_text"] or extract_text(Path(source["managed_path"]), source["source_kind"])
            text = normalize_text(text)
            if not text:
                raise ValueError("No readable text was found in this source.")
            db.execute("UPDATE sources SET extracted_text=?, status='processing', updated_at=? WHERE id=?", (text, now(), source_id))
            update_job(db, job_id, "processing", "chunking")
            db.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id=?)", (source_id,))
            db.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
            for ordinal, chunk_text in enumerate(make_chunks(text), start=1):
                chunk_id = str(uuid.uuid4())
                anchor = f"Section {ordinal}"
                db.execute("INSERT INTO chunks(id, source_id, ordinal, text, anchor, created_at) VALUES (?, ?, ?, ?, ?, ?)", (chunk_id, source_id, ordinal, chunk_text, anchor, now()))
                db.execute("INSERT INTO chunks_fts(chunk_id, text) VALUES (?, ?)", (chunk_id, chunk_text))
            update_job(db, job_id, "processing", "suggesting topics")
            for topic_name in suggested_topics(source["filename"], text):
                topic_id = get_or_create_topic(db, topic_name)
                affected_topics.append(topic_id)
                db.execute("INSERT OR REPLACE INTO source_topics(source_id, topic_id, confidence) VALUES (?, ?, ?)", (source_id, topic_id, 0.65))
            db.execute("UPDATE sources SET status='ready', updated_at=? WHERE id=?", (now(), source_id))
            update_job(db, job_id, "completed", "ready")
        except Exception as error:
            db.execute("UPDATE sources SET status='failed', error_message=?, updated_at=? WHERE id=?", (str(error), now(), source_id))
            update_job(db, job_id, "failed", "failed", str(error))
    # A draft is derived knowledge, never a silent edit. Creating it after the
    # transaction avoids competing SQLite writes while the source is indexed.
    for topic_id in dict.fromkeys(affected_topics):
        try:
            create_note_draft(topic_id)
        except ValueError:
            pass


def query_rows(query: str, limit: int = 8) -> list[sqlite3.Row]:
    terms = re.findall(r"[\w-]{2,}", query)
    if not terms:
        return []
    match = " OR ".join(f'"{term}"' for term in terms[:12])
    with db_connection() as db:
        try:
            return db.execute(
                """SELECT c.*, s.filename FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id
                   JOIN sources s ON s.id=c.source_id WHERE chunks_fts MATCH ?
                   ORDER BY bm25(chunks_fts) LIMIT ?""",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return db.execute(
                "SELECT c.*, s.filename FROM chunks c JOIN sources s ON s.id=c.source_id WHERE c.text LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()


def create_note_draft(topic_id: str) -> dict[str, Any]:
    with db_connection() as db:
        topic = db.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
        if not topic:
            raise KeyError(topic_id)
        pending = db.execute(
            """SELECT r.*, n.id AS resolved_note_id FROM notes n JOIN note_revisions r ON r.note_id=n.id
               WHERE n.topic_id=? AND r.status='draft' ORDER BY r.created_at DESC LIMIT 1""", (topic_id,)
        ).fetchone()
        if pending:
            return {"id": pending["id"], "note_id": pending["resolved_note_id"], "topic": topic["name"], "body": pending["body"], "status": "draft", "diff_summary": pending["diff_summary"]}
        sources = db.execute(
            """SELECT s.* FROM sources s JOIN source_topics st ON st.source_id=s.id
               WHERE st.topic_id=? AND s.status='ready' ORDER BY s.created_at DESC""", (topic_id,)
        ).fetchall()
        if not sources:
            raise ValueError("This topic has no processed sources yet.")
        facts = []
        for source in sources[:6]:
            excerpt = normalize_text(source["extracted_text"])[:420]
            facts.append(f"- {excerpt}")
        source_lines = [f"- [{source['filename']}](source:{source['id']})" for source in sources]
        body = f"""# {topic['name']}\n\n## Overview\nThis living note is based on {len(sources)} source(s) currently linked to **{topic['name']}**. Review and edit this draft before approval.\n\n## Current understanding\n{chr(10).join(facts)}\n\n## Open questions\n- What should I learn or verify next about {topic['name']}?\n\n## Sources\n{chr(10).join(source_lines)}\n\n_Last generated by Ren on {datetime.now().strftime('%B %d, %Y')}_\n"""
        note = db.execute("SELECT * FROM notes WHERE topic_id=?", (topic_id,)).fetchone()
        if not note:
            note_id = str(uuid.uuid4())
            db.execute("INSERT INTO notes(id, topic_id, status, created_at, updated_at) VALUES (?, ?, 'draft', ?, ?)", (note_id, topic_id, now(), now()))
        else:
            note_id = note["id"]
        revision_id = str(uuid.uuid4())
        db.execute("INSERT INTO note_revisions(id, note_id, status, body, diff_summary, created_at) VALUES (?, ?, 'draft', ?, ?, ?)", (revision_id, note_id, body, "Generated from newly linked sources.", now()))
        for source in sources:
            db.execute("INSERT INTO citations(id, revision_id, source_id, label) VALUES (?, ?, ?, ?)", (str(uuid.uuid4()), revision_id, source["id"], source["filename"]))
        db.execute("UPDATE notes SET status='draft', updated_at=? WHERE id=?", (now(), note_id))
        return {"id": revision_id, "note_id": note_id, "topic": topic["name"], "body": body, "status": "draft", "diff_summary": "Generated from newly linked sources."}


class TextCapture(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1)


class UrlCapture(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    url: str = Field(min_length=3, max_length=2048)
    context: str = ""


class SettingsInput(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: str | None = None


class QuestionInput(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


app = FastAPI(title="Ren Local Service", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "tauri://localhost"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
initialize_database()
_session_api_key: str | None = None


def store_api_key(value: str) -> bool:
    global _session_api_key
    try:
        import keyring
        keyring.set_password("Ren", "ai_api_key", value)
        _session_api_key = None
        return True
    except Exception:
        _session_api_key = value
        return False


def api_key_available() -> bool:
    if _session_api_key:
        return True
    try:
        import keyring
        return bool(keyring.get_password("Ren", "ai_api_key"))
    except Exception:
        return False


def get_api_key() -> str | None:
    if _session_api_key:
        return _session_api_key
    try:
        import keyring
        return keyring.get_password("Ren", "ai_api_key")
    except Exception:
        return None


def provider_answer(question: str, rows: list[sqlite3.Row]) -> str | None:
    """Ask the configured provider using only already-retrieved local context.

    Returning None deliberately selects the source-extractive fallback. A provider
    failure must never prevent a user from retrieving their own information.
    """
    key = get_api_key()
    if not key:
        return None
    with db_connection() as db:
        configured = {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM settings")}
    context = "\n\n".join(f"[S{index}] {row['filename']} — {row['anchor'] or 'Source'}\n{row['text'][:1200]}" for index, row in enumerate(rows, 1))
    prompt = (
        "Answer the user's question using only the supplied Ren source excerpts. "
        "If the excerpts do not establish an answer, say that clearly. Be concise. "
        "Cite claims inline using [S1], [S2], and so on.\n\n"
        f"Question: {question}\n\nSource excerpts:\n{context}"
    )
    payload = json.dumps({
        "model": configured.get("model", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode("utf-8")
    base_url = configured.get("base_url", "https://api.openai.com/v1").rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError):
        return None


def source_payload(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row) | {"topics": [item["name"] for item in get_source_topics(row["id"])]}


def get_source_topics(source_id: str) -> list[sqlite3.Row]:
    with db_connection() as db:
        return db.execute("SELECT t.* FROM topics t JOIN source_topics st ON st.topic_id=t.id WHERE st.source_id=?", (source_id,)).fetchall()


def create_source(filename: str, source_kind: str, content: bytes, original_path: str | None = None, url: str | None = None) -> tuple[str, str | None]:
    digest = hashlib.sha256(content).hexdigest()
    with db_connection() as db:
        duplicate = db.execute("SELECT id FROM sources WHERE content_hash=?", (digest,)).fetchone()
        if duplicate:
            return duplicate["id"], "duplicate"
        source_id = str(uuid.uuid4())
        suffix = Path(filename).suffix
        managed_path = LIBRARY_DIR / f"{source_id}{suffix}"
        managed_path.write_bytes(content)
        timestamp = now()
        db.execute("""INSERT INTO sources(id, filename, source_kind, original_path, managed_path, url, mime_type, content_hash, status, created_at, updated_at)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)""", (source_id, filename, source_kind, original_path, str(managed_path), url, mimetypes.guess_type(filename)[0], digest, timestamp, timestamp))
        return source_id, None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"name": APP_NAME, "status": "ok", "data_dir": str(DATA_DIR)}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    with db_connection() as db:
        values = {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM settings")}
    return {"base_url": values.get("base_url", "https://api.openai.com/v1"), "model": values.get("model", "gpt-4o-mini"), "api_key_configured": api_key_available()}


@app.put("/api/settings")
def save_settings(payload: SettingsInput) -> dict[str, Any]:
    with db_connection() as db:
        for key, value in {"base_url": payload.base_url.rstrip("/"), "model": payload.model}.items():
            db.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    vault_saved = None
    if payload.api_key:
        vault_saved = store_api_key(payload.api_key)
    return {"ok": True, "vault_saved": vault_saved, "api_key_configured": api_key_available()}


@app.post("/api/captures/files")
async def import_files(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)) -> list[dict[str, Any]]:
    results = []
    for upload in files:
        filename = upload.filename or "Untitled"
        content = await upload.read()
        source_id, duplicate = create_source(filename, "file", content)
        if duplicate:
            results.append({"id": source_id, "status": "duplicate"})
            continue
        job_id = str(uuid.uuid4())
        with db_connection() as db:
            db.execute("INSERT INTO processing_jobs(id, source_id, status, stage, created_at, updated_at) VALUES (?, ?, 'queued', 'waiting', ?, ?)", (job_id, source_id, now(), now()))
        background_tasks.add_task(process_source, source_id, job_id)
        results.append({"id": source_id, "job_id": job_id, "status": "queued"})
    return results


@app.post("/api/captures/text")
def import_text(payload: TextCapture, background_tasks: BackgroundTasks) -> dict[str, Any]:
    filename = f"{payload.title.strip()[:100]}.md"
    source_id, duplicate = create_source(filename, "text", payload.content.encode("utf-8"))
    if duplicate:
        return {"id": source_id, "status": "duplicate"}
    job_id = str(uuid.uuid4())
    with db_connection() as db:
        db.execute("INSERT INTO processing_jobs(id, source_id, status, stage, created_at, updated_at) VALUES (?, ?, 'queued', 'waiting', ?, ?)", (job_id, source_id, now(), now()))
    background_tasks.add_task(process_source, source_id, job_id)
    return {"id": source_id, "job_id": job_id, "status": "queued"}


@app.post("/api/captures/url")
def import_url(payload: UrlCapture, background_tasks: BackgroundTasks) -> dict[str, Any]:
    content = f"# {payload.title}\n\nURL: {payload.url}\n\n{payload.context}".encode("utf-8")
    source_id, duplicate = create_source(f"{payload.title.strip()[:100]}.md", "url", content, url=payload.url)
    if duplicate:
        return {"id": source_id, "status": "duplicate"}
    job_id = str(uuid.uuid4())
    with db_connection() as db:
        db.execute("INSERT INTO processing_jobs(id, source_id, status, stage, created_at, updated_at) VALUES (?, ?, 'queued', 'waiting', ?, ?)", (job_id, source_id, now(), now()))
    background_tasks.add_task(process_source, source_id, job_id)
    return {"id": source_id, "job_id": job_id, "status": "queued"}


@app.get("/api/sources")
def list_sources() -> list[dict[str, Any]]:
    with db_connection() as db:
        rows = db.execute("SELECT * FROM sources ORDER BY created_at DESC").fetchall()
    return [source_payload(row) for row in rows]


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    with db_connection() as db:
        return [dict(row) for row in db.execute("SELECT * FROM processing_jobs ORDER BY created_at DESC LIMIT 100")]


@app.post("/api/sources/{source_id}/retry")
def retry_source(source_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    with db_connection() as db:
        if not db.execute("SELECT 1 FROM sources WHERE id=?", (source_id,)).fetchone():
            raise HTTPException(404, "Source not found")
        job_id = str(uuid.uuid4())
        db.execute("UPDATE sources SET status='queued', error_message=NULL, updated_at=? WHERE id=?", (now(), source_id))
        db.execute("INSERT INTO processing_jobs(id, source_id, status, stage, created_at, updated_at) VALUES (?, ?, 'queued', 'waiting', ?, ?)", (job_id, source_id, now(), now()))
    background_tasks.add_task(process_source, source_id, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str) -> dict[str, bool]:
    with db_connection() as db:
        row = db.execute("SELECT managed_path FROM sources WHERE id=?", (source_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Source not found")
        db.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id=?)", (source_id,))
        db.execute("DELETE FROM sources WHERE id=?", (source_id,))
    Path(row["managed_path"]).unlink(missing_ok=True)
    return {"deleted": True}


@app.get("/api/sources/{source_id}/open")
def open_source(source_id: str):
    with db_connection() as db:
        row = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Source not found")
    return FileResponse(row["managed_path"], filename=row["filename"], media_type=row["mime_type"] or "application/octet-stream")


@app.get("/api/topics")
def list_topics() -> list[dict[str, Any]]:
    with db_connection() as db:
        rows = db.execute("""SELECT t.*, COUNT(st.source_id) source_count, n.status note_status
                              FROM topics t LEFT JOIN source_topics st ON st.topic_id=t.id
                              LEFT JOIN notes n ON n.topic_id=t.id GROUP BY t.id ORDER BY source_count DESC, t.name""").fetchall()
    return [dict(row) for row in rows]


@app.get("/api/topics/{topic_id}")
def get_topic(topic_id: str) -> dict[str, Any]:
    with db_connection() as db:
        topic = db.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
        if not topic:
            raise HTTPException(404, "Topic not found")
        sources = db.execute("SELECT s.* FROM sources s JOIN source_topics st ON st.source_id=s.id WHERE st.topic_id=?", (topic_id,)).fetchall()
        draft = db.execute("""SELECT r.* FROM notes n JOIN note_revisions r ON r.note_id=n.id
                              WHERE n.topic_id=? AND r.status='draft' ORDER BY r.created_at DESC LIMIT 1""", (topic_id,)).fetchone()
        current = db.execute("""SELECT r.* FROM notes n JOIN note_revisions r ON r.id=n.current_revision_id
                                WHERE n.topic_id=?""", (topic_id,)).fetchone()
    return {"topic": dict(topic), "sources": [source_payload(row) for row in sources], "draft": dict(draft) if draft else None, "current_note": dict(current) if current else None}


@app.post("/api/topics/{topic_id}/note-drafts")
def generate_note_draft(topic_id: str) -> dict[str, Any]:
    try:
        return create_note_draft(topic_id)
    except KeyError:
        raise HTTPException(404, "Topic not found")
    except ValueError as error:
        raise HTTPException(422, str(error))


@app.post("/api/note-revisions/{revision_id}/approve")
def approve_note(revision_id: str) -> dict[str, bool]:
    with db_connection() as db:
        revision = db.execute("SELECT * FROM note_revisions WHERE id=?", (revision_id,)).fetchone()
        if not revision:
            raise HTTPException(404, "Note draft not found")
        db.execute("UPDATE note_revisions SET status='superseded' WHERE note_id=? AND status='approved'", (revision["note_id"],))
        db.execute("UPDATE note_revisions SET status='approved', reviewed_at=? WHERE id=?", (now(), revision_id))
        db.execute("UPDATE notes SET status='approved', current_revision_id=?, updated_at=? WHERE id=?", (revision_id, now(), revision["note_id"]))
        topic = db.execute("SELECT t.name FROM topics t JOIN notes n ON n.topic_id=t.id WHERE n.id=?", (revision["note_id"],)).fetchone()
    safe_name = re.sub(r"[^A-Za-z0-9 _-]", "", topic["name"]).strip() or "Untitled"
    (NOTES_DIR / f"{safe_name}.md").write_text(revision["body"], encoding="utf-8")
    return {"approved": True}


@app.post("/api/note-revisions/{revision_id}/reject")
def reject_note(revision_id: str) -> dict[str, bool]:
    with db_connection() as db:
        row = db.execute("SELECT note_id FROM note_revisions WHERE id=?", (revision_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Note draft not found")
        db.execute("UPDATE note_revisions SET status='rejected', reviewed_at=? WHERE id=?", (now(), revision_id))
    return {"rejected": True}


@app.get("/api/search")
def search(q: str = "") -> list[dict[str, Any]]:
    if not q.strip():
        return []
    return [{"chunk_id": row["id"], "source_id": row["source_id"], "filename": row["filename"], "anchor": row["anchor"], "text": row["text"]} for row in query_rows(q)]


@app.post("/api/questions")
def answer_question(payload: QuestionInput) -> dict[str, Any]:
    rows = query_rows(payload.question, limit=5)
    if not rows:
        answer = "I couldn't find supporting material in your Ren library yet. Import a relevant source or try different words."
        citations: list[dict[str, str]] = []
    else:
        excerpts = []
        citations = []
        for row in rows:
            excerpts.append(row["text"][:650])
            citations.append({"source_id": row["source_id"], "filename": row["filename"], "anchor": row["anchor"] or "Source"})
        ai_answer = provider_answer(payload.question, rows)
        answer = ai_answer or ("Based on your imported sources:\n\n" + "\n\n".join(excerpts[:3]))
    with db_connection() as db:
        db.execute("INSERT INTO questions(id, question, answer, created_at) VALUES (?, ?, ?, ?)", (str(uuid.uuid4()), payload.question, answer, now()))
    return {"answer": answer, "citations": citations, "mode": "ai-grounded" if rows and api_key_available() and not answer.startswith("Based on") else "extractive"}


@app.get("/api/export")
def export_data() -> dict[str, Any]:
    with db_connection() as db:
        sources = [dict(row) for row in db.execute("SELECT id, filename, source_kind, original_path, url, status, created_at FROM sources")]
        topics = [dict(row) for row in db.execute("SELECT * FROM topics")]
        notes = [dict(row) for row in db.execute("SELECT * FROM note_revisions WHERE status='approved'")]
    return {"exported_at": now(), "sources": sources, "topics": topics, "approved_note_revisions": notes}

