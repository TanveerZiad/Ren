# Ren architecture

The React desktop UI speaks only to a FastAPI service on `127.0.0.1`. The service owns local storage and processing.

`Capture -> managed immutable source copy -> extraction -> chunks + FTS -> topic suggestions -> note draft -> user approval`

SQLite is the source of truth for metadata, processing jobs, source-topic relations, revisions, and question history. Imported files live under Ren's local app-data directory. Generated notes are versioned Markdown in that same directory.

The AI adapter is optional and OpenAI-compatible. No upload or AI call is made unless an API key has been explicitly configured. Every generated note and answer uses source identifiers; v0.1 has a deterministic local fallback so a missing provider never blocks capture. Processing creates a pending draft for newly discovered topics, never an approved note.

