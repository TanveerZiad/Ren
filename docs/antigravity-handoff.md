# Continuing Ren in Antigravity

Open the repository root, not the `frontend` or `backend` folder alone. Start from `README.md`, then run `run-dev.ps1` after dependencies are installed.

Use one vertical slice per session. A good request to an implementation agent is:

> Read `docs/architecture.md` and `docs/definition-of-done.md`. Implement only the selected backlog item, preserve the local-first model, add or update tests, and do not change original source files.

Before accepting generated code, check these invariants:

1. Original files are never modified or deleted.
2. An AI-produced note stays a draft until the user approves it.
3. Answers and notes show their source provenance.
4. API keys are never persisted in SQLite, exports, or logs.

Do not start knowledge graphs, browser extensions, reminders, or autonomous agent actions until the current import → note → answer workflow has been used with real files for at least two weeks.

