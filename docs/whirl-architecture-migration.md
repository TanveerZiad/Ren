# Ren platform foundation

Ren remains a local-first knowledge inbox. The supplied Whirl project is being used as an architectural reference, not copied into this repository: Whirl is a multi-tenant Next/Convex service and Ren is currently a local FastAPI, SQLite, React desktop application.

## What this update adds

- Persistent chat threads and message history in Ren's local SQLite database.
- OpenRouter-compatible model settings, including a model identifier and three thinking presets.
- Source-grounded chat: Ren retrieves local chunks before calling the configured provider and returns citations with the response.
- Incognito threads that do not receive long-term memories as context.
- Local schemas and API foundations for folders, memories, artifacts, integration registrations, and queued messages.
- A clear capability endpoint so the interface can avoid presenting unavailable features as working.

## What remains intentionally unimplemented

The following need distinct security and product work before they can be called complete: actual MCP/OAuth integration execution, web research, encrypted password-locked chats, public artifact sharing, background queue delivery, voice input, image generation, installable skills, PWA packaging, and remote collaboration.

## Why this route

Ren's core promise is still **Dump → Organize → Remind**. The chat layer gives those locally imported cards and source chunks a useful interface, but it does not replace the knowledge inbox. Moving wholesale to Whirl's stack would require credentialed Convex, Clerk, OpenRouter, and other service configuration; that is a separate migration rather than a safe incremental update.

## Local API surface added

- `GET/POST /api/chat/threads`
- `GET/DELETE /api/chat/threads/{id}`
- `POST /api/chat/threads/{id}/messages`
- `GET/POST/DELETE /api/memories`
- `GET/POST /api/chat/threads/{id}/artifacts`
- `GET/POST /api/folders`
- `GET/POST /api/integrations`
- `GET /api/platform/capabilities`

All persisted data stays in Ren's application-data directory. The provider API key is handled through the existing operating-system credential-store flow and is never placed in the repository.
