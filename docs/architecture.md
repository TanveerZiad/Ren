# Ren architecture: small MVP

Ren is a local-only pipeline:

`PDF / TXT / MD -> managed source copy -> text extraction -> topic suggestions -> knowledge cards -> review queue`

The FastAPI service owns the SQLite database and all local files. The React interface only displays Inbox, My Knowledge, and Remember views.

Each generated card stores its source, a topic label, review count, last-seen timestamp, and next-review timestamp. A reviewed card returns after 1, 3, 7, 14, then 30 days. A card can be archived when it is not useful.

The source file is permanent and untouched. Cards are derived, replaceable knowledge. No AI key, embedding index, cloud account, or chat model is required for this MVP.
