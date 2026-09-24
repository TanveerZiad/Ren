# Ren feature definition of done

- The feature preserves original user files and works from Ren-managed copies.
- Empty, loading, failure, retry, and cancellation-adjacent states are understandable.
- Any derived note or AI answer carries source provenance.
- The API key is not written to SQLite, exported data, logs, or the UI after saving.
- The feature works without an AI key whenever a local fallback is possible.
- A small manual test is documented or automated before the feature is considered complete.
