OP here — big update since I first posted this.

Engram went from ~2,300 lines to 6,200+. Main changes:

- Replaced the flat 30-day exponential decay with FSRS-6 (same spaced repetition algorithm behind modern Anki, 21 parameters, trained on hundreds of millions of reviews). Every memory access is an implicit review now — stuff agents actually use builds stability, stuff they don't fades out naturally. Power-law forgetting curve instead of exponential.

- Each memory tracks dual strengths (Bjork & Bjork 1992) — storage strength that accumulates and never decays, and retrieval strength that does. A memory can be deeply encoded but temporarily hard to retrieve. Recall scoring got noticeably better after this.

- Moved from SQLite to libsql for native vector search. Embeddings are FLOAT32(384) with an ANN index now — vector_top_k() instead of cosine similarity loops in JS. O(log n) vs O(n).

- Conversation storage and search — full chat logs linked to memory episodes. Useful for tracing why a memory exists.

- Episodic memory — group memories into sessions/episodes.

Everything from before still works — local embeddings, auto-linking, versioning, dedup, contradiction detection, time-travel, reflections, graph viz, multi-tenant, MCP server, TypeScript/Python SDKs. Still docker compose up, still MIT.

Demo: https://demo.engram.lol/gui
