GitHub: https://github.com/zanfiel/engram

Live demo: https://demo.engram.lol/gui (password: demo)

**Update post** — Engram has grown a lot since I first shared it. Went from ~2,300 lines to 6,200+. Here's what's new:

- **FSRS-6 spaced repetition** — replaced the old flat 30-day decay. Memories now decay on a power-law curve (same algorithm behind modern Anki). Every access counts as an implicit review, so frequently used memories stick around and unused ones fade naturally
- **Dual-strength memory model** — each memory tracks storage strength (deep encoding, never decays) and retrieval strength (current accessibility, decays over time). Based on Bjork & Bjork 1992. Makes recall scoring way more realistic
- **Native vector search via libsql** — moved from SQLite to libsql. Embeddings stored as FLOAT32(384) with ANN indexing. Search is O(log n) now instead of brute-force cosine similarity over everything
- **Conversation storage + search** — store full agent chat logs, search across messages, link to memory episodes
- **Episodic memory** — group memories into sessions/episodes

Everything from before is still there — local embeddings, auto-linking, versioning, dedup, four-layer recall, contradiction detection, time-travel queries, reflections, graph viz, multi-tenant, TypeScript/Python SDKs, MCP server.

Still one file, still `docker compose up`, still MIT.
