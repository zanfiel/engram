# Show HN: Engram – Open source persistent memory for AI agents (Bun + SQLite)

I built Engram because every AI agent I worked with had amnesia. Between sessions, everything was gone. I needed something that could store what agents learn, find it by meaning later, and do it without requiring a vector database, an OpenAI key, or any external service.

**What it does:** Engram gives AI agents long-term memory. Store memories, search them semantically, and recall relevant context automatically — all backed by a single SQLite file.

**How it works:**

```typescript
const engram = new Engram({ url: "http://localhost:4200" });

// Store what the agent learns
await engram.store("User prefers dark mode and Vim keybindings", {
  category: "decision",
  importance: 8,
});

// Later, in a new session
const context = await engram.recall("setting up editor");
// → Returns the dark mode + Vim preference automatically
```

**What makes it different from Mem0/Supermemory:**

- **Zero external dependencies.** Embeddings run locally (MiniLM-L6, 384-dim). No OpenAI key needed. No vector database. Just Bun + SQLite.
- **Auto-linking.** New memories automatically connect to related ones above 0.75 cosine similarity, forming a knowledge graph you can visualize.
- **Versioning.** Update a memory and the full version chain is preserved. Only the latest surfaces in search.
- **Auto-deduplication.** Detects near-identical memories and merges them. Source count tracks frequency — signal, not noise.
- **Auto-forget.** Set memories to expire. Low-importance memories decay. Critical ones persist.
- **Four-layer recall.** Instead of just similarity search, recall combines static facts + semantic matches + high-importance memories + recent activity into an optimal context window.
- **WebGL graph visualization.** Built-in GUI to explore your memory space as an interactive galaxy.

**Stack:** Bun, SQLite (FTS5), Xenova/transformers.js (local embeddings), single TypeScript file (2,300 lines).

**Deploy:** `docker compose up -d` — that's it.

SDKs for TypeScript and Python. Full OpenAPI spec. Multi-tenant with API keys if you need it.

GitHub: https://github.com/zanfiel/engram
Site: https://engram.lol

I'm using this in production for my own AI agents. Happy to answer questions about the architecture, the recall algorithm, or anything else.
