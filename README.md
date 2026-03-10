<div align="center">

# Engram

### Persistent memory for AI agents

Store, search, recall, and link memories with automatic embeddings,
fact extraction, versioning, deduplication, and graph visualization.

[Quick Start](#quick-start) · [API Reference](#api-reference) · [SDKs](#sdks) · [Self-Host](#self-hosting) · [GUI](#gui)

</div>

---

## What is Engram?

Engram gives your AI agents **long-term memory**. Instead of losing context between sessions, agents store what they learn and recall it when relevant — automatically.

```ts
import { Engram } from "@engram/sdk";

const engram = new Engram({ url: "http://localhost:4200", apiKey: "eg_..." });

// Store what the agent learns
await engram.store("User prefers dark mode and uses Vim keybindings", {
  category: "decision",
  importance: 8,
});

// Later, in a new session — recall relevant context
const context = await engram.recall("setting up the user's editor");
// → Returns the dark mode + Vim preference automatically
```

**Key features:**

- 🧠 **FSRS-6 spaced repetition** — cognitive science-backed memory decay using power-law forgetting curves (ported from open-spaced-repetition)
- 🔍 **Native vector search** — libsql FLOAT32 vector columns with ANN index (`vector_top_k`) — O(log n) instead of brute-force
- 💪 **Dual-strength memory model** — Bjork & Bjork (1992) storage strength (never decays) + retrieval strength (decays via power law)
- 🧬 **Semantic + full-text hybrid search** — find memories by meaning, keywords, or both
- 🔗 **Auto-linking** — memories automatically connect, forming a knowledge graph
- 📊 **Graph visualization** — explore your memory space in a WebGL galaxy
- 🔄 **Versioning** — update memories without losing history
- 🧹 **Auto-deduplication** — detects and merges near-duplicate memories
- ⏰ **Implicit spaced repetition** — every access is an FSRS review, building stability over time
- 🔍 **Fact extraction & auto-tagging** — LLM extracts facts, classifies, tags
- 💬 **Conversation extraction** — feed chat logs, get structured memories
- ⚡ **Contradiction detection** — find and resolve conflicting memories
- ⏪ **Time-travel queries** — query what you knew at any point in time
- 🎯 **Smart context builder** — token-budget-aware RAG context assembly
- 💭 **Reflections** — periodic meta-analysis that becomes searchable memory
- 🧬 **Derived memories** — inference engine finds patterns across memories
- 🗜️ **Auto-consolidation** — summarize large memory clusters automatically
- 🏆 **LLM reranker** — search results reranked for semantic precision
- 👥 **Multi-tenant** — isolated memory per user with API keys
- 📦 **Spaces, tags, episodes** — organize memories into named collections
- 🧩 **Entities & projects** — track people, servers, tools, projects
- 📬 **Webhooks & digests** — event hooks + scheduled HMAC-signed summaries
- 🔄 **Sync & import** — cross-instance sync, import from Mem0 / Supermemory
- 📥 **URL ingest** — extract facts from web pages or text blobs
- 🛠️ **MCP server & CLI** — IDE integrations + terminal workflows
- 🐳 **One-command deploy** — `docker compose up`

---

## What's New in v5.0

### FSRS-6 Spaced Repetition (replaces exponential decay)

Engram v5.0 replaces the simple 30-day half-life exponential decay with **FSRS-6** (Free Spaced Repetition Scheduler), a 21-parameter algorithm trained on millions of Anki reviews. Memory decay now follows a **power-law forgetting curve** — more accurate than exponential models.

**Key formula:** `R = (1 + factor × t/S)^(-w₂₀)` where S is stability (days until 90% recall probability).

Every memory access is an implicit FSRS review. Frequently accessed memories build stability and resist forgetting. Rarely accessed memories decay naturally. The **dual-strength model** (Bjork & Bjork 1992) tracks both:
- **Storage strength** — increases with each access, never decays (long-term consolidation)
- **Retrieval strength** — decays via power law, reset on access (current accessibility)

New endpoints: `POST /fsrs/review`, `GET /fsrs/state?id=N`, `POST /fsrs/init`

### Native Vector Search (replaces brute-force cosine similarity)

Engram v5.0 uses **libsql** with native `FLOAT32(384)` vector columns and ANN indexing. Search is now `O(log n)` via `vector_top_k()` instead of `O(n)` JavaScript cosine similarity loops over all embeddings.

```sql
-- What Engram does under the hood
SELECT rowid FROM vector_top_k('memories_vec_idx', vector(?), 25)
```

The ONNX embedding model (Xenova/all-MiniLM-L6-v2) still runs locally for generating embeddings — no external API needed. libsql just stores and indexes them natively.

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/zanfiel/engram.git
cd engram
cp .env.example .env
# Edit .env — set ENGRAM_GUI_PASSWORD

docker compose up -d
```

Engram is now running at `http://localhost:4200`.

### From source (Bun)

```bash
git clone https://github.com/zanfiel/engram.git
cd engram
bun install
ENGRAM_GUI_PASSWORD=your-password bun run server.ts
```

### Create an API key

```bash
curl -X POST http://localhost:4200/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "my-agent", "scopes": "read,write"}'
```

Save the returned `eg_...` key — it's shown only once.

---

## SDKs

### TypeScript / JavaScript

```bash
npm install @engram/sdk
```

```ts
import { Engram } from "@engram/sdk";

const engram = new Engram({
  url: "http://localhost:4200",
  apiKey: "eg_your_key_here",
});

// Store
const result = await engram.store("Deployed v2.3 to production", {
  category: "task",
  importance: 7,
});

// Search
const results = await engram.search("deployment history");

// Recall (optimized for agent context)
const context = await engram.recall("What was deployed recently?");

// List
const recent = await engram.list({ limit: 10, category: "task" });

// Update (creates new version)
await engram.update(result.id, "Deployed v2.4 to production");

// FSRS state — check memory health
const fsrs = await engram.fsrsState(result.id);
// → { retrievability: 0.95, stability: 4.2, next_review_days: 4 }

// Spaces
engram.useSpace("project-alpha");
await engram.store("Sprint 3 completed", { category: "task" });
```

### Python

```bash
pip install engram-sdk
```

```python
from engram import Engram

client = Engram("http://localhost:4200", api_key="eg_your_key_here")

# Store
result = client.store("User prefers dark mode", category="decision", importance=8)

# Search
memories = client.search("user preferences", limit=5)
for m in memories:
    print(f"[{m.category}] {m.content} (score: {m.score:.2f})")

# Recall
context = client.recall("What does the user like?")

# Export everything
data = client.export()
```

### cURL

```bash
# Store
curl -X POST http://localhost:4200/store \
  -H "Authorization: Bearer eg_your_key" \
  -H "Content-Type: application/json" \
  -d '{"content": "Server migrated to new IP", "category": "state", "importance": 7}'

# Search
curl -X POST http://localhost:4200/search \
  -H "Authorization: Bearer eg_your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "server migration", "limit": 5}'

# Recall
curl -X POST http://localhost:4200/recall \
  -H "Authorization: Bearer eg_your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "infrastructure changes"}'

# FSRS state
curl http://localhost:4200/fsrs/state?id=42 \
  -H "Authorization: Bearer eg_your_key"
# → {"id":42,"retrievability":0.92,"next_review_days":7,"fsrs_stability":8.3,...}
```

---

## API Reference

### Authentication

All endpoints accept `Authorization: Bearer eg_...` header. Without a key, requests default to the owner account (single-user mode).

Use `X-Space: space-name` header to scope operations to a specific memory space.

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/store` | Store a memory |
| `POST` | `/search` | Semantic + full-text search |
| `POST` | `/recall` | Contextual recall (agent-optimized) |
| `POST` | `/context` | Smart context builder (token-budget RAG) |
| `GET` | `/list` | List recent memories |
| `GET` | `/profile` | User profile (static facts + recent) |
| `GET` | `/graph` | Full memory graph (nodes + edges) |

### Memory Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memory/:id/update` | Create new version |
| `POST` | `/memory/:id/forget` | Soft delete |
| `POST` | `/memory/:id/archive` | Archive (hidden from recall) |
| `POST` | `/memory/:id/unarchive` | Restore from archive |
| `DELETE` | `/memory/:id` | Permanent delete |
| `GET` | `/versions/:id` | Version chain for a memory |

### FSRS-6 Spaced Repetition

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/fsrs/review` | Manual review (grade 1-4: Again/Hard/Good/Easy) |
| `GET` | `/fsrs/state?id=N` | Retrievability, stability, next review interval |
| `POST` | `/fsrs/init` | Backfill FSRS state for all memories |
| `POST` | `/decay/refresh` | Recalculate all decay scores |
| `GET` | `/decay/scores` | View decay scores + FSRS state |

### Intelligence

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/add` | Extract memories from conversations |
| `POST` | `/ingest` | Extract facts from URLs or text |
| `POST` | `/derive` | Generate inferred memories |
| `POST` | `/reflect` | Generate period reflection |
| `GET` | `/reflections` | List past reflections |
| `GET` | `/contradictions` | Find conflicting memories |
| `POST` | `/contradictions/resolve` | Resolve a contradiction |
| `POST` | `/timetravel` | Query memory state at a past time |

### Organization

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/tags` | List all tags |
| `POST` | `/tags/search` | Search by tags |
| `POST` | `/episodes` | Create episode |
| `GET` | `/episodes` | List episodes |
| `POST` | `/entities` | Create entity |
| `GET` | `/entities` | List entities |
| `POST` | `/projects` | Create project |
| `GET` | `/projects` | List projects |

### Data & Sync

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/export` | Export all memories + links (JSON/JSONL) |
| `POST` | `/import` | Bulk import memories |
| `POST` | `/import/mem0` | Import from Mem0 |
| `POST` | `/import/supermemory` | Import from Supermemory |
| `GET` | `/sync/changes` | Get changes since timestamp |
| `POST` | `/sync/receive` | Receive synced changes |

### Platform

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhooks` | Create webhook |
| `GET` | `/webhooks` | List webhooks |
| `POST` | `/digests` | Create scheduled digest |
| `GET` | `/digests` | List digests |
| `POST` | `/digests/send` | Manually trigger a digest |
| `POST` | `/pack` | Pack memories into token budget |
| `GET` | `/prompt` | Generate prompt template |

### Auth & Multi-tenant

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/users` | Create user (admin) |
| `GET` | `/users` | List users (admin) |
| `POST` | `/keys` | Create API key |
| `GET` | `/keys` | List API keys |
| `DELETE` | `/keys/:id` | Revoke key |
| `POST` | `/spaces` | Create space |
| `GET` | `/spaces` | List spaces |
| `DELETE` | `/spaces/:id` | Delete space |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (30+ feature flags) |
| `GET` | `/stats` | Detailed statistics |

---

## How It Works

### Memory Lifecycle

1. **Store** — Memory content is embedded using MiniLM (384-dim vectors) and stored in libsql with native FLOAT32 vector indexing and FTS5 full-text search.

2. **Auto-link** — New memories are compared against existing ones via ANN vector search. Memories above 0.7 cosine similarity are linked with typed relationships (similarity, updates, extends, contradicts, caused_by, prerequisite_for).

3. **FSRS-6 initialization** — Each new memory gets initial FSRS state: stability, difficulty, storage strength, retrieval strength. The power-law forgetting curve starts tracking retrievability.

4. **Fact extraction** — If an LLM is configured, Engram analyzes new memories, extracts static facts, auto-tags with keywords, classifies importance, and detects relationships to existing memories.

5. **Recall** — Four retrieval strategies combined: static facts (always), semantic matches (ANN vector search), high-importance (weighted by FSRS retrievability), recent (temporal). Every recalled memory gets an implicit FSRS review, building stability.

6. **Spaced repetition** — Each access is an FSRS-6 review graded as "Good". Archived/forgotten memories receive an "Again" grade. Stability grows with successful recalls — frequently accessed memories can have stability measured in months or years.

7. **Dual-strength decay** — Storage strength (0-10) accumulates over time, representing deep consolidation. Retrieval strength (0-1) decays via power law, representing current accessibility. Together they produce a retention score: `0.7 × retrieval + 0.3 × (storage/10)`.

8. **Contradiction detection** — Scans for memories that conflict. LLM verification eliminates false positives. Contradictions can be resolved by keeping one side, both, or merging.

9. **Consolidation** — Large clusters of related memories get summarized into a single dense memory. Originals are archived, links preserved.

10. **Reflection** — On-demand meta-analysis generates insights about themes, progress, and patterns. Reflections become searchable memories themselves.

### Architecture

```
┌─────────────────────────────────────────────┐
│                  Engram Server               │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  FSRS-6  │  │  Vector  │  │  FTS5    │  │
│  │  Engine   │  │  Search  │  │  Search  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │        │
│  ┌────┴──────────────┴──────────────┴────┐  │
│  │         libsql (SQLite + vectors)     │  │
│  │    FLOAT32(384) + ANN index + FTS5    │  │
│  └───────────────────────────────────────┘  │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  MiniLM  │  │  LLM     │  │  Graph   │  │
│  │  Embedder │  │  (opt.)  │  │  Engine  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

- **Runtime:** Bun
- **Database:** libsql (SQLite fork with native vector search)
- **Embeddings:** Xenova/all-MiniLM-L6-v2 (384-dim, runs locally)
- **Vector index:** `FLOAT32(384)` column + `libsql_vector_idx` ANN
- **LLM:** Optional, for fact extraction / reranking / consolidation
- **Decay:** FSRS-6 (21-parameter power-law forgetting curve)

---

## GUI

Engram includes a WebGL graph visualization at `/gui`. Login with your `ENGRAM_GUI_PASSWORD`.

**Features:**
- Interactive galaxy-style memory graph
- Click memories to view details
- Create, edit, archive, and delete memories
- Semantic search with hybrid client/API matching
- Category filters and sorting
- Keyboard shortcuts (L=list, N=new, Z=fit, C=center, arrows=navigate)
- Export data

---

## Self-Hosting

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_PORT` / `ZANMEMORY_PORT` | `4200` | Server port |
| `ENGRAM_HOST` / `ZANMEMORY_HOST` | `0.0.0.0` | Bind address |
| `ENGRAM_GUI_PASSWORD` / `MEGAMIND_GUI_PASSWORD` | `changeme` | GUI login password |
| `LLM_URL` | — | OpenAI-compatible API URL |
| `LLM_API_KEY` | — | API key for LLM |
| `LLM_MODEL` | — | Model name (e.g., `gpt-4o`, `claude-sonnet-4-20250514`) |

### Storage

All data lives in a single libsql database (`data/memory.db`). The file includes both the SQLite data and vector index. Back it up however you back up files.

### Reverse Proxy

```nginx
server {
    server_name memory.example.com;

    location / {
        proxy_pass http://127.0.0.1:4200;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Comparison

| Feature | Engram | Mem0 | Supermemory |
|---------|--------|------|-------------|
| **Spaced repetition (FSRS-6)** | ✅ | ❌ | ❌ |
| **Native vector search (ANN)** | ✅ | ❌ | ❌ |
| **Dual-strength memory model** | ✅ | ❌ | ❌ |
| Semantic search | ✅ | ✅ | ✅ |
| Local embeddings (no API) | ✅ | ❌ | ❌ |
| Full-text search (FTS5) | ✅ | ❌ | ❌ |
| Graph visualization | ✅ | ❌ | ✅ |
| Memory versioning | ✅ | ❌ | ❌ |
| Auto-deduplication | ✅ | ❌ | ❌ |
| Auto-forget / TTL | ✅ | ❌ | ❌ |
| Contradiction detection | ✅ | ❌ | ❌ |
| Time-travel queries | ✅ | ❌ | ❌ |
| Smart context builder (RAG) | ✅ | ❌ | ❌ |
| Reflections | ✅ | ❌ | ❌ |
| Derived memories | ✅ | ❌ | ❌ |
| Auto-consolidation | ✅ | ❌ | ❌ |
| LLM reranker | ✅ | ❌ | ❌ |
| Fact extraction + auto-tagging | ✅ | ✅ | ❌ |
| Conversation extraction | ✅ | ✅ | ❌ |
| Multi-tenant + API keys | ✅ | ✅ | ❌ |
| Spaces / collections | ✅ | ❌ | ✅ |
| Entities & projects | ✅ | ❌ | ❌ |
| Episodic memory | ✅ | ❌ | ❌ |
| Conversation log + search | ✅ | ❌ | ❌ |
| Webhooks & digests | ✅ | ❌ | ❌ |
| Cross-instance sync | ✅ | ❌ | ❌ |
| URL ingest | ✅ | ❌ | ❌ |
| Import from Mem0 / Supermemory | ✅ | — | — |
| Self-hosted | ✅ | ✅ | ✅ |
| SQLite-based (zero deps) | ✅ | ❌ | ❌ |

---

## License

MIT
