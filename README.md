<div align="center">

# Engram

### Persistent memory for AI agents

Store, search, recall, and link memories with automatic embeddings,
fact extraction, versioning, deduplication, and graph visualization.

[Quick Start](#quick-start) · [API Reference](#api-reference) · [SDKs](#sdks) · [MCP Server](#mcp-server) · [CLI](#cli) · [Self-Host](#self-hosting) · [GUI](#gui)

</div>

---

## What is Engram?

Engram gives your AI agents **long-term memory**. Instead of losing context between sessions, agents store what they learn and recall it when relevant — automatically.

```bash
# Store what the agent learns
curl -X POST http://localhost:4200/store \
  -H "Authorization: Bearer eg_your_key" \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode and uses Vim keybindings", "category": "decision", "importance": 8}'

# Later, in a new session — recall relevant context
curl -X POST http://localhost:4200/recall \
  -H "Authorization: Bearer eg_your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "setting up the user editor"}'
# → Returns the dark mode + Vim preference automatically
```

**Key features:**

- 🧠 **FSRS-6 spaced repetition** — cognitive science-backed memory decay using power-law forgetting curves (ported from [open-spaced-repetition](https://github.com/open-spaced-repetition/fsrs4anki))
- 💪 **Dual-strength memory model** — Bjork & Bjork (1992) storage strength (never decays) + retrieval strength (decays via power law)
- 🧬 **Hybrid semantic + full-text search** — MiniLM embeddings (384-dim, runs locally) combined with FTS5 full-text search
- 🔗 **Auto-linking** — memories automatically connect via cosine similarity, forming a knowledge graph
- 📊 **Graph visualization** — explore your memory space in a WebGL galaxy
- 🔄 **Versioning** — update memories without losing history
- 🧹 **Auto-deduplication** — detects and merges near-duplicate memories
- ⏰ **Implicit spaced repetition** — every access is an FSRS review, building stability over time
- 🔍 **Fact extraction & auto-tagging** — LLM extracts facts, classifies, tags (optional, requires LLM)
- 💬 **Conversation extraction** — feed chat logs, get structured memories
- ⚡ **Contradiction detection** — find and resolve conflicting memories
- ⏪ **Time-travel queries** — query what you knew at any point in time
- 🎯 **Smart context builder** — token-budget-aware RAG context assembly
- 💭 **Reflections** — periodic meta-analysis that becomes searchable memory
- 🧬 **Derived memories** — inference engine finds patterns across memories
- 🗜️ **Auto-consolidation** — summarize large memory clusters automatically
- 🏆 **LLM reranker** — search results reranked for semantic precision (optional)
- 👥 **Multi-tenant** — isolated memory per user with API keys
- 📦 **Spaces, tags, episodes** — organize memories into named collections
- 🧩 **Entities & projects** — track people, servers, tools, projects
- 📬 **Webhooks & digests** — event hooks + scheduled HMAC-signed summaries
- 🔄 **Sync & import** — cross-instance sync, import from Mem0 / Supermemory
- 📥 **URL ingest** — extract facts from web pages or text blobs
- 🛠️ **MCP server** — JSON-RPC 2.0 stdio transport for Claude Desktop, Cursor, Windsurf
- ⌨️ **CLI** — full-featured command-line interface (`engram store`, `engram search`, etc.)
- 📥 **Review queue / inbox** — auto-detected memories land in review; explicit stores bypass
- 🔒 **Security hardening** — auth required by default, body/content limits, IP allowlists, timing-safe auth
- 📋 **Audit trail** — every mutation logged (who, what, when, from where)
- 📊 **Structured JSON logging** — configurable log levels, request IDs, zero raw console output
- 💾 **Backup & checkpoint** — download SQLite DB via API, manual WAL checkpoint, graceful shutdown
- 🐳 **One-command deploy** — `docker compose up`

---

## What's New in v5.6

### Node.js 22 Runtime (Primary)

- **Node.js 22+ as primary runtime** — `node --experimental-strip-types` for native TypeScript support. Bun support maintained for compatibility.
- **Optimized MCP server** — rewritten for Node.js, reduced from 529 to 168 lines. Faster startup, lower memory footprint.
- **vitest test framework** — 76+ test cases covering API, FSRS, indexing, and CLI.

### Graph Intelligence Layer

- **Graphology integration** — store memories as knowledge graph nodes with relationship edges.
- **Auto-linking metrics** — compute centrality, shortest paths, community detection.
- **Relationship inference** — LLM extracts "mentions", "depends_on", "causes", "related_to" edges from memory content.
- **Graph visualization** — updated galaxy view shows relationship strength.

### MCP Server Improvements

- **Better error handling** — detailed error context propagated to Claude Desktop.
- **Streaming responses** — large memory exports use progress streams.
- **Tool introspection** — dynamic tool discovery for client apps.

### Security Hardening (v5.3+)

- **Auth required by default** — unauthenticated requests are rejected. Set `ENGRAM_OPEN_ACCESS=1` for single-user mode.
- **Rate limit fix** — rate-limited requests no longer escalate to admin privileges.
- **Body size limits** — 1MB per request, 100KB per memory content (configurable).
- **GUI auth rate limiting** — 5 attempts per minute, 10-minute lockout.
- **Timing-safe password comparison** — `timingSafeEqual` for GUI auth.
- **Security headers** — `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy` on every response.
- **CORS origin pinning** — `ENGRAM_CORS_ORIGIN` instead of wildcard `*`.
- **IP allowlisting** — `ENGRAM_ALLOWED_IPS` restricts access to specific addresses.
- **Env var cleanup** — `ENGRAM_PORT`/`ENGRAM_HOST` (old `ZANMEMORY_*` still works with deprecation warning).

### Audit Trail

Every mutation is logged to the `audit_log` table:

```bash
curl http://localhost:4200/audit?limit=5 \
  -H "Authorization: Bearer eg_your_key"
```

### Structured Logging

All output is JSON with configurable levels:

```json
{"level":"info","ts":"2026-03-10T09:32:06Z","msg":"server_started","version":"5.3","port":4200}
```

Levels: `debug`, `info`, `warn`, `error`, `none`. Set via `ENGRAM_LOG_LEVEL`.

### FSRS-6 Spaced Repetition (v5.0)

Replaces simple exponential decay with **FSRS-6** — a 21-parameter algorithm trained on millions of Anki reviews. Memory decay follows a **power-law forgetting curve**.

**Key formula:** `R = (1 + factor × t/S)^(-w₂₀)` where S is stability (days until 90% recall probability).

Every memory access is an implicit FSRS review. The **dual-strength model** (Bjork & Bjork 1992) tracks:
- **Storage strength** — increases with each access, never decays (long-term consolidation)
- **Retrieval strength** — decays via power law, reset on access (current accessibility)

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

### From source (Node.js 22+ — recommended)

```bash
git clone https://github.com/zanfiel/engram.git
cd engram
npm install
ENGRAM_GUI_PASSWORD=your-password node --experimental-strip-types server.ts
```

### From source (Bun — legacy)

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

The SDK lives in `sdk/` within the repo. Zero dependencies — uses native `fetch`.

```bash
cd sdk && npm install && npm run build
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

> **npm:** `@engram/sdk` will be published to npm soon. For now, install from the repo.

### Python

The Python SDK lives in `sdk-py/`. Requires `httpx`.

```bash
cd sdk-py && pip install -e .
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

> **PyPI:** `engram-sdk` will be published to PyPI soon. For now, install from the repo.

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
```

---

## MCP Server

Engram includes a real [Model Context Protocol](https://modelcontextprotocol.io/) server for integration with Claude Desktop, Cursor, Windsurf, and other MCP-compatible tools.

**Transport:** JSON-RPC 2.0 over stdio

### Setup (Claude Desktop)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": ["path/to/engram/mcp-server.mjs"],
      "env": {
        "ENGRAM_URL": "http://localhost:4200",
        "ENGRAM_API_KEY": "eg_your_key"
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a new memory |
| `memory_search` | Semantic + full-text search |
| `memory_recall` | Agent-optimized contextual recall |
| `memory_pack` | Token-budget-aware context packing |
| `memory_forget` | Soft-delete a memory |
| `memory_update` | Create a new version of a memory |
| `memory_tags` | List all tags |
| `memory_list` | List recent memories |

### Available Resources

| Resource | Description |
|----------|-------------|
| `engram://health` | Server health + feature flags |
| `engram://profile` | User profile (static facts + recent activity) |
| `engram://stats` | Detailed statistics |

---

## CLI

Full-featured command-line interface in `sdk/cli.mjs`.

```bash
# Symlink for global access
ln -s $(pwd)/sdk/cli.mjs ~/.local/bin/engram

# Or run directly
node sdk/cli.mjs <command>
```

### Commands

```bash
engram store "Server migrated to new IP" --category state --importance 7
engram search "deployment history" --limit 5
engram recall "What was deployed recently?"
engram list --limit 10 --category task
engram get 42
engram update 42 "Server migrated to 10.0.0.5"
engram forget 42
engram tags
engram episodes
engram health
engram stats
engram export > backup.json
engram import < backup.json
engram pack "current project context" --tokens 4000
```

Environment variables: `ENGRAM_URL` (default: `http://localhost:4200`), `ENGRAM_API_KEY`.

Supports stdin piping: `echo "Deploy completed" | engram store --category task`

---

## API Reference

### Authentication

All endpoints require `Authorization: Bearer eg_...` header by default. Set `ENGRAM_OPEN_ACCESS=1` for unauthenticated single-user mode.

Use `X-Space: space-name` (or `X-Engram-Space`) header to scope operations to a specific memory space. Every response includes an `X-Request-Id` header for correlation.

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

### Conversations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/conversations/bulk` | Bulk store conversation (`agent` + `messages` required) |
| `POST` | `/conversations/upsert` | Upsert by session_id |
| `GET` | `/conversations` | List conversations |
| `GET` | `/conversations/:id/messages` | Get conversation messages |
| `POST` | `/messages/search` | Search across all messages |

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

### Review Queue

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/inbox` | List pending memories |
| `POST` | `/inbox/:id/approve` | Approve a pending memory |
| `POST` | `/inbox/:id/reject` | Reject (archive + set reason) |
| `POST` | `/inbox/:id/edit` | Edit content + auto-approve |
| `POST` | `/inbox/bulk` | Bulk approve/reject |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (30+ feature flags) |
| `GET` | `/stats` | Detailed statistics |
| `GET` | `/audit` | Query audit log (admin) |
| `POST` | `/checkpoint` | Manual WAL checkpoint (admin) |
| `GET` | `/backup` | Download SQLite database (admin) |

---

## How It Works

### Memory Lifecycle

1. **Store** — Memory content is embedded using MiniLM (384-dim vectors, runs locally via ONNX) and stored in libsql with FTS5 full-text indexing.

2. **Auto-link** — New memories are compared against existing ones via in-memory cosine similarity. Memories above 0.7 similarity are linked with typed relationships (similarity, updates, extends, contradicts, caused_by, prerequisite_for).

3. **FSRS-6 initialization** — Each new memory gets initial FSRS state: stability, difficulty, storage strength, retrieval strength. The power-law forgetting curve starts tracking retrievability.

4. **Fact extraction** — If an LLM is configured, Engram analyzes new memories, extracts static facts, auto-tags with keywords, classifies importance, and detects relationships to existing memories.

5. **Recall** — Four retrieval strategies combined: static facts (always), semantic matches (cosine similarity), high-importance (weighted by FSRS retrievability), recent (temporal). Every recalled memory gets an implicit FSRS review, building stability.

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
│  │  FSRS-6  │  │ Cosine   │  │  FTS5    │  │
│  │  Engine   │  │ Sim.     │  │  Search  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │        │
│  ┌────┴──────────────┴──────────────┴────┐  │
│  │    libsql (SQLite + vector columns)   │  │
│  │       FLOAT32(384) + FTS5             │  │
│  └───────────────────────────────────────┘  │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  MiniLM  │  │  LLM     │  │  Graph   │  │
│  │  Embedder │  │  (opt.)  │  │  Engine  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

- **Runtime:** Bun (primary) or Node.js 22+ (with `--experimental-strip-types`)
- **Database:** libsql (SQLite fork with vector column support)
- **Embeddings:** Xenova/all-MiniLM-L6-v2 (384-dim, runs locally via ONNX)
- **Search:** In-memory cosine similarity + FTS5 full-text hybrid
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
| `ENGRAM_PORT` | `4200` | Server port |
| `ENGRAM_HOST` | `0.0.0.0` | Bind address |
| `ENGRAM_GUI_PASSWORD` | `changeme` | GUI login password |
| `ENGRAM_OPEN_ACCESS` | `0` | Set `1` for unauthenticated single-user mode |
| `ENGRAM_LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error`, `none` |
| `ENGRAM_CORS_ORIGIN` | `*` | Pin to your domain in production |
| `ENGRAM_MAX_BODY_SIZE` | `1048576` | Max request body (bytes) |
| `ENGRAM_MAX_CONTENT_SIZE` | `102400` | Max memory content (bytes) |
| `ENGRAM_ALLOWED_IPS` | — | Comma-separated IP allowlist |
| `ENGRAM_HOT_RELOAD` | `0` | Set `1` to reload GUI from disk each request |
| `LLM_URL` | — | OpenAI-compatible API URL |
| `LLM_API_KEY` | — | API key for LLM |
| `LLM_MODEL` | — | Model name (e.g., `gpt-4o`, `claude-sonnet-4-20250514`) |

### Storage

All data lives in a single libsql database (`data/memory.db`). Vector embeddings are stored as `FLOAT32(384)` columns.

**Backup:** `GET /backup` returns a downloadable copy (admin required). WAL checkpoints every 5 minutes and on graceful shutdown. Manual checkpoint via `POST /checkpoint`.

**Audit:** `GET /audit` shows all mutations — who stored, deleted, archived, or modified memories, from which IP, with request IDs.

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
| **Dual-strength memory model** | ✅ | ❌ | ❌ |
| Semantic search (hybrid) | ✅ | ✅ | ✅ |
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
| MCP server (JSON-RPC stdio) | ✅ | ❌ | ❌ |
| CLI | ✅ | ❌ | ❌ |
| Multi-tenant + API keys | ✅ | ✅ | ❌ |
| Spaces / collections | ✅ | ❌ | ✅ |
| Entities & projects | ✅ | ❌ | ❌ |
| Episodic memory | ✅ | ❌ | ❌ |
| Conversation log + search | ✅ | ❌ | ❌ |
| Webhooks & digests | ✅ | ❌ | ❌ |
| Cross-instance sync | ✅ | ❌ | ❌ |
| URL ingest | ✅ | ❌ | ❌ |
| Import from Mem0 / Supermemory | ✅ | — | — |
| Review queue / inbox | ✅ | ❌ | ❌ |
| Audit trail | ✅ | ❌ | ❌ |
| Structured JSON logging | ✅ | ❌ | ❌ |
| API backup & WAL checkpoint | ✅ | ❌ | ❌ |
| Self-hosted | ✅ | ✅ | ✅ |
| Single-file DB (zero deps) | ✅ | ❌ | ❌ |

---

## Test Suite

```bash
# Start the server, then:
cd engram
node --test tests/api.test.mjs
# 38 tests, 12 suites, 0 failures
```

---

## License

MIT
