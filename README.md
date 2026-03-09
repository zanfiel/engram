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

- 🧠 **Semantic + full-text search** — find memories by meaning or keywords, locally
- 🔗 **Auto-linking** — memories automatically connect, forming a knowledge graph
- 📊 **Graph visualization** — explore your memory space in a WebGL galaxy
- 🔄 **Versioning** — update memories without losing history
- 🧹 **Auto-deduplication** — detects and merges near-duplicate memories
- ⏰ **Auto-forget & decay** — time-weighted importance with access reinforcement
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

# Context manager
with Engram("http://localhost:4200", api_key="eg_...") as e:
    e.store("temporary session data", category="state")
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

### Maintenance

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/duplicates` | Find duplicate clusters |
| `POST` | `/deduplicate` | Auto-merge duplicates |
| `POST` | `/consolidate` | Consolidate memory cluster |
| `POST` | `/sweep` | Run forget sweep |
| `POST` | `/backfill` | Backfill missing embeddings |
| `POST` | `/decay/refresh` | Recalculate decay scores |
| `GET` | `/decay/scores` | View decay scores |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (28 feature flags) |
| `GET` | `/stats` | Detailed statistics |

### Store Request

```json
{
  "content": "The deployment uses blue-green strategy",
  "category": "decision",
  "source": "my-agent",
  "importance": 7,
  "session_id": "session-abc"
}
```

**Categories:** `task`, `discovery`, `decision`, `state`, `issue`, `general`

**Importance:** 1-10. Higher importance memories surface more in recall and resist auto-forgetting.

### Search Response

```json
{
  "results": [
    {
      "id": 42,
      "content": "The deployment uses blue-green strategy",
      "category": "decision",
      "importance": 7,
      "score": 0.89,
      "created_at": "2024-01-15T10:30:00Z",
      "version": 1,
      "is_static": false,
      "source_count": 3
    }
  ]
}
```

### Recall Response

```json
{
  "memories": [...],
  "breakdown": {
    "static_facts": 5,
    "semantic_matches": 8,
    "important": 4,
    "recent": 8
  }
}
```

Recall combines four retrieval strategies:
1. **Static facts** — always included (high source_count, marked static)
2. **Semantic matches** — embedding similarity to query
3. **High importance** — memories with importance ≥ 7
4. **Recent activity** — latest memories for temporal context

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
| `LLM_URL` | — | OpenAI-compatible API URL for fact extraction |
| `LLM_API_KEY` | — | API key for LLM |
| `LLM_MODEL` | — | Model name (e.g., `gpt-4o`, `claude-sonnet-4-20250514`) |

### Architecture

- **Runtime:** Bun
- **Database:** SQLite (with FTS5 for full-text search)
- **Embeddings:** Xenova/all-MiniLM-L6-v2 (384-dim, runs locally — no external API needed)
- **LLM:** Optional, for fact extraction only. Any OpenAI-compatible API.

### Storage

All data lives in a single SQLite file (`data/memory.db` in Docker, or `./memory.db` from source). Back it up however you back up files.

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

## How It Works

1. **Store** — Memory content is embedded using MiniLM (384-dim vectors) and stored in SQLite with full-text indexing.

2. **Auto-link** — New memories are compared against existing ones. Memories above 0.75 similarity are linked with typed relationships (similarity, updates, extends, contradicts, caused_by, prerequisite_for).

3. **Fact extraction** — If an LLM is configured, Engram analyzes new memories, extracts static facts, auto-tags with keywords, classifies importance, and detects relationships to existing memories (updates, duplicates, contradictions).

4. **Recall** — When an agent needs context, Engram combines four strategies: static facts (always), semantic matches (relevant), high-importance (critical), and recent (temporal). The Smart Context Builder adds token-budget awareness and graph expansion.

5. **Contradiction detection** — Periodically scans for memories that conflict. LLM verification eliminates false positives. Contradictions can be resolved by keeping one side, both, or merging.

6. **Deduplication** — Periodic sweeps find near-identical memories and merge them, incrementing source_count to track how often something was mentioned.

7. **Decay & forget** — Memories decay over time based on access patterns. Static memories are immune. Auto-forget sweeps expire TTL-based memories every 5 minutes.

8. **Versioning** — Updating a memory creates a new version linked to the original. Time-travel queries can reconstruct the knowledge state at any past timestamp.

9. **Consolidation** — Large clusters of related memories get summarized into a single dense memory. Originals are archived, links preserved.

10. **Reflection** — On-demand meta-analysis generates insights about themes, progress, and patterns over any time period. Reflections become searchable memories themselves.

---

## Comparison

| Feature | Engram | Mem0 | Supermemory |
|---------|--------|------|-------------|
| Semantic search | ✅ | ✅ | ✅ |
| Local embeddings | ✅ | ❌ | ❌ |
| Full-text search (FTS5) | ✅ | ❌ | ❌ |
| Graph visualization | ✅ | ❌ | ✅ |
| Memory versioning | ✅ | ❌ | ❌ |
| Auto-deduplication | ✅ | ❌ | ❌ |
| Auto-forget / TTL | ✅ | ❌ | ❌ |
| Decay scoring | ✅ | ❌ | ❌ |
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
| Webhooks & digests | ✅ | ❌ | ❌ |
| Cross-instance sync | ✅ | ❌ | ❌ |
| MCP server + CLI | ✅ | ❌ | ❌ |
| URL ingest | ✅ | ❌ | ❌ |
| Self-hosted | ✅ | ✅ | ✅ |
| No external API needed | ✅ | ❌ | ❌ |
| SQLite (zero deps) | ✅ | ❌ | ❌ |

---

## License

MIT
