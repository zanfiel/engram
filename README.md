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

- 🧠 **Semantic search** — find memories by meaning, not just keywords
- 🔗 **Auto-linking** — memories automatically connect to related memories
- 📊 **Graph visualization** — explore your memory space in a WebGL galaxy
- 🔄 **Versioning** — update memories without losing history
- 🧹 **Auto-deduplication** — detects and merges near-duplicate memories
- ⏰ **Auto-forget** — set memories to expire after a duration
- 🔍 **Fact extraction** — LLM-powered extraction of static facts from conversations
- 👥 **Multi-tenant** — isolated memory per user with API keys
- 📦 **Spaces** — organize memories into named collections
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
| `POST` | `/search` | Semantic search |
| `POST` | `/recall` | Contextual recall (agent-optimized) |
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

### Data

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/export` | Export all memories + links (JSON/JSONL) |
| `POST` | `/import` | Bulk import memories |

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
| `POST` | `/sweep` | Run forget sweep |
| `POST` | `/backfill` | Backfill missing embeddings |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
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

All data lives in a single SQLite file (`data/megamind.db` in Docker, or `./megamind.db` from source). Back it up however you back up files.

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

2. **Auto-link** — New memories are compared against existing ones. Memories above 0.75 similarity are linked, forming a knowledge graph.

3. **Fact extraction** — If an LLM is configured, Engram analyzes memories and extracts static facts (things that are persistently true). These become high-priority recall candidates.

4. **Recall** — When an agent needs context, Engram combines four strategies: static facts (always), semantic matches (relevant), high-importance (critical), and recent (temporal). This creates a rich context window without overwhelming the agent.

5. **Deduplication** — Periodic sweeps find near-identical memories and merge them, incrementing source_count to track how often something was mentioned.

6. **Auto-forget** — Memories can be set to expire. A background sweep runs every 5 minutes to mark expired memories as forgotten.

7. **Versioning** — Updating a memory creates a new version linked to the original. The full chain is preserved, but only the latest version surfaces in search/recall.

---

## Comparison

| Feature | Engram | Mem0 | Supermemory |
|---------|--------|------|-------------|
| Semantic search | ✅ | ✅ | ✅ |
| Local embeddings | ✅ | ❌ | ❌ |
| Graph visualization | ✅ | ❌ | ✅ |
| Memory versioning | ✅ | ❌ | ❌ |
| Auto-deduplication | ✅ | ❌ | ❌ |
| Auto-forget | ✅ | ❌ | ❌ |
| Fact extraction | ✅ | ✅ | ❌ |
| Multi-tenant | ✅ | ✅ | ❌ |
| Spaces/collections | ✅ | ❌ | ✅ |
| Self-hosted | ✅ | ✅ | ✅ |
| No external API needed | ✅ | ❌ | ❌ |
| SQLite (zero deps) | ✅ | ❌ | ❌ |

---

## License

MIT
