# Contributing to Engram

Thanks for your interest in contributing to Engram! This document provides guidelines and information for contributors.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/zanfiel/engram.git
cd engram

# Install dependencies
npm install

# Copy environment config
cp .env.example .env

# Start in dev mode (auto-restart on changes)
npm run dev
```

**Requirements:**
- Node.js ≥ 22.0.0
- ~200MB disk for embedding model (auto-downloaded on first run)

## Architecture

Engram is a single-process TypeScript server with zero external service dependencies.

```
server.ts          — Monolith server (~7400 lines, legacy — use server-split.ts)
server-split.ts    — Modular entrypoint (imports from src/)
mcp-server.ts      — MCP server (JSON-RPC 2.0 stdio transport)
src/
├── auth/          — API keys, GUI cookies, RBAC
├── config/        — Environment and runtime configuration
├── db/            — libsql with FTS5 + FLOAT32 vectors
├── embeddings/    — Xenova/all-MiniLM-L6-v2 via @huggingface/transformers
├── fsrs/          — FSRS-6 spaced repetition (21 trained weights)
├── graph/         — Graphology-based knowledge graph + community detection
├── gui/           — GUI route handlers
├── helpers/       — Shared utilities
├── intelligence/  — Fact extraction, consolidation, reflections, contradiction detection
├── llm/           — LLM client (optional, OpenAI-compatible)
├── memory/        — Core memory CRUD + versioning
├── organization/  — Tags, episodes, entities, projects, spaces
├── platform/      — Webhooks, digests, sync, import/export
├── routes/        — HTTP route definitions
└── tier4/         — Advanced features (time-travel, derived memories, reranker)

engram-gui.html    — WebGL galaxy visualization (standalone HTML)
engram-login.html  — Login page
landing.html       — Marketing landing page
```

### Key Design Decisions

1. **Modular architecture**: The server has been split from a monolith (`server.ts`, ~7400 lines) into `src/` modules. The modular entrypoint is `server-split.ts`. Both work, but new development targets `src/`.

2. **libsql, not better-sqlite3**: We use libsql for native FLOAT32 vector columns and HNSW index support. This gives us vector search without an external service.

3. **In-memory embedding cache**: All embeddings are loaded into RAM on startup (~1.5KB per memory). This makes vector search sub-millisecond for thousands of memories. The tradeoff is O(n) memory usage.

4. **FSRS-6 over exponential decay**: Every other memory system uses exponential decay. We use the FSRS-6 algorithm (power-law forgetting curve) with 21 weights trained on millions of Anki reviews. This is mathematically more accurate and gives us features nobody else has (dual-strength model, same-day review handling, optimal review intervals).

5. **Node.js HTTP, not Express/Hono**: Zero framework dependency. The server uses `createServer` with a Web Request/Response adapter. Core dependencies: libsql, @huggingface/transformers, graphology, and @modelcontextprotocol/sdk.

## Code Style

- TypeScript with `--experimental-strip-types` (no build step)
- Structured JSON logging via the `log` object
- Prepared statements for all repeated queries
- `migrate()` helper for safe schema evolution
- All API responses use the `json()` helper with security headers

## Testing

API tests use Node.js built-in test runner. Start the server, then:

```bash
# Run against localhost:4200 (default)
node --test tests/api.test.mjs

# Or specify a different URL
ENGRAM_URL=http://localhost:4201 node --test tests/api.test.mjs
```

33 tests across 14 suites covering core API, multi-tenant isolation, CRUD, FSRS, and more.

We always need more coverage:
- [ ] Benchmark suite for search latency vs competitors
- [ ] Stress tests for large memory sets (10k+ memories)
- [ ] Multi-user isolation tests (two API keys, verify data separation)

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes with clear commit messages
3. Ensure no regressions (run the server, test affected endpoints)
4. Update CHANGELOG.md under `[Unreleased]`
5. Submit a PR with a clear description of what changed and why

## Areas Where Help Is Needed

- **OpenAPI Spec**: Generate from route definitions → Swagger UI at `/docs`
- **SDK publishing**: `@engram/sdk` (TypeScript) and `engram-sdk` (Python) need npm/PyPI publishing
- **Benchmarks**: Measure and publish latency vs Mem0, SuperMemory, ChromaDB
- **Encryption at rest**: SQLCipher integration or envelope encryption
- **Documentation**: Deployment guides, architecture diagrams, more examples

## License

Elastic License 2.0 — see [LICENSE](LICENSE) for details.
