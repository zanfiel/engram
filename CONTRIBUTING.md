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
server.ts          — Main server (being modularized into src/)
├── Embedding      — Xenova/all-MiniLM-L6-v2 via @huggingface/transformers
├── Database       — SQLite via libsql with FTS5 + FLOAT32 vectors
├── FSRS-6         — Spaced repetition scheduler (21 trained weights)
├── LLM            — Fact extraction, consolidation, reflections (optional)
├── Auth           — API keys + GUI cookies + RBAC
└── HTTP           — Node.js createServer with Web Request/Response adapter

engram-gui.html    — WebGL galaxy visualization (standalone HTML)
engram-login.html  — Login page
landing.html       — Marketing landing page
```

### Key Design Decisions

1. **Single file (for now)**: The server is ~6600 lines in one file. This is being split into `src/` modules, but the monolith works fine for a single-process server with no dependency injection needed.

2. **libsql, not better-sqlite3**: We use libsql for native FLOAT32 vector columns and HNSW index support. This gives us vector search without an external service.

3. **In-memory embedding cache**: All embeddings are loaded into RAM on startup (~1.5KB per memory). This makes vector search sub-millisecond for thousands of memories. The tradeoff is O(n) memory usage.

4. **FSRS-6 over exponential decay**: Every other memory system uses exponential decay. We use the FSRS-6 algorithm (power-law forgetting curve) with 21 weights trained on millions of Anki reviews. This is mathematically more accurate and gives us features nobody else has (dual-strength model, same-day review handling, optimal review intervals).

5. **Node.js HTTP, not Express/Hono/Bun**: Zero framework dependency. The server uses `createServer` with a Web Request/Response adapter. This keeps the dependency count at 2 (libsql + @huggingface/transformers).

## Code Style

- TypeScript with `--experimental-strip-types` (no build step)
- Structured JSON logging via the `log` object
- Prepared statements for all repeated queries
- `migrate()` helper for safe schema evolution
- All API responses use the `json()` helper with security headers

## Testing

Currently tested via integration (real server + real SQLite). We need:
- [ ] Unit tests for FSRS-6 math
- [ ] Unit tests for hybrid search scoring
- [ ] Integration test suite for API endpoints
- [ ] Benchmark suite for search latency

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes with clear commit messages
3. Ensure no regressions (run the server, test affected endpoints)
4. Update CHANGELOG.md under `[Unreleased]`
5. Submit a PR with a clear description of what changed and why

## Areas Where Help Is Needed

- **MCP Server**: Full Model Context Protocol implementation for Claude/Cursor/VS Code
- **OpenAPI Spec**: Generate from route definitions → Swagger UI at `/docs`
- **SDK Clients**: `@engram/client` (TypeScript), `engram-client` (Python)
- **Benchmarks**: Measure and publish latency vs Mem0, SuperMemory, ChromaDB
- **Encryption at rest**: SQLCipher integration or envelope encryption
- **Test suite**: Unit + integration tests
- **Documentation**: API reference, deployment guides, architecture diagrams

## License

MIT — see [LICENSE](LICENSE) for details.
