# Changelog

All notable changes to Engram will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.7.0] - 2026-03-14

### Security
- **Multi-tenant data isolation**: Complete audit and fix of 30+ cross-tenant data boundaries
  - `getCachedEmbeddings` now accepts optional `userId` filter — all callers in search, auto-link, contradiction detection, deduplication, and fact extraction pass the authenticated user's ID
  - `autoLink` scoped to same-user memories — prevents cross-user link creation
  - `addToEmbeddingCache` now includes `user_id` — fixes cache filtering for newly added memories
  - Conversation prepared statements (`listConversations`, `listConversationsByAgent`, `searchMessages`, `getConversationBySession`, `deleteConversation`) all filter by `user_id`
  - Conversation CRUD endpoints (`GET/PATCH/DELETE /conversations/:id`, `POST /conversations/:id/messages`, `POST /conversations/bulk`, `POST /conversations/upsert`, `POST /messages/search`) have ownership checks and write scope guards
  - `/contradictions` and `/contradictions/resolve` scoped to authenticated user with ownership verification
  - `/duplicates` and `/deduplicate` scoped to authenticated user
  - `/memory/:id/update` now requires write scope and verifies memory ownership
  - Entity and project `GET /:id` endpoints verify ownership
  - Entity/project link and unlink endpoints verify ownership of both the entity/project and the memory
  - `/links/:id` and `/versions/:id` verify memory ownership before returning data
  - `/episodes/:id` GET and PATCH verify episode ownership
  - `/decay/scores` filtered by `user_id`
  - `/consolidations` filtered by `user_id`
  - `/fsrs/init` scoped to user's memories with write scope guard
  - `/graph` BFS traversal joins against `memories` table for user_id filtering
  - `/stats` consolidated from 13 unfiltered `COUNT(*)` queries to 4 user-scoped queries; `db_path` removed from non-admin response
- Write scope guards added to `/sweep`, `/backfill`, `/deduplicate`

### Fixed
- `propagateConfidence` in `src/llm/index.ts` fully implemented (was a no-op stub in the modular split)
  - "updates" relation: sets superseded memory confidence to 0.3
  - "contradicts" relation: reduces both memories' confidence, fires `contradiction.detected` webhook
  - "extends" relation: boosts corroborated memory confidence by 5%
- `insertConversation` and `insertConversationTx` now include `user_id` column — conversations are attributed to the authenticated user at insert time instead of via follow-up UPDATE

### Added
- API test suite: `tests/api.test.mjs` — 33 tests across 14 suites covering health, store, search, recall, memory CRUD, conversations, stats, contradictions, duplicates, graph, episodes, entities, projects, FSRS, and cleanup

## [5.5.0] - 2026-03-13

### Changed
- **Modular architecture**: Monolith `server.ts` split into focused modules under `src/`
  - `src/config/` — environment config and logger
  - `src/db/` — schema, migrations, prepared statements
  - `src/embeddings/` — model init, in-memory cache, similarity
  - `src/fsrs/` — FSRS-6 spaced repetition engine
  - `src/intelligence/` — fact extraction, consolidation
  - `src/llm/` — LLM client, reranking
  - `src/memory/` — hybrid search, auto-linking, profile
  - `src/platform/` — webhooks, digests
  - `src/auth/` — authentication middleware
  - `src/tier4/` — causal chains, predictive recall, emotional valence, reconsolidation
  - `src/routes/` — all HTTP handlers
- Entry point is now `server-split.ts` (imports from `src/`)

### Fixed
- `FSRSRating` is now both a const value and a type (declaration merging) — fixes TypeScript errors across FSRS consumers
- Exported `FSRSMemoryState` interface from `src/fsrs/`
- Aliased `calculateDecayScore` import to resolve shadowing by local route wrapper
- Exported `graphCache` + `setGraphCache` from `src/embeddings/` to break module boundary
- Added `"corrects"` to `FactExtractionResult` relation type union — was in the LLM prompt but missing from the TypeScript type
- Fixed `trackAccessWithFSRS` local wrapper to accept optional `grade` parameter
- Fixed `addToEmbeddingCache` calls missing `user_id` field

## [5.4.0] - 2026-03-10

### Security
- **S1**: Cookie verification now uses `timingSafeEqual` — prevents timing attacks on GUI auth
- **S2**: Fixed undefined `reason` variable in inbox reject — was crashing on every reject
- **S3**: Fixed `require("fs")` in ESM backup cleanup — temp files were never deleted
- **S4**: Fixed `extractFacts()` call signatures in `/add` and `/ingest` — was passing memory ID as content and Float32Array as similar memories, sending garbage to LLM
- **S5**: Added per-IP rate limiting for `OPEN_ACCESS` mode — prevents DoS
- **S6**: Added webhook URL validation — blocks SSRF to private/internal IP ranges
- **S7**: Split `/health` into light (unauthenticated: status + version + count) and full (authenticated: all config + stats)
- Added `Strict-Transport-Security` header (HSTS)
- Added `Content-Security-Policy` header on all responses

### Added
- **RBAC**: Users now have roles (`admin`, `writer`, `reader`) with enforced scope restrictions
  - `reader`: can only search/recall, cannot create/modify memories
  - `writer`: can create/modify own memories, cannot access admin endpoints
  - `admin`: full access (default for backwards compatibility)
- Source filter on `/list` endpoint (`?source=conversation`)
- Body fields `is_static`, `forget_after`, `forget_reason`, `is_inference` now respected on `/store`
- `.env.example` with all configuration variables documented
- `tsconfig.json` with strict mode
- `CHANGELOG.md` (this file)
- `CONTRIBUTING.md` with development guidelines

### Fixed
- **B1**: `/store` now respects `is_static` from request body (was always 0)
- **B2**: `/store` now respects `forget_after` from request body (was always null)
- **B3**: `/list` now supports `?source=` query parameter
- **B5**: GUI PATCH now invalidates embedding cache after edits
- **B6**: Bulk inbox error handler no longer re-reads consumed request body

## [5.3.0] - 2026-03-09

### Added
- FSRS-6 spaced repetition system (21 trained weights from Anki dataset)
- Dual-strength model (Bjork & Bjork 1992) — storage strength vs retrieval strength
- LLM-based fact extraction with relationship detection (updates, extends, contradicts, caused_by, prerequisite_for)
- Contradiction detection and resolution (`/contradictions`, `/contradictions/resolve`)
- Time travel queries (`/timetravel`) — query memory state at any past timestamp
- Smart context builder (`/context`) with token-budgeted packing and strategies (balanced/precision/breadth)
- Memory reflections (`/reflect`) — periodic meta-analysis with theme detection
- Scheduled digests (`/digests`) — webhook delivery of memory summaries
- Derived memories (`/derive`) — LLM inference of new facts from existing clusters
- Auto-consolidation — summarizes large memory clusters automatically
- Import from Mem0 and SuperMemory formats
- Entity and project management with scoped search
- Review queue / inbox (`/inbox`) with approve/reject/edit workflow
- Comprehensive audit log with per-action tracking
- Webhook event system with HMAC signing
- Multi-instance sync (`/sync/changes`, `/sync/receive`)
- Graph API with BFS traversal, entity overlay, and project grouping
- Episode tracking for conversation sessions
- Duplicate detection and deduplication (`/duplicates`, `/deduplicate`)
- WebGL galaxy visualization GUI
- Prompt template engine (`/prompt`) with Anthropic/OpenAI/LlamaIndex formats
- Context window optimizer (`/pack`) with greedy token packing

## [5.0.0] - 2026-03-08

### Added
- Multi-tenant architecture: users, API keys, spaces
- API key authentication with scoped permissions
- Per-key rate limiting
- Web GUI with password authentication
- Full-text search (FTS5) + vector hybrid search
- In-memory embedding cache for sub-millisecond search
- Auto-linking based on embedding similarity
- Version chains with parent/root memory tracking
- Memory forgetting with TTL (`forget_after`)
- Static vs dynamic memory classification
- Confidence scoring with propagation
- libsql native FLOAT32 vector column with HNSW index

## [4.0.0] - 2026-03-07

### Added
- SQLite + FTS5 persistent storage
- Xenova/all-MiniLM-L6-v2 local embeddings (384 dimensions)
- Basic CRUD: store, search, recall, delete
- Conversation storage with message search
- JSON export/import
- Node.js HTTP server (zero dependencies beyond libsql + transformers)

## [3.0.0] - 2026-03-06

### Added
- Initial release as ZanMemory/MegaMind
- In-memory storage with periodic persistence
- Basic embedding search
