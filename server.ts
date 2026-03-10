import Database from "libsql";
import { resolve } from "path";
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

const DATA_DIR = resolve(import.meta.dir, "data");
const DB_PATH = resolve(DATA_DIR, "memory.db");
const PORT = Number(process.env.ZANMEMORY_PORT || 4200);
const HOST = process.env.ZANMEMORY_HOST || "0.0.0.0";

// Embedding config
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const EMBEDDING_DIM = 384;
const AUTO_LINK_THRESHOLD = 0.7;
const AUTO_LINK_MAX = 3;
const DEFAULT_IMPORTANCE = 5;

// LLM config (for fact extraction)
const LLM_URL = process.env.LLM_URL || "http://127.0.0.1:4100/v1/chat/completions";
const LLM_API_KEY = process.env.LLM_API_KEY || "";
const LLM_MODEL = process.env.LLM_MODEL || "claude-sonnet-4-20250514";

// Auto-forget sweep interval (every 5 minutes)
const FORGET_SWEEP_INTERVAL = 5 * 60 * 1000;

// FSRS-6 configuration (replaces simple exponential decay)
const FSRS_DEFAULT_RETENTION = 0.9; // target 90% recall probability
const CONSOLIDATION_THRESHOLD = 8; // auto-consolidate clusters with 8+ related memories
const CONSOLIDATION_INTERVAL = 30 * 60 * 1000; // check every 30 minutes

// Reranker config
const RERANKER_ENABLED = process.env.RERANKER !== "0";
const RERANKER_TOP_K = Number(process.env.RERANKER_TOP_K || 20); // rerank top K candidates

// API key config
const API_KEY_PREFIX = "eg_";
const DEFAULT_RATE_LIMIT = 120; // requests per minute
const RATE_WINDOW_MS = 60_000;

// Ensure data directory exists
await Bun.$`mkdir -p ${DATA_DIR}`;

// ============================================================================
// EMBEDDING MODEL
// ============================================================================

let embedder: FeatureExtractionPipeline | null = null;

async function initEmbedder(): Promise<void> {
  const start = Date.now();
  console.log("Loading embedding model...");
  embedder = await pipeline("feature-extraction", EMBEDDING_MODEL, {
    dtype: "fp32",
  }) as FeatureExtractionPipeline;
  console.log(`Embedding model loaded in ${Date.now() - start}ms (${EMBEDDING_MODEL})`);
}

async function embed(text: string): Promise<Float32Array> {
  if (!embedder) throw new Error("Embedding model not loaded");
  const result = await embedder(text, { pooling: "mean", normalize: true });
  return new Float32Array(result.data as Float32Array);
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function embeddingToBuffer(emb: Float32Array): Buffer {
  return Buffer.from(emb.buffer, emb.byteOffset, emb.byteLength);
}

function bufferToEmbedding(buf: Buffer | Uint8Array | ArrayBuffer): Float32Array {
  if (buf instanceof ArrayBuffer) return new Float32Array(buf);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return new Float32Array(ab);
}

/** Convert Float32Array to JSON string for libsql vector() function */
function embeddingToVectorJSON(emb: Float32Array): string {
  return "[" + Array.from(emb).join(",") + "]";
}

// ============================================================================
// LLM-BASED RERANKER
// ============================================================================

async function rerank(
  query: string,
  candidates: Array<{ id: number; content: string; score: number; [k: string]: any }>,
  topK: number = RERANKER_TOP_K
): Promise<typeof candidates> {
  if (!RERANKER_ENABLED || !LLM_API_KEY || candidates.length <= 3) return candidates;

  // Only rerank the top candidates to save tokens
  const toRerank = candidates.slice(0, Math.min(topK, candidates.length));
  const numbered = toRerank.map((c, i) => `[${i}] ${c.content.substring(0, 200)}`).join("\n");

  const prompt = `Given the query, rank the following documents by relevance. Return ONLY a JSON array of indices from most to least relevant. Include ALL indices.

Query: "${query}"

Documents:
${numbered}

Return format: [most_relevant_index, next_most_relevant, ...]`;

  try {
    const resp = await callLLM("You are a document reranking engine. Return only a JSON array of integer indices.", prompt);
    if (!resp) return candidates;

    // Parse the index array
    const match = resp.match(/\[[\d,\s]+\]/);
    if (!match) return candidates;
    const indices = JSON.parse(match[0]) as number[];

    // Build reranked list with adjusted scores
    const reranked: typeof candidates = [];
    for (let rank = 0; rank < indices.length; rank++) {
      const idx = indices[rank];
      if (idx >= 0 && idx < toRerank.length) {
        const item = { ...toRerank[idx] };
        item.score = item.score * (1 + (indices.length - rank) / indices.length * 0.5); // boost by rank
        (item as any).reranked = true;
        reranked.push(item);
      }
    }

    // Add any candidates that weren't reranked (beyond topK)
    const rerankedIds = new Set(reranked.map(r => r.id));
    for (const c of candidates) {
      if (!rerankedIds.has(c.id)) reranked.push(c);
    }

    return reranked;
  } catch {
    return candidates; // fallback to original ranking
  }
}

// ============================================================================
// DATABASE
// ============================================================================

const db = new Database(DB_PATH);
db.exec("PRAGMA journal_mode=WAL");
db.exec("PRAGMA foreign_keys=ON");
db.exec("PRAGMA busy_timeout=5000");

// ============================================================================
// SCHEMA — v3 with versioning, forgetting, static/dynamic, sourceCount
// ============================================================================

db.exec(`
  CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    source TEXT NOT NULL DEFAULT 'unknown',
    session_id TEXT,
    importance INTEGER NOT NULL DEFAULT 5,
    embedding BLOB,
    version INTEGER NOT NULL DEFAULT 1,
    is_latest BOOLEAN NOT NULL DEFAULT 1,
    parent_memory_id INTEGER REFERENCES memories(id),
    root_memory_id INTEGER REFERENCES memories(id),
    source_count INTEGER NOT NULL DEFAULT 1,
    is_static BOOLEAN NOT NULL DEFAULT 0,
    is_forgotten BOOLEAN NOT NULL DEFAULT 0,
    forget_after TEXT,
    forget_reason TEXT,
    is_inference BOOLEAN NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    category,
    source,
    content='memories',
    content_rowid='id',
    tokenize='porter unicode61'
  );

  CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, category, source)
    VALUES (new.id, new.content, new.category, new.source);
  END;

  CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, category, source)
    VALUES ('delete', old.id, old.content, old.category, old.source);
  END;

  CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, category, source)
    VALUES ('delete', old.id, old.content, old.category, old.source);
    INSERT INTO memories_fts(rowid, content, category, source)
    VALUES (new.id, new.content, new.category, new.source);
  END;
`);

// v2 -> v3 migrations (safe to re-run)
const v3Columns: [string, string][] = [
  ["version", "INTEGER NOT NULL DEFAULT 1"],
  ["is_latest", "BOOLEAN NOT NULL DEFAULT 1"],
  ["parent_memory_id", "INTEGER"],
  ["root_memory_id", "INTEGER"],
  ["source_count", "INTEGER NOT NULL DEFAULT 1"],
  ["is_static", "BOOLEAN NOT NULL DEFAULT 0"],
  ["is_forgotten", "BOOLEAN NOT NULL DEFAULT 0"],
  ["forget_after", "TEXT"],
  ["forget_reason", "TEXT"],
  ["is_inference", "BOOLEAN NOT NULL DEFAULT 0"],
  ["is_archived", "BOOLEAN NOT NULL DEFAULT 0"],
];
for (const [col, def] of v3Columns) {
  try { db.exec(`ALTER TABLE memories ADD COLUMN ${col} ${def}`); } catch {}
}
try { db.exec("ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 5"); } catch {}
try { db.exec("ALTER TABLE memories ADD COLUMN embedding BLOB"); } catch {}

// v3.1 indexes
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(is_archived) WHERE is_archived = 1"); } catch {}

// v3 indexes
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_root ON memories(root_memory_id)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_parent ON memories(parent_memory_id)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_latest ON memories(is_latest) WHERE is_latest = 1"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(is_forgotten)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_forget_after ON memories(forget_after) WHERE forget_after IS NOT NULL"); } catch {}

// v4.1 — Access tracking, tags, episodes
const v41Columns: [string, string][] = [
  ["last_accessed_at", "TEXT"],
  ["access_count", "INTEGER NOT NULL DEFAULT 0"],
  ["tags", "TEXT"],  // JSON array: ["tag1", "tag2"]
  ["episode_id", "INTEGER"],
  ["decay_score", "REAL"],  // cached effective score
];
for (const [col, def] of v41Columns) {
  try { db.exec(`ALTER TABLE memories ADD COLUMN ${col} ${def}`); } catch {}
}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags) WHERE tags IS NOT NULL"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_episode ON memories(episode_id) WHERE episode_id IS NOT NULL"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(access_count DESC)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_decay ON memories(decay_score DESC)"); } catch {}

// Episodes table
try {
  db.exec(`
    CREATE TABLE IF NOT EXISTS episodes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT,
      session_id TEXT,
      agent TEXT,
      summary TEXT,
      user_id INTEGER DEFAULT 1,
      memory_count INTEGER NOT NULL DEFAULT 0,
      started_at TEXT NOT NULL DEFAULT (datetime('now')),
      ended_at TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent);
  `);
} catch {}

// Consolidation tracking
try {
  db.exec(`
    CREATE TABLE IF NOT EXISTS consolidations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      summary_memory_id INTEGER NOT NULL REFERENCES memories(id),
      source_memory_ids TEXT NOT NULL,
      cluster_label TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);
} catch {}

// v4.2 — Confidence, sync, webhooks
const v42Columns: [string, string][] = [
  ["confidence", "REAL NOT NULL DEFAULT 1.0"],
  ["sync_id", "TEXT"],  // UUID for cross-instance sync
];
for (const [col, def] of v42Columns) {
  try { db.exec(`ALTER TABLE memories ADD COLUMN ${col} ${def}`); } catch {}
}
try { db.exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_sync_id ON memories(sync_id) WHERE sync_id IS NOT NULL"); } catch {}

// v5.1 — Review queue: status column (pending/approved/rejected)
try { db.exec("ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'approved'"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)"); } catch {}

// v5.0 — FSRS-6 spaced repetition columns
const v50Columns: [string, string][] = [
  ["fsrs_stability", "REAL"],
  ["fsrs_difficulty", "REAL"],
  ["fsrs_storage_strength", "REAL DEFAULT 1.0"],
  ["fsrs_retrieval_strength", "REAL DEFAULT 1.0"],
  ["fsrs_learning_state", "INTEGER DEFAULT 0"],
  ["fsrs_reps", "INTEGER DEFAULT 0"],
  ["fsrs_lapses", "INTEGER DEFAULT 0"],
  ["fsrs_last_review_at", "TEXT"],
];
for (const [col, def] of v50Columns) {
  try { db.exec(`ALTER TABLE memories ADD COLUMN ${col} ${def}`); } catch {}
}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_fsrs_stability ON memories(fsrs_stability) WHERE fsrs_stability IS NOT NULL"); } catch {}

// v5.0 — Native vector column (libsql FLOAT32)
try { db.exec(`ALTER TABLE memories ADD COLUMN embedding_vec FLOAT32(384)`); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS memories_vec_idx ON memories(libsql_vector_idx(embedding_vec))"); } catch {}

// Webhooks table
try {
  db.exec(`
    CREATE TABLE IF NOT EXISTS webhooks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      url TEXT NOT NULL,
      events TEXT NOT NULL DEFAULT '["*"]',
      secret TEXT,
      user_id INTEGER DEFAULT 1,
      active BOOLEAN NOT NULL DEFAULT 1,
      last_triggered_at TEXT,
      failure_count INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_webhooks_user ON webhooks(user_id);
  `);
} catch {}

// v4.3 — Entities, Projects
try {
  db.exec(`
    CREATE TABLE IF NOT EXISTS entities (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      type TEXT NOT NULL DEFAULT 'generic',
      description TEXT,
      aka TEXT,
      metadata TEXT,
      user_id INTEGER DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
    CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);

    CREATE TABLE IF NOT EXISTS entity_relationships (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
      target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
      relationship TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(source_entity_id, target_entity_id, relationship)
    );
    CREATE INDEX IF NOT EXISTS idx_entrel_source ON entity_relationships(source_entity_id);
    CREATE INDEX IF NOT EXISTS idx_entrel_target ON entity_relationships(target_entity_id);

    CREATE TABLE IF NOT EXISTS memory_entities (
      memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
      entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
      PRIMARY KEY (memory_id, entity_id)
    );
    CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);

    CREATE TABLE IF NOT EXISTS projects (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      description TEXT,
      status TEXT NOT NULL DEFAULT 'active',
      metadata TEXT,
      user_id INTEGER DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);
    CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);

    CREATE TABLE IF NOT EXISTS memory_projects (
      memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
      project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
      PRIMARY KEY (memory_id, project_id)
    );
    CREATE INDEX IF NOT EXISTS idx_mp_project ON memory_projects(project_id);
  `);
} catch {}

// v4.5 — Digests, Reflections, Contradiction tracking
try {
  db.exec(`
    CREATE TABLE IF NOT EXISTS digests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL DEFAULT 1,
      schedule TEXT NOT NULL DEFAULT 'daily',
      webhook_url TEXT NOT NULL,
      webhook_secret TEXT,
      include_stats BOOLEAN NOT NULL DEFAULT 1,
      include_new_memories BOOLEAN NOT NULL DEFAULT 1,
      include_contradictions BOOLEAN NOT NULL DEFAULT 1,
      include_reflections BOOLEAN NOT NULL DEFAULT 1,
      last_sent_at TEXT,
      next_send_at TEXT,
      active BOOLEAN NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_digests_next ON digests(next_send_at) WHERE active = 1;
    CREATE INDEX IF NOT EXISTS idx_digests_user ON digests(user_id);

    CREATE TABLE IF NOT EXISTS reflections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL DEFAULT 1,
      content TEXT NOT NULL,
      themes TEXT,
      period_start TEXT NOT NULL,
      period_end TEXT NOT NULL,
      memory_count INTEGER NOT NULL DEFAULT 0,
      source_memory_ids TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_reflections_user ON reflections(user_id);
    CREATE INDEX IF NOT EXISTS idx_reflections_period ON reflections(period_end DESC);
  `);
} catch {}

// ============================================================================
// SCHEMA v4 — Multi-tenant: users, API keys, spaces
// ============================================================================

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    is_admin BOOLEAN NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT 'default',
    scopes TEXT NOT NULL DEFAULT 'read,write',
    rate_limit INTEGER NOT NULL DEFAULT ${DEFAULT_RATE_LIMIT},
    is_active BOOLEAN NOT NULL DEFAULT 1,
    last_used_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
  CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

  CREATE TABLE IF NOT EXISTS spaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, name)
  );
  CREATE INDEX IF NOT EXISTS idx_spaces_user ON spaces(user_id);
`);

// v4 migrations — add user_id and space_id columns
for (const [tbl, col, def] of [
  ["memories", "user_id", "INTEGER NOT NULL DEFAULT 1"],
  ["memories", "space_id", "INTEGER"],
  ["conversations", "user_id", "INTEGER NOT NULL DEFAULT 1"],
] as const) {
  try { db.exec(`ALTER TABLE ${tbl} ADD COLUMN ${col} ${def}`); } catch {}
}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_space ON memories(space_id)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)"); } catch {}

// Ensure default user exists (backwards compat — all existing data is user_id=1)
const defaultUser = db.prepare("SELECT id FROM users WHERE id = 1").get();
if (!defaultUser) {
  db.exec("INSERT INTO users (id, username, is_admin) VALUES (1, 'owner', 1)");
  console.log("Created default owner user (id=1)");
}

// Ensure default space for owner
const defaultSpace = db.prepare("SELECT id FROM spaces WHERE user_id = 1 AND name = 'default'").get();
if (!defaultSpace) {
  db.exec("INSERT INTO spaces (user_id, name, description) VALUES (1, 'default', 'Default memory space')");
  console.log("Created default space for owner");
}

// ============================================================================
// MEMORY LINKS TABLE — v3 with typed relationships
// ============================================================================

db.exec(`
  CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    similarity REAL NOT NULL,
    type TEXT NOT NULL DEFAULT 'similarity',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, type)
  );
  CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
  CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);
`);

// v3 migration: add type column
try { db.exec("ALTER TABLE memory_links ADD COLUMN type TEXT NOT NULL DEFAULT 'similarity'"); } catch {}

// ============================================================================
// CONVERSATIONS TABLE (unchanged from v2)
// ============================================================================

db.exec(`
  CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    session_id TEXT,
    title TEXT,
    metadata TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_conv_agent ON conversations(agent);
  CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
  CREATE INDEX IF NOT EXISTS idx_conv_started ON conversations(started_at DESC);

  CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at);

  CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    role,
    content='messages',
    content_rowid='id',
    tokenize='porter unicode61'
  );

  CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, role)
    VALUES (new.id, new.content, new.role);
  END;

  CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, role)
    VALUES ('delete', old.id, old.content, old.role);
  END;
`);

// ============================================================================
// PREPARED STATEMENTS — memories (v3)
// ============================================================================

const insertMemory = db.prepare(
  `INSERT INTO memories (content, category, source, session_id, importance, embedding,
    version, is_latest, parent_memory_id, root_memory_id, source_count, is_static,
    is_forgotten, forget_after, forget_reason, is_inference)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
   RETURNING id, created_at`
);

const updateMemoryEmbedding = db.prepare(
  `UPDATE memories SET embedding = ? WHERE id = ?`
);

const updateMemoryVec = db.prepare(
  `UPDATE memories SET embedding_vec = vector(?) WHERE id = ?`
);

/** Write vector column for a newly inserted memory (call after insertMemory) */
function writeVec(memoryId: number, embArray: Float32Array | null): void {
  if (!embArray) return;
  try { updateMemoryVec.run(embeddingToVectorJSON(embArray), memoryId); } catch {}
}

const getAllEmbeddings = db.prepare(
  `SELECT id, content, category, importance, embedding, is_latest, is_forgotten, is_static, source_count
   FROM memories WHERE embedding IS NOT NULL AND is_forgotten = 0`
);

const getLatestEmbeddings = db.prepare(
  `SELECT id, content, category, importance, embedding, is_static, source_count
   FROM memories WHERE embedding IS NOT NULL AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND status = 'approved'`
);

const searchMemoriesFTS = db.prepare(
  `SELECT m.id, m.content, m.category, m.source, m.session_id, m.importance, m.created_at,
     m.version, m.is_latest, m.parent_memory_id, m.root_memory_id, m.source_count,
     m.is_static, m.is_forgotten, m.is_inference,
     rank as fts_rank
   FROM memories_fts f
   JOIN memories m ON f.rowid = m.id
   WHERE memories_fts MATCH ? AND m.is_forgotten = 0
   ORDER BY rank
   LIMIT ?`
);

const listRecent = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at,
     version, is_latest, parent_memory_id, root_memory_id, source_count,
     is_static, is_forgotten, is_inference, forget_after, is_archived, status
   FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND status != 'pending' AND user_id = ? ORDER BY created_at DESC LIMIT ?`
);

const listByCategory = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at,
     version, is_latest, parent_memory_id, root_memory_id, source_count,
     is_static, is_forgotten, is_inference, forget_after, is_archived, status
   FROM memories WHERE category = ? AND is_forgotten = 0 AND is_archived = 0 AND status != 'pending' AND user_id = ? ORDER BY created_at DESC LIMIT ?`
);

const deleteMemory = db.prepare(`DELETE FROM memories WHERE id = ?`);
const getMemory = db.prepare(`SELECT * FROM memories WHERE id = ?`);

const getMemoryWithoutEmbedding = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at, updated_at,
     version, is_latest, parent_memory_id, root_memory_id, source_count,
     is_static, is_forgotten, forget_after, forget_reason, is_inference, is_archived
   FROM memories WHERE id = ?`
);

// Version chain queries
const getVersionChain = db.prepare(
  `SELECT id, content, category, version, is_latest, created_at, source_count
   FROM memories WHERE root_memory_id = ? OR id = ?
   ORDER BY version ASC`
);

const markSuperseded = db.prepare(
  `UPDATE memories SET is_latest = 0, updated_at = datetime('now') WHERE id = ?`
);

const incrementSourceCount = db.prepare(
  `UPDATE memories SET source_count = source_count + 1, updated_at = datetime('now') WHERE id = ?`
);

// Forgetting
const markForgotten = db.prepare(
  `UPDATE memories SET is_forgotten = 1, updated_at = datetime('now') WHERE id = ?`
);

const markArchived = db.prepare(
  `UPDATE memories SET is_archived = 1, updated_at = datetime('now') WHERE id = ?`
);

const markUnarchived = db.prepare(
  `UPDATE memories SET is_archived = 0, updated_at = datetime('now') WHERE id = ?`
);

const getExpiredMemories = db.prepare(
  `SELECT id, content, forget_reason FROM memories
   WHERE forget_after IS NOT NULL AND forget_after <= datetime('now')
   AND is_forgotten = 0`
);

// Memory links — v3 typed
const insertLink = db.prepare(
  `INSERT OR IGNORE INTO memory_links (source_id, target_id, similarity, type) VALUES (?, ?, ?, ?)`
);

const getLinksFor = db.prepare(
  `SELECT ml.target_id as id, ml.similarity, ml.type, m.content, m.category, m.importance, m.created_at,
     m.is_latest, m.is_forgotten, m.version, m.source_count
   FROM memory_links ml
   JOIN memories m ON ml.target_id = m.id
   WHERE ml.source_id = ?
   UNION
   SELECT ml.source_id as id, ml.similarity, ml.type, m.content, m.category, m.importance, m.created_at,
     m.is_latest, m.is_forgotten, m.version, m.source_count
   FROM memory_links ml
   JOIN memories m ON ml.source_id = m.id
   WHERE ml.target_id = ?
   ORDER BY similarity DESC`
);

const countNoEmbedding = db.prepare(
  `SELECT COUNT(*) as count FROM memories WHERE embedding IS NULL`
);
const getNoEmbedding = db.prepare(
  `SELECT id, content FROM memories WHERE embedding IS NULL LIMIT ?`
);

// Profile queries
const getStaticMemories = db.prepare(
  `SELECT id, content, category, source_count, created_at, updated_at
   FROM memories WHERE is_static = 1 AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND status = 'approved' AND user_id = ?
   ORDER BY source_count DESC, updated_at DESC`
);

// Access tracking + FSRS review processing
const trackAccess = db.prepare(
  `UPDATE memories SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?`
);
const updateFSRS = db.prepare(
  `UPDATE memories SET fsrs_stability = ?, fsrs_difficulty = ?, fsrs_storage_strength = ?,
   fsrs_retrieval_strength = ?, fsrs_learning_state = ?, fsrs_reps = ?, fsrs_lapses = ?,
   fsrs_last_review_at = ? WHERE id = ?`
);
const getFSRS = db.prepare(
  `SELECT fsrs_stability, fsrs_difficulty, fsrs_storage_strength, fsrs_retrieval_strength,
   fsrs_learning_state, fsrs_reps, fsrs_lapses, fsrs_last_review_at, last_accessed_at, created_at
   FROM memories WHERE id = ?`
);

/** Track access AND process as FSRS review (Grade: Good=recall, Again=forget) */
function trackAccessWithFSRS(memoryId: number, grade: FSRSRating = FSRSRating.Good): void {
  trackAccess.run(memoryId);
  const row = getFSRS.get(memoryId) as any;
  if (!row) return;

  const state: FSRSMemoryState | null = row.fsrs_stability != null ? {
    stability: row.fsrs_stability, difficulty: row.fsrs_difficulty,
    storage_strength: row.fsrs_storage_strength ?? 1, retrieval_strength: row.fsrs_retrieval_strength ?? 1,
    learning_state: row.fsrs_learning_state ?? 0, reps: row.fsrs_reps ?? 0, lapses: row.fsrs_lapses ?? 0,
    last_review_at: row.fsrs_last_review_at ?? row.created_at,
  } : null;

  const refTime = state?.last_review_at || row.last_accessed_at || row.created_at;
  const elapsed = (Date.now() - new Date(refTime + "Z").getTime()) / 86400000;
  const newState = fsrsProcessReview(state, grade, elapsed);

  updateFSRS.run(
    newState.stability, newState.difficulty, newState.storage_strength,
    newState.retrieval_strength, newState.learning_state, newState.reps, newState.lapses,
    newState.last_review_at, memoryId
  );
}

// Tags
const getByTag = db.prepare(
  `SELECT id, content, category, source, importance, created_at, tags, access_count, episode_id
   FROM memories WHERE tags LIKE ? AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
   ORDER BY created_at DESC LIMIT ?`
);

const getAllTags = db.prepare(
  `SELECT DISTINCT tags FROM memories WHERE tags IS NOT NULL AND is_forgotten = 0 AND is_archived = 0 AND user_id = ?`
);

// Inbox / Review queue prepared statements
const listPending = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at, tags, confidence, decay_score, status
   FROM memories WHERE status = 'pending' AND is_forgotten = 0 AND user_id = ?
   ORDER BY created_at DESC LIMIT ? OFFSET ?`
);
const countPending = db.prepare(
  `SELECT COUNT(*) as count FROM memories WHERE status = 'pending' AND is_forgotten = 0 AND user_id = ?`
);
const approveMemory = db.prepare(
  `UPDATE memories SET status = 'approved', updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);
const rejectMemory = db.prepare(
  `UPDATE memories SET status = 'rejected', is_archived = 1, updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);

// Episodes
const insertEpisode = db.prepare(
  `INSERT INTO episodes (title, session_id, agent, user_id) VALUES (?, ?, ?, ?) RETURNING id, started_at`
);
const updateEpisode = db.prepare(
  `UPDATE episodes SET title = COALESCE(?, title), summary = COALESCE(?, summary),
   ended_at = COALESCE(?, ended_at), memory_count = (SELECT COUNT(*) FROM memories WHERE episode_id = episodes.id)
   WHERE id = ?`
);
const getEpisode = db.prepare(`SELECT * FROM episodes WHERE id = ?`);
const getEpisodeBySession = db.prepare(
  `SELECT * FROM episodes WHERE session_id = ? AND agent = ? AND user_id = ? ORDER BY started_at DESC LIMIT 1`
);
const listEpisodes = db.prepare(
  `SELECT * FROM episodes WHERE user_id = ? ORDER BY started_at DESC LIMIT ?`
);
const getEpisodeMemories = db.prepare(
  `SELECT id, content, category, source, importance, created_at, tags, access_count
   FROM memories WHERE episode_id = ? AND is_forgotten = 0 ORDER BY created_at ASC`
);
const assignToEpisode = db.prepare(
  `UPDATE memories SET episode_id = ? WHERE id = ?`
);

// Consolidation queries
const getClusterCandidates = db.prepare(
  `SELECT source_id, COUNT(*) as link_count FROM memory_links
   JOIN memories m ON memory_links.source_id = m.id
   WHERE m.is_forgotten = 0 AND m.is_archived = 0 AND m.is_latest = 1
   GROUP BY source_id HAVING link_count >= ?
   ORDER BY link_count DESC LIMIT 10`
);

const getClusterMembers = db.prepare(
  `SELECT DISTINCT m.id, m.content, m.category, m.importance, m.created_at, m.access_count
   FROM memory_links ml
   JOIN memories m ON (ml.target_id = m.id OR ml.source_id = m.id)
   WHERE (ml.source_id = ? OR ml.target_id = ?) AND m.is_forgotten = 0 AND m.is_archived = 0
   ORDER BY m.importance DESC, m.created_at DESC`
);

// Confidence updates
const updateConfidence = db.prepare(
  `UPDATE memories SET confidence = ?, updated_at = datetime('now') WHERE id = ?`
);

// Webhook queries
const insertWebhook = db.prepare(
  `INSERT INTO webhooks (url, events, secret, user_id) VALUES (?, ?, ?, ?) RETURNING id, created_at`
);
const listWebhooks = db.prepare(
  `SELECT id, url, events, active, last_triggered_at, failure_count, created_at
   FROM webhooks WHERE user_id = ? ORDER BY created_at DESC`
);
const deleteWebhook = db.prepare(`DELETE FROM webhooks WHERE id = ? AND user_id = ?`);
const getActiveWebhooks = db.prepare(
  `SELECT id, url, events, secret FROM webhooks WHERE active = 1 AND user_id = ?`
);
const webhookTriggered = db.prepare(
  `UPDATE webhooks SET last_triggered_at = datetime('now') WHERE id = ?`
);
const webhookFailed = db.prepare(
  `UPDATE webhooks SET failure_count = failure_count + 1,
   active = CASE WHEN failure_count >= 9 THEN 0 ELSE active END WHERE id = ?`
);

// Sync queries
const getChangesSince = db.prepare(
  `SELECT id, content, category, source, session_id, importance, tags, confidence,
     sync_id, is_static, is_forgotten, is_archived, version, created_at, updated_at
   FROM memories WHERE updated_at > ? AND user_id = ?
   ORDER BY updated_at ASC LIMIT ?`
);
const getMemoryBySyncId = db.prepare(
  `SELECT id, updated_at FROM memories WHERE sync_id = ?`
);

// Entity queries
const insertEntity = db.prepare(
  `INSERT INTO entities (name, type, description, aka, metadata, user_id)
   VALUES (?, ?, ?, ?, ?, ?) RETURNING id, created_at`
);
const getEntity = db.prepare(
  `SELECT e.*, GROUP_CONCAT(DISTINCT me.memory_id) as memory_ids
   FROM entities e LEFT JOIN memory_entities me ON me.entity_id = e.id
   WHERE e.id = ? GROUP BY e.id`
);
const listEntities = db.prepare(
  `SELECT e.id, e.name, e.type, e.description, e.aka, e.created_at,
     (SELECT COUNT(*) FROM memory_entities WHERE entity_id = e.id) as memory_count
   FROM entities e WHERE e.user_id = ? ORDER BY e.name COLLATE NOCASE`
);
const listEntitiesByType = db.prepare(
  `SELECT e.id, e.name, e.type, e.description, e.aka, e.created_at,
     (SELECT COUNT(*) FROM memory_entities WHERE entity_id = e.id) as memory_count
   FROM entities e WHERE e.user_id = ? AND e.type = ? ORDER BY e.name COLLATE NOCASE`
);
const searchEntities = db.prepare(
  `SELECT e.id, e.name, e.type, e.description, e.aka, e.created_at,
     (SELECT COUNT(*) FROM memory_entities WHERE entity_id = e.id) as memory_count
   FROM entities e WHERE e.user_id = ? AND (e.name LIKE ? OR e.aka LIKE ? OR e.description LIKE ?)
   ORDER BY e.name COLLATE NOCASE LIMIT ?`
);
const updateEntity = db.prepare(
  `UPDATE entities SET name = COALESCE(?, name), type = COALESCE(?, type),
   description = COALESCE(?, description), aka = COALESCE(?, aka),
   metadata = COALESCE(?, metadata), updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);
const deleteEntity = db.prepare(`DELETE FROM entities WHERE id = ? AND user_id = ?`);
const linkMemoryEntity = db.prepare(
  `INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)`
);
const unlinkMemoryEntity = db.prepare(
  `DELETE FROM memory_entities WHERE memory_id = ? AND entity_id = ?`
);
const getEntityMemories = db.prepare(
  `SELECT m.id, m.content, m.category, m.importance, m.tags, m.created_at, m.decay_score, m.confidence
   FROM memories m JOIN memory_entities me ON me.memory_id = m.id
   WHERE me.entity_id = ? AND m.is_forgotten = 0 AND m.is_archived = 0
   ORDER BY m.created_at DESC LIMIT ?`
);
const insertEntityRelationship = db.prepare(
  `INSERT OR IGNORE INTO entity_relationships (source_entity_id, target_entity_id, relationship) VALUES (?, ?, ?)`
);
const deleteEntityRelationship = db.prepare(
  `DELETE FROM entity_relationships WHERE source_entity_id = ? AND target_entity_id = ? AND relationship = ?`
);
const getEntityRelationships = db.prepare(
  `SELECT er.id, er.relationship, er.created_at,
     CASE WHEN er.source_entity_id = ? THEN er.target_entity_id ELSE er.source_entity_id END as related_entity_id,
     CASE WHEN er.source_entity_id = ? THEN 'outgoing' ELSE 'incoming' END as direction,
     e.name as related_entity_name, e.type as related_entity_type
   FROM entity_relationships er
   JOIN entities e ON e.id = CASE WHEN er.source_entity_id = ? THEN er.target_entity_id ELSE er.source_entity_id END
   WHERE er.source_entity_id = ? OR er.target_entity_id = ?`
);

// Project queries
const insertProject = db.prepare(
  `INSERT INTO projects (name, description, status, metadata, user_id)
   VALUES (?, ?, ?, ?, ?) RETURNING id, created_at`
);
const getProject = db.prepare(
  `SELECT p.*, GROUP_CONCAT(DISTINCT mp.memory_id) as memory_ids
   FROM projects p LEFT JOIN memory_projects mp ON mp.project_id = p.id
   WHERE p.id = ? GROUP BY p.id`
);
const listProjects = db.prepare(
  `SELECT p.id, p.name, p.description, p.status, p.created_at,
     (SELECT COUNT(*) FROM memory_projects WHERE project_id = p.id) as memory_count
   FROM projects p WHERE p.user_id = ? ORDER BY p.status = 'active' DESC, p.name COLLATE NOCASE`
);
const listProjectsByStatus = db.prepare(
  `SELECT p.id, p.name, p.description, p.status, p.created_at,
     (SELECT COUNT(*) FROM memory_projects WHERE project_id = p.id) as memory_count
   FROM projects p WHERE p.user_id = ? AND p.status = ? ORDER BY p.name COLLATE NOCASE`
);
const updateProject = db.prepare(
  `UPDATE projects SET name = COALESCE(?, name), description = COALESCE(?, description),
   status = COALESCE(?, status), metadata = COALESCE(?, metadata),
   updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);
const deleteProject = db.prepare(`DELETE FROM projects WHERE id = ? AND user_id = ?`);
const linkMemoryProject = db.prepare(
  `INSERT OR IGNORE INTO memory_projects (memory_id, project_id) VALUES (?, ?)`
);
const unlinkMemoryProject = db.prepare(
  `DELETE FROM memory_projects WHERE memory_id = ? AND project_id = ?`
);
const getProjectMemories = db.prepare(
  `SELECT m.id, m.content, m.category, m.importance, m.tags, m.created_at, m.decay_score, m.confidence
   FROM memories m JOIN memory_projects mp ON mp.memory_id = m.id
   WHERE mp.project_id = ? AND m.is_forgotten = 0 AND m.is_archived = 0
   ORDER BY m.created_at DESC LIMIT ?`
);

const getRecentDynamicMemories = db.prepare(
  `SELECT id, content, category, source_count, created_at
   FROM memories WHERE is_static = 0 AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
   ORDER BY created_at DESC LIMIT ?`
);

// Graph data
const getAllMemoriesForGraph = db.prepare(
  `SELECT id, content, category, importance, is_latest, is_forgotten, is_static,
     is_inference, version, parent_memory_id, root_memory_id, source_count,
     forget_after, created_at, tags, access_count, episode_id, decay_score
   FROM memories WHERE user_id = ? ORDER BY created_at DESC`
);

const getAllLinksForGraph = db.prepare(
  `SELECT ml.source_id, ml.target_id, ml.similarity, ml.type FROM memory_links ml
   JOIN memories m ON ml.source_id = m.id WHERE m.user_id = ?`
);

// ============================================================================
// PREPARED STATEMENTS — conversations (unchanged)
// ============================================================================

const insertConversation = db.prepare(
  `INSERT INTO conversations (agent, session_id, title, metadata) VALUES (?, ?, ?, ?) RETURNING id, started_at`
);
const updateConversation = db.prepare(
  `UPDATE conversations SET title = COALESCE(?, title), metadata = COALESCE(?, metadata), updated_at = datetime('now') WHERE id = ?`
);
const getConversation = db.prepare(`SELECT * FROM conversations WHERE id = ?`);
const getConversationBySession = db.prepare(
  `SELECT * FROM conversations WHERE agent = ? AND session_id = ? ORDER BY started_at DESC LIMIT 1`
);
const listConversations = db.prepare(
  `SELECT c.id, c.agent, c.session_id, c.title, c.metadata, c.started_at, c.updated_at,
     (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
   FROM conversations c ORDER BY c.updated_at DESC LIMIT ?`
);
const listConversationsByAgent = db.prepare(
  `SELECT c.id, c.agent, c.session_id, c.title, c.metadata, c.started_at, c.updated_at,
     (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
   FROM conversations c WHERE c.agent = ? ORDER BY c.updated_at DESC LIMIT ?`
);
const deleteConversation = db.prepare(`DELETE FROM conversations WHERE id = ?`);
const insertMessage = db.prepare(
  `INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?) RETURNING id, created_at`
);
const getMessages = db.prepare(
  `SELECT id, role, content, metadata, created_at FROM messages
   WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?`
);
const searchMessages = db.prepare(
  `SELECT m.id, m.conversation_id, m.role, m.content, m.metadata, m.created_at,
     c.agent, c.title as conv_title
   FROM messages_fts f
   JOIN messages m ON f.rowid = m.id
   JOIN conversations c ON m.conversation_id = c.id
   WHERE messages_fts MATCH ?
   ORDER BY m.created_at DESC
   LIMIT ?`
);
const touchConversation = db.prepare(
  `UPDATE conversations SET updated_at = datetime('now') WHERE id = ?`
);

const bulkInsertConvo = db.transaction(
  (agent: string, sessionId: string | null, title: string | null, metadata: string | null,
   msgs: Array<{ role: string; content: string; metadata?: string | null }>) => {
    const conv = insertConversation.get(agent, sessionId, title, metadata) as { id: number; started_at: string };
    for (const msg of msgs) {
      insertMessage.run(conv.id, msg.role, msg.content, msg.metadata || null);
    }
    return conv;
  }
);

// ============================================================================
// LLM FACT EXTRACTION
// ============================================================================

interface FactExtractionResult {
  facts: Array<{
    content: string;
    category: string;
    is_static: boolean;
    forget_after?: string | null;
    forget_reason?: string | null;
    importance: number;
  }>;
  relation_to_existing: {
    type: "none" | "updates" | "extends" | "duplicate" | "contradicts" | "caused_by" | "prerequisite_for";
    existing_memory_id?: number | null;
    reason?: string;
  };
}

async function callLLM(systemPrompt: string, userPrompt: string): Promise<string> {
  if (!LLM_API_KEY) throw new Error("LLM_API_KEY not configured");

  const resp = await fetch(LLM_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${LLM_API_KEY}`,
    },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      max_tokens: 2000,
      temperature: 0.1,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`LLM request failed (${resp.status}): ${text}`);
  }

  const data = await resp.json() as any;
  return data.choices?.[0]?.message?.content || "";
}

const FACT_EXTRACTION_PROMPT = `You are a fact extraction engine for a persistent memory system. Your job is to analyze new content being stored and compare it with existing memories.

Given the NEW CONTENT and up to 3 SIMILAR EXISTING MEMORIES, you must:
1. Determine if this new content updates, extends, or duplicates any existing memory
2. Classify whether each fact is STATIC (permanent, unlikely to change — like preferences, identity, infrastructure) or DYNAMIC (temporary, likely to change — like current tasks, recent events, moods)
3. For dynamic facts, estimate when they should be forgotten (if applicable)
4. Rate importance 1-10

Respond with ONLY valid JSON (no markdown, no backticks):
{
  "facts": [
    {
      "content": "extracted fact text",
      "category": "task|discovery|decision|state|issue",
      "is_static": true/false,
      "forget_after": "ISO datetime or null",
      "forget_reason": "reason or null",
      "importance": 1-10
    }
  ],
  "tags": ["lowercase", "keyword", "tags"],
  "relation_to_existing": {
    "type": "none|updates|extends|duplicate|contradicts|caused_by|prerequisite_for",
    "existing_memory_id": number_or_null,
    "reason": "why this relation was determined"
  }
}

Rules:
- "updates" = new content contradicts or supersedes an existing memory
- "extends" = new content adds to/enriches an existing memory without contradicting it
- "duplicate" = new content says essentially the same thing as an existing memory
- "contradicts" = new content directly conflicts with an existing memory (both may be valid)
- "caused_by" = new content describes something that was caused by an existing memory's event
- "prerequisite_for" = existing memory describes something that requires this new content first
- "none" = no meaningful relation to any existing memory
- For forget_after: use ISO 8601 datetime. Tasks/events might expire in days-weeks. Permanent facts = null.
- The "facts" array should contain the KEY discrete facts from the content (1-3 facts usually)
- Category should match the original content's category if it makes sense
- Include a "tags" array: 2-5 lowercase keywords that classify this memory (e.g. ["database", "postgresql", "migration"])`;

async function extractFacts(
  content: string,
  category: string,
  similarMemories: Array<{ id: number; content: string; category: string; score: number }>
): Promise<FactExtractionResult | null> {
  try {
    let userPrompt = `NEW CONTENT (category: ${category}):\n${content}\n\n`;

    if (similarMemories.length > 0) {
      userPrompt += "SIMILAR EXISTING MEMORIES:\n";
      for (const m of similarMemories) {
        userPrompt += `[ID: ${m.id}, category: ${m.category}, similarity: ${m.score.toFixed(3)}]\n${m.content}\n\n`;
      }
    } else {
      userPrompt += "SIMILAR EXISTING MEMORIES: none found\n";
    }

    const response = await callLLM(FACT_EXTRACTION_PROMPT, userPrompt);

    let jsonStr = response.trim();
    if (jsonStr.startsWith("```")) {
      jsonStr = jsonStr.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
    }

    const result = JSON.parse(jsonStr) as FactExtractionResult;
    return result;
  } catch (e: any) {
    console.error("Fact extraction failed:", e.message);
    return null;
  }
}

// ============================================================================
// PROCESS FACT EXTRACTION RESULTS
// ============================================================================

function processExtractionResult(
  newMemoryId: number,
  result: FactExtractionResult,
  embArray: Float32Array | null
): void {
  const rel = result.relation_to_existing;

  if (rel.type === "duplicate" && rel.existing_memory_id) {
    const existing = getMemoryWithoutEmbedding.get(rel.existing_memory_id) as any;
    if (existing && !existing.is_forgotten) {
      incrementSourceCount.run(rel.existing_memory_id);
      markSuperseded.run(newMemoryId);
      insertLink.run(newMemoryId, rel.existing_memory_id, 1.0, "derives");
      console.log(`Memory #${newMemoryId} is duplicate of #${rel.existing_memory_id}, incremented source_count`);
      return;
    }
  }

  if (rel.type === "updates" && rel.existing_memory_id) {
    const existing = getMemoryWithoutEmbedding.get(rel.existing_memory_id) as any;
    if (existing) {
      markSuperseded.run(rel.existing_memory_id);
      const rootId = existing.root_memory_id || existing.id;
      const newVersion = (existing.version || 1) + 1;
      db.prepare(
        `UPDATE memories SET version = ?, root_memory_id = ?, parent_memory_id = ?, is_latest = 1
         WHERE id = ?`
      ).run(newVersion, rootId, existing.id, newMemoryId);
      insertLink.run(newMemoryId, rel.existing_memory_id, 1.0, "updates");
      propagateConfidence(newMemoryId, "updates", rel.existing_memory_id);
      console.log(`Memory #${newMemoryId} updates #${rel.existing_memory_id} (v${newVersion}, root=#${rootId})`);
    }
  }

  if (rel.type === "extends" && rel.existing_memory_id) {
    insertLink.run(newMemoryId, rel.existing_memory_id, 0.9, "extends");
    propagateConfidence(newMemoryId, "extends", rel.existing_memory_id);
    console.log(`Memory #${newMemoryId} extends #${rel.existing_memory_id}`);
  }

  if (rel.type === "contradicts" && rel.existing_memory_id) {
    insertLink.run(newMemoryId, rel.existing_memory_id, 0.85, "contradicts");
    insertLink.run(rel.existing_memory_id, newMemoryId, 0.85, "contradicts");
    propagateConfidence(newMemoryId, "contradicts", rel.existing_memory_id);
    console.log(`Memory #${newMemoryId} contradicts #${rel.existing_memory_id}`);
  }

  if (rel.type === "caused_by" && rel.existing_memory_id) {
    insertLink.run(newMemoryId, rel.existing_memory_id, 0.8, "caused_by");
    console.log(`Memory #${newMemoryId} caused by #${rel.existing_memory_id}`);
  }

  if (rel.type === "prerequisite_for" && rel.existing_memory_id) {
    insertLink.run(rel.existing_memory_id, newMemoryId, 0.8, "prerequisite_for");
    console.log(`Memory #${rel.existing_memory_id} is prerequisite for #${newMemoryId}`);
  }

  // Apply fact classifications to the memory
  if (result.facts.length > 0) {
    const primaryFact = result.facts[0];
    db.prepare(
      `UPDATE memories SET is_static = ?, forget_after = ?, forget_reason = ?,
       importance = CASE WHEN importance = 5 THEN ? ELSE importance END,
       updated_at = datetime('now') WHERE id = ?`
    ).run(
      primaryFact.is_static ? 1 : 0,
      primaryFact.forget_after || null,
      primaryFact.forget_reason || null,
      primaryFact.importance,
      newMemoryId
    );
  }

  // Auto-tagging: apply LLM-inferred tags if present
  if ((result as any).tags && Array.isArray((result as any).tags)) {
    const inferred = (result as any).tags.map((t: any) => String(t).trim().toLowerCase()).filter(Boolean);
    if (inferred.length > 0) {
      // Merge with existing tags
      const mem = getMemoryWithoutEmbedding.get(newMemoryId) as any;
      let existing: string[] = [];
      if (mem?.tags) try { existing = JSON.parse(mem.tags); } catch {}
      const merged = [...new Set([...existing, ...inferred])];
      db.prepare("UPDATE memories SET tags = ? WHERE id = ?").run(JSON.stringify(merged), newMemoryId);
      console.log(`Auto-tagged memory #${newMemoryId}: [${merged.join(", ")}]`);
    }
  }
}

// ============================================================================
// FSRS-6 — Free Spaced Repetition Scheduler (ported from Vestige/Rust)
// Power-law forgetting curve, 20-30% more efficient than SM-2/exponential
// Reference: https://github.com/open-spaced-repetition/fsrs4anki
// ============================================================================

// 21 default weights trained on millions of Anki reviews
const FSRS6_WEIGHTS: number[] = [
  0.212,   // w0:  Initial stability for Again
  1.2931,  // w1:  Initial stability for Hard
  2.3065,  // w2:  Initial stability for Good
  8.2956,  // w3:  Initial stability for Easy
  6.4133,  // w4:  Initial difficulty base
  0.8334,  // w5:  Initial difficulty grade modifier
  3.0194,  // w6:  Difficulty delta
  0.001,   // w7:  Difficulty mean reversion
  1.8722,  // w8:  Stability increase base
  0.1666,  // w9:  Stability saturation
  0.796,   // w10: Retrievability influence on stability
  1.4835,  // w11: Forget stability base
  0.0614,  // w12: Forget difficulty influence
  0.2629,  // w13: Forget stability influence
  1.6483,  // w14: Forget retrievability influence
  0.6014,  // w15: Hard penalty
  1.8729,  // w16: Easy bonus
  0.5425,  // w17: Same-day review base (FSRS-6)
  0.0912,  // w18: Same-day review grade modifier (FSRS-6)
  0.0658,  // w19: Same-day review stability influence (FSRS-6)
  0.1542,  // w20: Forgetting curve decay (FSRS-6, personalizable)
];

const FSRS_MIN_STABILITY = 0.1;
const FSRS_MAX_STABILITY = 36500; // 100 years
const FSRS_MIN_DIFFICULTY = 1.0;
const FSRS_MAX_DIFFICULTY = 10.0;
const FSRS_MAX_STORAGE = 10.0;

enum FSRSRating { Again = 1, Hard = 2, Good = 3, Easy = 4 }
enum FSRSState { New = 0, Learning = 1, Review = 2, Relearning = 3 }

/** Forgetting factor: 0.9^(-1/w20) - 1 */
function fsrsForgettingFactor(w20: number = FSRS6_WEIGHTS[20]): number {
  return Math.pow(0.9, -1.0 / w20) - 1.0;
}

/**
 * FSRS-6 Retrievability — probability of recall at time t
 * R = (1 + factor * t / S)^(-w20)  — power law, more accurate than exponential
 */
function fsrsRetrievability(stability: number, elapsedDays: number, w20: number = FSRS6_WEIGHTS[20]): number {
  if (stability <= 0) return 0;
  if (elapsedDays <= 0) return 1;
  const factor = fsrsForgettingFactor(w20);
  return Math.max(0, Math.min(1, Math.pow(1 + factor * elapsedDays / stability, -w20)));
}

/** Initial difficulty: D0(G) = w4 - e^(w5*(G-1)) + 1 */
function fsrsInitialDifficulty(grade: FSRSRating): number {
  const d = FSRS6_WEIGHTS[4] - Math.exp(FSRS6_WEIGHTS[5] * (grade - 1)) + 1;
  return Math.max(FSRS_MIN_DIFFICULTY, Math.min(FSRS_MAX_DIFFICULTY, d));
}

/** Initial stability: S0(G) = w[G-1] */
function fsrsInitialStability(grade: FSRSRating): number {
  return Math.max(FSRS_MIN_STABILITY, FSRS6_WEIGHTS[grade - 1]);
}

/** Next difficulty with mean reversion: D' = w7*D0(4) + (1-w7)*(D + delta*((10-D)/9)) */
function fsrsNextDifficulty(currentD: number, grade: FSRSRating): number {
  const d0 = fsrsInitialDifficulty(FSRSRating.Easy);
  const delta = -FSRS6_WEIGHTS[6] * (grade - 3);
  const meanReversionScale = (10 - currentD) / 9;
  const newD = currentD + delta * meanReversionScale;
  const finalD = FSRS6_WEIGHTS[7] * d0 + (1 - FSRS6_WEIGHTS[7]) * newD;
  return Math.max(FSRS_MIN_DIFFICULTY, Math.min(FSRS_MAX_DIFFICULTY, finalD));
}

/** Stability after successful recall: S' = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) * HP * EB + 1) */
function fsrsRecallStability(S: number, D: number, R: number, grade: FSRSRating): number {
  if (grade === FSRSRating.Again) return fsrsForgetStability(D, S, R);
  const hardPenalty = grade === FSRSRating.Hard ? FSRS6_WEIGHTS[15] : 1;
  const easyBonus = grade === FSRSRating.Easy ? FSRS6_WEIGHTS[16] : 1;
  const factor = Math.exp(FSRS6_WEIGHTS[8]) * (11 - D)
    * Math.pow(S, -FSRS6_WEIGHTS[9])
    * (Math.exp(FSRS6_WEIGHTS[10] * (1 - R)) - 1)
    * hardPenalty * easyBonus + 1;
  return Math.max(FSRS_MIN_STABILITY, Math.min(FSRS_MAX_STABILITY, S * factor));
}

/** Stability after lapse: S'f = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R)) */
function fsrsForgetStability(D: number, S: number, R: number): number {
  const newS = FSRS6_WEIGHTS[11] * Math.pow(D, -FSRS6_WEIGHTS[12])
    * (Math.pow(S + 1, FSRS6_WEIGHTS[13]) - 1)
    * Math.exp(FSRS6_WEIGHTS[14] * (1 - R));
  return Math.max(FSRS_MIN_STABILITY, Math.min(Math.min(newS, S), FSRS_MAX_STABILITY));
}

/** Same-day review stability (FSRS-6): S'(S,G) = S * e^(w17*(G-3+w18)) * S^(-w19) */
function fsrsSameDayStability(S: number, grade: FSRSRating): number {
  const newS = S * Math.exp(FSRS6_WEIGHTS[17] * (grade - 3 + FSRS6_WEIGHTS[18]))
    * Math.pow(S, -FSRS6_WEIGHTS[19]);
  return Math.max(FSRS_MIN_STABILITY, Math.min(FSRS_MAX_STABILITY, newS));
}

/** Optimal review interval: t = S/factor * (R^(-1/w20) - 1) */
function fsrsNextInterval(stability: number, desiredR: number = FSRS_DEFAULT_RETENTION): number {
  if (stability <= 0 || desiredR >= 1 || desiredR <= 0) return 0;
  const factor = fsrsForgettingFactor();
  return Math.max(0, Math.round(stability / factor * (Math.pow(desiredR, -1 / FSRS6_WEIGHTS[20]) - 1)));
}

// --- Dual-Strength Model (Bjork & Bjork 1992) ---

interface DualStrength { storage: number; retrieval: number; }

function dualStrengthRetention(ds: DualStrength): number {
  return (ds.retrieval * 0.7) + ((ds.storage / FSRS_MAX_STORAGE) * 0.3);
}

function dualStrengthOnRecall(ds: DualStrength): DualStrength {
  return { storage: Math.min(ds.storage + 0.1, FSRS_MAX_STORAGE), retrieval: 1.0 };
}

function dualStrengthOnLapse(ds: DualStrength): DualStrength {
  return { storage: Math.min(ds.storage + 0.3, FSRS_MAX_STORAGE), retrieval: 1.0 };
}

function dualStrengthDecay(ds: DualStrength, elapsedDays: number, stability: number): DualStrength {
  if (elapsedDays <= 0 || stability <= 0) return ds;
  const retrieval = Math.max(0, Math.min(1,
    Math.pow(1 + elapsedDays / (9.0 * stability), -1.0 / 0.5)
  ));
  return { storage: ds.storage, retrieval };
}

// --- Review processor: maps memory access events to FSRS state updates ---

interface FSRSMemoryState {
  stability: number;
  difficulty: number;
  storage_strength: number;
  retrieval_strength: number;
  learning_state: number; // FSRSState enum
  reps: number;
  lapses: number;
  last_review_at: string;
}

function fsrsProcessReview(
  state: FSRSMemoryState | null,
  grade: FSRSRating,
  elapsedDays: number
): FSRSMemoryState {
  const now = new Date().toISOString().replace("T", " ").slice(0, 19);

  // First review (new memory)
  if (!state || state.learning_state === FSRSState.New) {
    const d = fsrsInitialDifficulty(grade);
    const s = fsrsInitialStability(grade);
    return {
      stability: s, difficulty: d,
      storage_strength: 1.0, retrieval_strength: 1.0,
      learning_state: grade <= FSRSRating.Hard ? FSRSState.Learning : FSRSState.Review,
      reps: 1, lapses: grade === FSRSRating.Again ? 1 : 0,
      last_review_at: now,
    };
  }

  const R = fsrsRetrievability(state.stability, elapsedDays);
  const isSameDay = elapsedDays < 1;
  let ds: DualStrength = { storage: state.storage_strength, retrieval: state.retrieval_strength };

  let newS: number, newD: number, newState: number, newLapses: number;

  if (isSameDay) {
    // Same-day review — FSRS-6 special handling
    newS = fsrsSameDayStability(state.stability, grade);
    newD = fsrsNextDifficulty(state.difficulty, grade);
    newState = state.learning_state;
    newLapses = state.lapses;
    ds = dualStrengthOnRecall(ds);
  } else if (grade === FSRSRating.Again) {
    // Lapse — forgot it
    newS = fsrsForgetStability(state.difficulty, state.stability, R);
    newD = fsrsNextDifficulty(state.difficulty, FSRSRating.Again);
    newState = FSRSState.Relearning;
    newLapses = state.lapses + 1;
    ds = dualStrengthOnLapse(ds);
  } else {
    // Successful recall
    newS = fsrsRecallStability(state.stability, state.difficulty, R, grade);
    newD = fsrsNextDifficulty(state.difficulty, grade);
    newState = FSRSState.Review;
    newLapses = state.lapses;
    ds = dualStrengthOnRecall(ds);
  }

  return {
    stability: Math.round(newS * 1000) / 1000,
    difficulty: Math.round(newD * 1000) / 1000,
    storage_strength: Math.round(ds.storage * 1000) / 1000,
    retrieval_strength: Math.round(ds.retrieval * 1000) / 1000,
    learning_state: newState,
    reps: state.reps + 1,
    lapses: newLapses,
    last_review_at: now,
  };
}

// --- Backward-compatible decay score using FSRS retrievability ---

function calculateDecayScore(
  importance: number,
  createdAt: string,
  accessCount: number = 0,
  lastAccessedAt: string | null = null,
  isStatic: boolean = false,
  sourceCount: number = 1,
  stability?: number
): number {
  if (isStatic) return importance;

  const now = Date.now();
  // Use last_accessed_at if available, otherwise created_at
  const refTime = lastAccessedAt
    ? new Date(lastAccessedAt + "Z").getTime()
    : new Date(createdAt + "Z").getTime();
  const elapsedDays = (now - refTime) / (1000 * 60 * 60 * 24);

  // Use FSRS stability if available, otherwise estimate from access patterns
  const effectiveStability = stability && stability > 0
    ? stability
    : fsrsInitialStability(FSRSRating.Good) * (1 + Math.min(accessCount * 0.3, 3) + Math.min((sourceCount - 1) * 0.2, 1));

  // FSRS-6 power-law retrievability instead of exponential decay
  const R = fsrsRetrievability(effectiveStability, elapsedDays);
  return importance * R;
}

function updateDecayScores(): number {
  const memories = db.prepare(
    `SELECT id, importance, created_at, access_count, last_accessed_at, is_static, source_count, fsrs_stability
     FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1`
  ).all() as Array<any>;

  let updated = 0;
  const updateDecay = db.prepare(`UPDATE memories SET decay_score = ? WHERE id = ?`);

  const batch = db.transaction(() => {
    for (const m of memories) {
      const score = calculateDecayScore(
        m.importance, m.created_at, m.access_count, m.last_accessed_at,
        !!m.is_static, m.source_count, m.fsrs_stability
      );
      updateDecay.run(Math.round(score * 1000) / 1000, m.id);
      updated++;
    }
  });
  batch();

  return updated;
}

// ============================================================================
// MEMORY CONSOLIDATION — Auto-summarize large clusters
// ============================================================================

const CONSOLIDATION_PROMPT = `You are a memory consolidation engine. Given a cluster of related memories, create a single concise summary that captures all key information.

Rules:
- Preserve ALL important facts, decisions, and specific values (versions, IPs, dates, names)
- The summary should be self-contained — someone reading only the summary should understand the full picture
- Keep it under 500 words
- Format as clear, dense paragraphs — not bullet points
- Include the most important specifics, not just generalizations

Respond with ONLY a JSON object:
{
  "summary": "the consolidated summary text",
  "title": "short 3-5 word cluster label",
  "importance": 1-10
}`;

async function consolidateCluster(
  centerMemoryId: number,
  userId: number = 1
): Promise<{ summaryId: number; archivedCount: number } | null> {
  if (!LLM_API_KEY) return null;

  const members = getClusterMembers.all(centerMemoryId, centerMemoryId) as Array<any>;
  if (members.length < CONSOLIDATION_THRESHOLD) return null;

  // Check if already consolidated
  const existing = db.prepare(
    `SELECT id FROM consolidations WHERE source_memory_ids LIKE ?`
  ).get(`%${centerMemoryId}%`) as any;
  if (existing) return null;

  const memberContents = members.map(m =>
    `[#${m.id}, ${m.category}, imp=${m.importance}]: ${m.content}`
  ).join("\n\n");

  try {
    const response = await callLLM(CONSOLIDATION_PROMPT, memberContents);
    let jsonStr = response.trim();
    if (jsonStr.startsWith("```")) {
      jsonStr = jsonStr.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
    }
    const result = JSON.parse(jsonStr) as { summary: string; title: string; importance: number };

    // Create summary memory
    const embArray = await embed(result.summary);
    const embBuffer = embeddingToBuffer(embArray);
    const imp = Math.max(1, Math.min(10, result.importance || 8));

    const summaryMem = insertMemory.get(
      `[Consolidated: ${result.title}] ${result.summary}`,
      "discovery", "consolidation", null, imp, embBuffer,
      1, 1, null, null, members.length, 1, 0, null, null, 0
    ) as { id: number; created_at: string };
    db.prepare("UPDATE memories SET user_id = ?, tags = ? WHERE id = ?").run(
      userId, JSON.stringify(["consolidated", result.title.toLowerCase().replace(/\s+/g, "-")]), summaryMem.id
    );

    // Archive source memories and link to summary
    let archived = 0;
    for (const m of members) {
      markArchived.run(m.id);
      insertLink.run(summaryMem.id, m.id, 1.0, "consolidates");
      archived++;
    }

    // Track consolidation
    db.prepare(
      `INSERT INTO consolidations (summary_memory_id, source_memory_ids, cluster_label)
       VALUES (?, ?, ?)`
    ).run(summaryMem.id, JSON.stringify(members.map(m => m.id)), result.title);

    writeVec(summaryMem.id, embArray);
    await autoLink(summaryMem.id, embArray);
    console.log(`Consolidated ${archived} memories into #${summaryMem.id}: "${result.title}"`);
    return { summaryId: summaryMem.id, archivedCount: archived };
  } catch (e: any) {
    console.error(`Consolidation failed for cluster around #${centerMemoryId}:`, e.message);
    return null;
  }
}

async function runConsolidationSweep(userId: number = 1): Promise<number> {
  const candidates = getClusterCandidates.all(CONSOLIDATION_THRESHOLD) as Array<{ source_id: number; link_count: number }>;
  let totalConsolidated = 0;
  for (const c of candidates) {
    const result = await consolidateCluster(c.source_id, userId);
    if (result) totalConsolidated += result.archivedCount;
  }
  return totalConsolidated;
}

// ============================================================================
// WEBHOOK EVENT SYSTEM
// ============================================================================

async function emitWebhookEvent(
  event: string,
  payload: Record<string, unknown>,
  userId: number = 1
): Promise<void> {
  const hooks = getActiveWebhooks.all(userId) as Array<{
    id: number; url: string; events: string; secret: string | null;
  }>;

  for (const hook of hooks) {
    try {
      const events = JSON.parse(hook.events) as string[];
      if (!events.includes("*") && !events.includes(event)) continue;

      const body = JSON.stringify({ event, timestamp: new Date().toISOString(), data: payload });
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (hook.secret) {
        const hmac = new Bun.CryptoHasher("sha256", hook.secret).update(body).digest("hex");
        headers["X-Engram-Signature"] = `sha256=${hmac}`;
      }

      fetch(hook.url, { method: "POST", headers, body, signal: AbortSignal.timeout(10000) })
        .then(resp => {
          if (resp.ok) { webhookTriggered.run(hook.id); }
          else { webhookFailed.run(hook.id); }
        })
        .catch(() => { webhookFailed.run(hook.id); });
    } catch {}
  }
}

// ============================================================================
// CONFIDENCE PROPAGATION
// ============================================================================

function propagateConfidence(memoryId: number, relationType: string, existingMemoryId: number): void {
  if (relationType === "updates") {
    // Old memory's confidence drops — it's been superseded
    updateConfidence.run(0.3, existingMemoryId);
  } else if (relationType === "contradicts") {
    // Both memories get reduced confidence — conflict needs resolution
    const existing = getMemoryWithoutEmbedding.get(existingMemoryId) as any;
    const current = getMemoryWithoutEmbedding.get(memoryId) as any;
    if (existing) {
      const newConf = Math.max(0.2, (existing.confidence || 1.0) * 0.6);
      updateConfidence.run(newConf, existingMemoryId);
    }
    if (current) {
      updateConfidence.run(0.7, memoryId); // newer info gets slight benefit of doubt
    }

    emitWebhookEvent("contradiction.detected", {
      memory_id: memoryId,
      contradicts_memory_id: existingMemoryId,
      memory_content: current?.content,
      existing_content: existing?.content,
    });
  } else if (relationType === "extends") {
    // Extended memory gets a small confidence boost — it's been corroborated
    const existing = getMemoryWithoutEmbedding.get(existingMemoryId) as any;
    if (existing) {
      const newConf = Math.min(1.0, (existing.confidence || 1.0) * 1.05);
      updateConfidence.run(newConf, existingMemoryId);
    }
  }
}

// ============================================================================
// HYBRID SEARCH — v3 with relationship expansion + version awareness
// ============================================================================

interface SearchResult {
  id: number;
  content: string;
  category: string;
  source?: string;
  importance: number;
  created_at: string;
  score: number;
  decay_score?: number;
  version?: number;
  is_latest?: boolean;
  is_static?: boolean;
  source_count?: number;
  root_memory_id?: number;
  tags?: string[];
  access_count?: number;
  episode_id?: number;
  linked?: Array<{ id: number; content: string; category: string; similarity: number; type: string }>;
  version_chain?: Array<{ id: number; content: string; version: number; is_latest: boolean }>;
}

async function hybridSearch(
  query: string,
  limit: number = 10,
  includeLinks: boolean = false,
  expandRelationships: boolean = false,
  latestOnly: boolean = true
): Promise<SearchResult[]> {
  const results = new Map<number, SearchResult>();

  // 1. Vector search (native libsql vector index)
  try {
    const queryEmb = await embed(query);
    const vecJson = embeddingToVectorJSON(queryEmb);

    // Use native vector_top_k for ANN search — O(log n) vs O(n) full scan
    const vecHits = db.prepare(`
      SELECT v.rowid, m.id, m.content, m.category, m.importance, m.is_static, m.source_count
      FROM vector_top_k('memories_vec_idx', vector(?), ?) v
      JOIN memories m ON m.rowid = v.rowid
      WHERE m.is_forgotten = 0 AND m.is_archived = 0 ${latestOnly ? "AND m.is_latest = 1" : ""}
    `).all(vecJson, limit * 5) as Array<any>;

    for (const mem of vecHits) {
      // vector_top_k returns results ordered by distance — compute similarity score
      // For cosine distance: similarity = 1 - distance (libsql returns distance)
      results.set(mem.id, {
        id: mem.id,
        content: mem.content,
        category: mem.category,
        importance: mem.importance,
        created_at: "",
        score: 0.55, // base vector match score — ANN pre-filters to relevant
        is_static: !!mem.is_static,
        source_count: mem.source_count || 1,
      });
    }

    // Fallback: if vector index has no data yet, use legacy BLOB scan
    if (vecHits.length === 0) {
      const allMems = (latestOnly ? getLatestEmbeddings : getAllEmbeddings).all() as Array<any>;
      for (const mem of allMems) {
        if (!mem.embedding) continue;
        const memEmb = bufferToEmbedding(mem.embedding);
        const sim = cosineSimilarity(queryEmb, memEmb);
        if (sim > 0.25) {
          results.set(mem.id, {
            id: mem.id, content: mem.content, category: mem.category,
            importance: mem.importance, created_at: "",
            score: sim * 0.55, is_static: !!mem.is_static,
            source_count: mem.source_count || 1,
          });
        }
      }
    }
  } catch (e: any) {
    console.error("Vector search failed:", e.message);
  }

  // 2. FTS5 keyword search
  const sanitized = sanitizeFTS(query);
  if (sanitized) {
    try {
      const ftsResults = searchMemoriesFTS.all(sanitized, limit * 3) as Array<{
        id: number; content: string; category: string; source: string;
        session_id: string; importance: number; created_at: string; fts_rank: number;
        version: number; is_latest: boolean; parent_memory_id: number; root_memory_id: number;
        source_count: number; is_static: boolean; is_forgotten: boolean; is_inference: boolean;
      }>;

      const maxRank = ftsResults.length > 0 ? Math.abs(ftsResults[0].fts_rank) : 1;

      for (const r of ftsResults) {
        const ftsScore = (Math.abs(r.fts_rank) / maxRank) * 0.35;
        const existing = results.get(r.id);
        if (existing) {
          existing.score += ftsScore;
          existing.created_at = r.created_at;
          existing.source = r.source;
          existing.version = r.version;
          existing.is_latest = !!r.is_latest;
          existing.root_memory_id = r.root_memory_id;
        } else {
          results.set(r.id, {
            id: r.id,
            content: r.content,
            category: r.category,
            source: r.source,
            importance: r.importance,
            created_at: r.created_at,
            score: ftsScore,
            version: r.version,
            is_latest: !!r.is_latest,
            is_static: !!r.is_static,
            source_count: r.source_count || 1,
            root_memory_id: r.root_memory_id,
          });
        }
      }
    } catch {}
  }

  // 3. Boost by importance + source_count + static priority + decay
  for (const r of results.values()) {
    // Use decay score instead of raw importance
    const decayScore = calculateDecayScore(
      r.importance, r.created_at, (r as any).access_count || 0, null,
      !!r.is_static, r.source_count || 1
    );
    r.decay_score = Math.round(decayScore * 1000) / 1000;
    r.score += (decayScore / 10) * 0.05;
    r.score += Math.min((r.source_count || 1) / 10, 1) * 0.03;
    if (r.is_static) r.score += 0.02;
  }

  // 4. Relationship expansion
  if (expandRelationships) {
    const topIds = Array.from(results.entries())
      .sort((a, b) => b[1].score - a[1].score)
      .slice(0, 5)
      .map(([id]) => id);

    for (const id of topIds) {
      const links = getLinksFor.all(id, id) as Array<{
        id: number; similarity: number; type: string; content: string;
        category: string; importance: number; created_at: string;
        is_latest: boolean; is_forgotten: boolean; version: number; source_count: number;
      }>;

      for (const link of links) {
        if (link.is_forgotten) continue;
        if (!results.has(link.id)) {
          const relBonus = link.type === "updates" ? 0.15 : link.type === "extends" ? 0.12 : 0.08;
          results.set(link.id, {
            id: link.id,
            content: link.content,
            category: link.category,
            importance: link.importance,
            created_at: link.created_at,
            score: relBonus * link.similarity,
            version: link.version,
            is_latest: !!link.is_latest,
            source_count: link.source_count || 1,
          });
        }
      }
    }
  }

  // 5. Sort and limit
  let sorted = Array.from(results.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  // 6. Fill missing fields
  for (const r of sorted) {
    if (!r.created_at || !r.version) {
      const mem = getMemoryWithoutEmbedding.get(r.id) as any;
      if (mem) {
        r.created_at = r.created_at || mem.created_at;
        r.source = r.source || mem.source;
        r.version = mem.version;
        r.is_latest = !!mem.is_latest;
        r.is_static = !!mem.is_static;
        r.source_count = mem.source_count;
        r.root_memory_id = mem.root_memory_id;
      }
    }
  }

  // 7. Include linked memories + version chain
  if (includeLinks) {
    for (const r of sorted) {
      const links = getLinksFor.all(r.id, r.id) as Array<{
        id: number; similarity: number; type: string; content: string; category: string;
        importance: number; created_at: string; is_latest: boolean; is_forgotten: boolean;
        version: number; source_count: number;
      }>;
      if (links.length > 0) {
        r.linked = links
          .filter(l => !l.is_forgotten)
          .map(l => ({
            id: l.id,
            content: l.content,
            category: l.category,
            similarity: Math.round(l.similarity * 1000) / 1000,
            type: l.type,
          }));
      }

      const rootId = r.root_memory_id || r.id;
      const chain = getVersionChain.all(rootId, rootId) as Array<{
        id: number; content: string; category: string; version: number; is_latest: boolean;
        created_at: string; source_count: number;
      }>;
      if (chain.length > 1) {
        r.version_chain = chain.map(c => ({
          id: c.id,
          content: c.content,
          version: c.version,
          is_latest: !!c.is_latest,
        }));
      }
    }
  }

  return sorted;
}

// ============================================================================
// AUTO-LINKING
// ============================================================================

async function autoLink(memoryId: number, embedding: Float32Array): Promise<number> {
  const vecJson = embeddingToVectorJSON(embedding);

  // Use native vector search for finding similar memories
  let similarities: Array<{ id: number; similarity: number }> = [];

  try {
    const vecHits = db.prepare(`
      SELECT v.rowid, m.id FROM vector_top_k('memories_vec_idx', vector(?), ?) v
      JOIN memories m ON m.rowid = v.rowid
      WHERE m.is_forgotten = 0 AND m.is_latest = 1 AND m.id != ?
    `).all(vecJson, AUTO_LINK_MAX + 10, memoryId) as Array<any>;

    if (vecHits.length > 0) {
      // Compute exact cosine similarity for the top candidates from ANN
      for (const hit of vecHits) {
        const row = db.prepare("SELECT embedding FROM memories WHERE id = ?").get(hit.id) as any;
        if (!row?.embedding) continue;
        const memEmb = bufferToEmbedding(row.embedding);
        const sim = cosineSimilarity(embedding, memEmb);
        if (sim >= AUTO_LINK_THRESHOLD) similarities.push({ id: hit.id, similarity: sim });
      }
    }
  } catch { /* vector index not ready yet */ }

  // Fallback to full scan if vector index returned nothing
  if (similarities.length === 0) {
    const allMems = getLatestEmbeddings.all() as Array<any>;
    for (const mem of allMems) {
      if (mem.id === memoryId || !mem.embedding) continue;
      const memEmb = bufferToEmbedding(mem.embedding);
      const sim = cosineSimilarity(embedding, memEmb);
      if (sim >= AUTO_LINK_THRESHOLD) similarities.push({ id: mem.id, similarity: sim });
    }
  }

  similarities.sort((a, b) => b.similarity - a.similarity);
  const toLink = similarities.slice(0, AUTO_LINK_MAX);

  let linked = 0;
  for (const { id, similarity } of toLink) {
    insertLink.run(memoryId, id, similarity, "similarity");
    insertLink.run(id, memoryId, similarity, "similarity");
    linked++;
  }

  return linked;
}

// ============================================================================
// AUTO-FORGET SWEEP
// ============================================================================

function sweepExpiredMemories(): number {
  const expired = getExpiredMemories.all() as Array<{ id: number; content: string; forget_reason: string }>;
  for (const mem of expired) {
    markForgotten.run(mem.id);
    console.log(`Auto-forgot memory #${mem.id}: ${mem.forget_reason || "expired"}`);
  }
  return expired.length;
}

// ============================================================================
// DYNAMIC PROFILE GENERATION
// ============================================================================

interface UserProfile {
  static_facts: Array<{ id: number; content: string; category: string; source_count: number }>;
  recent_activity: Array<{ id: number; content: string; category: string; created_at: string }>;
  summary?: string;
}

async function generateProfile(userId: number = 1, generateSummary: boolean = false): Promise<UserProfile> {
  const staticFacts = getStaticMemories.all(userId) as Array<{
    id: number; content: string; category: string; source_count: number; created_at: string; updated_at: string;
  }>;

  const recentDynamic = getRecentDynamicMemories.all(userId, 20) as Array<{
    id: number; content: string; category: string; source_count: number; created_at: string;
  }>;

  const profile: UserProfile = {
    static_facts: staticFacts.map(f => ({
      id: f.id,
      content: f.content,
      category: f.category,
      source_count: f.source_count,
    })),
    recent_activity: recentDynamic.map(f => ({
      id: f.id,
      content: f.content,
      category: f.category,
      created_at: f.created_at,
    })),
  };

  if (generateSummary && LLM_API_KEY) {
    try {
      const staticText = staticFacts.map(f => `- [${f.category}] ${f.content}`).join("\n");
      const dynamicText = recentDynamic.slice(0, 10).map(f => `- [${f.category}] ${f.content}`).join("\n");

      const summary = await callLLM(
        "You are a profile summarizer. Given a user's permanent facts and recent activity, write a concise 2-4 sentence profile summary. Be factual and direct.",
        `PERMANENT FACTS:\n${staticText || "None"}\n\nRECENT ACTIVITY:\n${dynamicText || "None"}`
      );
      profile.summary = summary.trim();
    } catch (e: any) {
      console.error("Profile summary generation failed:", e.message);
    }
  }

  return profile;
}

// ============================================================================
// BACKFILL
// ============================================================================

async function backfillEmbeddings(batchSize: number = 50): Promise<number> {
  const missing = getNoEmbedding.all(batchSize) as Array<{ id: number; content: string }>;
  let count = 0;

  for (const mem of missing) {
    try {
      const emb = await embed(mem.content);
      updateMemoryEmbedding.run(embeddingToBuffer(emb), mem.id); try { updateMemoryVec.run(embeddingToVectorJSON(emb), mem.id); } catch {}
      count++;
    } catch (e: any) {
      console.error(`Failed to embed memory #${mem.id}: ${e.message}`);
    }
  }

  if (count > 0) {
    for (const mem of missing.slice(0, count)) {
      const row = getMemory.get(mem.id) as any;
      if (row?.embedding) {
        const emb = bufferToEmbedding(row.embedding);
        await autoLink(mem.id, emb);
      }
    }
  }

  return count;
}

// ============================================================================
// HELPERS
// ============================================================================

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    },
  });
}

function errorResponse(message: string, status = 400) {
  return json({ error: message }, status);
}

function sanitizeFTS(query: string): string {
  return query
    .replace(/[^\w\s-]/g, "")
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => `"${w}"`)
    .join(" OR ");
}

// ============================================================================
// API KEY AUTHENTICATION
// ============================================================================

interface AuthContext {
  user_id: number;
  space_id: number | null;
  key_id: number | null;
  scopes: string[];
  is_admin: boolean;
}

const rateLimitMap = new Map<number, { count: number; reset: number }>();

function authenticate(req: Request): AuthContext | null {
  const authHeader = req.headers.get("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer eg_")) return null;

  const key = authHeader.slice(7); // strip "Bearer "
  const prefix = key.slice(0, 11); // "eg_" + 8 chars
  const hash = new Bun.CryptoHasher("sha256").update(key).digest("hex");

  const row = db.prepare(
    `SELECT ak.id, ak.user_id, ak.scopes, ak.rate_limit, u.is_admin
     FROM api_keys ak JOIN users u ON ak.user_id = u.id
     WHERE ak.key_prefix = ? AND ak.key_hash = ? AND ak.is_active = 1`
  ).get(prefix, hash) as any;

  if (!row) return null;

  // Update last_used_at (non-blocking)
  db.prepare("UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?").run(row.id);

  // Rate limiting
  const now = Date.now();
  let rl = rateLimitMap.get(row.id);
  if (!rl || now > rl.reset) {
    rl = { count: 0, reset: now + RATE_WINDOW_MS };
    rateLimitMap.set(row.id, rl);
  }
  rl.count++;
  if (rl.count > row.rate_limit) return null; // rate limited

  // Determine space — only filter by space if explicitly requested
  let space_id: number | null = null;
  const spaceHeader = req.headers.get("X-Space") || req.headers.get("X-Engram-Space");
  if (spaceHeader) {
    const space = db.prepare("SELECT id FROM spaces WHERE user_id = ? AND name = ?").get(row.user_id, spaceHeader) as any;
    if (space) space_id = space.id;
  }

  return {
    user_id: row.user_id,
    space_id,
    key_id: row.id,
    scopes: (row.scopes || "read,write").split(","),
    is_admin: !!row.is_admin,
  };
}

function getAuthOrDefault(req: Request): AuthContext {
  const auth = authenticate(req);
  if (auth) return auth;
  // Default to owner user (backwards compat for unauthenticated access)
  return { user_id: 1, space_id: null, key_id: null, scopes: ["read", "write", "admin"], is_admin: true };
}

function hasScope(auth: AuthContext, scope: string): boolean {
  return auth.scopes.includes("all") || auth.scopes.includes(scope) || auth.scopes.includes("admin");
}

function generateApiKey(): { key: string; prefix: string; hash: string } {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  const raw = Array.from(bytes).map(b => b.toString(16).padStart(2, "0")).join("");
  const key = `eg_${raw}`;
  const prefix = key.slice(0, 11);
  const hash = new Bun.CryptoHasher("sha256").update(key).digest("hex");
  return { key, prefix, hash };
}

// ============================================================================
// WEB GUI AUTH
// ============================================================================

const GUI_PASSWORD = process.env.ENGRAM_GUI_PASSWORD || process.env.MEGAMIND_GUI_PASSWORD /* legacy fallback */ || "changeme";
const GUI_HMAC_SECRET = process.env.ENGRAM_HMAC_SECRET || crypto.randomUUID() + crypto.randomUUID();
const GUI_COOKIE_MAX_AGE = 7 * 24 * 60 * 60;

function guiSignCookie(ts: number): string {
  const h = new Bun.CryptoHasher("sha256");
  h.update(GUI_HMAC_SECRET + ":" + String(ts));
  return ts + "." + h.digest("hex");
}

function guiVerifyCookie(cookie: string): boolean {
  const dot = cookie.indexOf(".");
  if (dot < 1) return false;
  const ts = cookie.substring(0, dot), sig = cookie.substring(dot + 1);
  const t = parseInt(ts);
  if (isNaN(t) || Date.now() / 1000 - t > GUI_COOKIE_MAX_AGE) return false;
  const h = new Bun.CryptoHasher("sha256");
  h.update(GUI_HMAC_SECRET + ":" + ts);
  return h.digest("hex") === sig;
}

function guiAuthed(req: Request): boolean {
  const ck = (req.headers.get("cookie") || "")
    .split(";").map(c => c.trim())
    .find(c => c.startsWith("engram_auth="));
  if (!ck) return false;
  return guiVerifyCookie(ck.split("=").slice(1).join("="));
}

// ============================================================================
// WEB GUI HTML
// ============================================================================

const GUI_HTML = await Bun.file(import.meta.dir + "/engram-gui.html").text();
const LOGIN_HTML = await Bun.file(import.meta.dir + "/engram-login.html").text();

// ============================================================================
// DIGEST HELPERS
// ============================================================================

async function buildDigestPayload(digest: any, userId: number): Promise<any> {
  const now = new Date();
  const sinceStr = digest.last_sent_at || new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString().replace("T", " ").replace("Z", "");

  const payload: any = {
    type: "engram_digest",
    schedule: digest.schedule,
    generated_at: now.toISOString(),
    period_since: sinceStr,
  };

  if (digest.include_stats) {
    const total = (db.prepare("SELECT COUNT(*) as c FROM memories WHERE is_forgotten = 0 AND user_id = ?").get(userId) as any).c;
    const newCount = (db.prepare("SELECT COUNT(*) as c FROM memories WHERE created_at > ? AND user_id = ?").get(sinceStr, userId) as any).c;
    const archivedCount = (db.prepare("SELECT COUNT(*) as c FROM memories WHERE is_archived = 1 AND updated_at > ? AND user_id = ?").get(sinceStr, userId) as any).c;
    const updatedCount = (db.prepare("SELECT COUNT(*) as c FROM memories WHERE updated_at > ? AND created_at <= ? AND user_id = ?").get(sinceStr, sinceStr, userId) as any).c;

    payload.stats = { total_memories: total, new: newCount, updated: updatedCount, archived: archivedCount };
  }

  if (digest.include_new_memories) {
    const newMems = db.prepare(
      `SELECT id, content, category, importance, tags, created_at
       FROM memories WHERE created_at > ? AND is_forgotten = 0 AND user_id = ?
       ORDER BY importance DESC, created_at DESC LIMIT 20`
    ).all(sinceStr, userId) as any[];

    payload.new_memories = newMems.map(m => ({
      id: m.id, content: m.content.substring(0, 300), category: m.category,
      importance: m.importance, tags: m.tags ? JSON.parse(m.tags) : [],
    }));
  }

  if (digest.include_contradictions) {
    const contras = db.prepare(
      `SELECT ml.source_id, ml.target_id,
         ms.content as a_content, mt.content as b_content
       FROM memory_links ml
       JOIN memories ms ON ml.source_id = ms.id
       JOIN memories mt ON ml.target_id = mt.id
       WHERE ml.type = 'contradicts' AND ml.created_at > ? AND ms.is_forgotten = 0 AND mt.is_forgotten = 0
       ORDER BY ml.created_at DESC LIMIT 10`
    ).all(sinceStr) as any[];

    payload.contradictions = contras.map(c => ({
      memory_a: { id: c.source_id, content: c.a_content.substring(0, 200) },
      memory_b: { id: c.target_id, content: c.b_content.substring(0, 200) },
    }));
  }

  if (digest.include_reflections) {
    const refl = db.prepare(
      `SELECT content, themes, period_start, period_end, created_at FROM reflections
       WHERE user_id = ? AND created_at > ? ORDER BY created_at DESC LIMIT 1`
    ).get(userId, sinceStr) as any;

    if (refl) {
      payload.reflection = {
        content: refl.content,
        themes: refl.themes ? JSON.parse(refl.themes) : [],
        period: { start: refl.period_start, end: refl.period_end },
      };
    }
  }

  return payload;
}

async function sendDigestWebhook(digest: any, payload: any): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (digest.webhook_secret) {
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey("raw", encoder.encode(digest.webhook_secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    const sig = await crypto.subtle.sign("HMAC", key, encoder.encode(JSON.stringify(payload)));
    headers["X-Engram-Signature"] = Array.from(new Uint8Array(sig)).map(b => b.toString(16).padStart(2, "0")).join("");
  }

  try {
    const resp = await fetch(digest.webhook_url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });

    if (resp.ok) {
      db.prepare("UPDATE digests SET last_sent_at = datetime('now'), next_send_at = ? WHERE id = ?").run(
        calculateNextSend(digest.schedule), digest.id
      );
    } else {
      db.prepare("UPDATE digests SET next_send_at = ? WHERE id = ?").run(
        calculateNextSend(digest.schedule), digest.id
      );
      console.error(`Digest #${digest.id} webhook returned ${resp.status}`);
    }
  } catch (e: any) {
    db.prepare("UPDATE digests SET next_send_at = ? WHERE id = ?").run(
      calculateNextSend(digest.schedule), digest.id
    );
    console.error(`Digest #${digest.id} webhook failed: ${e.message}`);
  }
}

function calculateNextSend(schedule: string): string {
  const now = new Date();
  let next: Date;
  if (schedule === "hourly") next = new Date(now.getTime() + 60 * 60 * 1000);
  else if (schedule === "weekly") next = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
  else next = new Date(now.getTime() + 24 * 60 * 60 * 1000);
  return next.toISOString().replace("T", " ").replace("Z", "");
}

async function processScheduledDigests(): Promise<number> {
  const nowStr = new Date().toISOString().replace("T", " ").replace("Z", "");
  const due = db.prepare(
    `SELECT * FROM digests WHERE active = 1 AND next_send_at <= ? ORDER BY next_send_at ASC LIMIT 10`
  ).all(nowStr) as any[];

  let sent = 0;
  for (const digest of due) {
    try {
      const payload = await buildDigestPayload(digest, digest.user_id);
      await sendDigestWebhook(digest, payload);
      sent++;
    } catch (e: any) {
      console.error(`Digest #${digest.id} processing failed: ${e.message}`);
    }
  }
  return sent;
}

// ============================================================================
// SERVER
// ============================================================================

await initEmbedder();

const server = Bun.serve({
  port: PORT,
  hostname: HOST,
  async fetch(req) {
    const url = new URL(req.url);
    const method = req.method;

    // CORS preflight
    if (method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
      });
    }

    // ========================================================================
    // WEB GUI AUTH
    // ========================================================================
    if (url.pathname === "/gui/auth" && method === "POST") {
      try {
        const body = await req.json() as { password?: string };
        if (body.password === GUI_PASSWORD) {
          const cookie = guiSignCookie(Math.floor(Date.now() / 1000));
          return new Response(JSON.stringify({ ok: true }), {
            headers: {
              "Content-Type": "application/json",
              "Set-Cookie": `engram_auth=${cookie}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${GUI_COOKIE_MAX_AGE}`
            }
          });
        }
        return json({ error: "Invalid password" }, 401);
      } catch { return json({ error: "Bad request" }, 400); }
    }

    if (url.pathname === "/gui/logout" && method === "GET") {
      return new Response(LOGIN_HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8", "Set-Cookie": "engram_auth=; Path=/; HttpOnly; Max-Age=0" }
      });
    }

    // ========================================================================
    // WEB GUI
    // ========================================================================
    if ((url.pathname === "/" || url.pathname === "/gui") && method === "GET") {
      if (guiAuthed(req)) {
        return new Response(GUI_HTML, { headers: { "Content-Type": "text/html; charset=utf-8" } });
      }
      return new Response(LOGIN_HTML, { headers: { "Content-Type": "text/html; charset=utf-8" } });
    }

    // ========================================================================
    // AUTH CONTEXT — extract user from API key or default to owner
    // ========================================================================
    const auth = getAuthOrDefault(req);

    // ========================================================================
    // USER MANAGEMENT (admin only)
    // ========================================================================

    if (url.pathname === "/users" && method === "POST") {
      if (!auth.is_admin) return errorResponse("Admin required", 403);
      try {
        const body = await req.json() as any;
        if (!body.username) return errorResponse("username is required");
        const result = db.prepare(
          "INSERT INTO users (username, email) VALUES (?, ?) RETURNING id, created_at"
        ).get(body.username.trim(), body.email || null) as any;
        // Create default space for new user
        db.prepare("INSERT INTO spaces (user_id, name, description) VALUES (?, 'default', 'Default memory space')").run(result.id);
        return json({ id: result.id, username: body.username.trim(), created_at: result.created_at });
      } catch (e: any) {
        if (e.message?.includes("UNIQUE")) return errorResponse("Username already exists", 409);
        return errorResponse(`Failed: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/users" && method === "GET") {
      if (!auth.is_admin) return errorResponse("Admin required", 403);
      const users = db.prepare(
        `SELECT u.id, u.username, u.email, u.is_admin, u.created_at,
           (SELECT COUNT(*) FROM memories WHERE user_id = u.id) as memory_count,
           (SELECT COUNT(*) FROM api_keys WHERE user_id = u.id AND is_active = 1) as key_count
         FROM users u ORDER BY u.id`
      ).all();
      return json({ users });
    }

    // ========================================================================
    // API KEY MANAGEMENT
    // ========================================================================

    if (url.pathname === "/keys" && method === "POST") {
      if (!hasScope(auth, "admin")) return errorResponse("Admin scope required", 403);
      try {
        const body = await req.json() as any;
        const targetUserId = body.user_id || auth.user_id;
        // Only admin can create keys for other users
        if (targetUserId !== auth.user_id && !auth.is_admin) return errorResponse("Cannot create keys for other users", 403);
        const { key, prefix, hash } = generateApiKey();
        const name = body.name || "default";
        const scopes = body.scopes || "read,write";
        const rateLimit = Math.min(Math.max(Number(body.rate_limit) || DEFAULT_RATE_LIMIT, 10), 10000);
        db.prepare(
          "INSERT INTO api_keys (user_id, key_prefix, key_hash, name, scopes, rate_limit) VALUES (?, ?, ?, ?, ?, ?)"
        ).run(targetUserId, prefix, hash, name, scopes, rateLimit);
        return json({ key, name, scopes, rate_limit: rateLimit, user_id: targetUserId, message: "Save this key — it cannot be retrieved again." });
      } catch (e: any) {
        return errorResponse(`Failed: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/keys" && method === "GET") {
      const keys = db.prepare(
        `SELECT id, key_prefix, name, scopes, rate_limit, is_active, last_used_at, created_at
         FROM api_keys WHERE user_id = ? ORDER BY created_at DESC`
      ).all(auth.user_id);
      return json({ keys });
    }

    if (url.pathname.match(/^\/keys\/\d+$/) && method === "DELETE") {
      const id = Number(url.pathname.split("/")[2]);
      const key = db.prepare("SELECT user_id FROM api_keys WHERE id = ?").get(id) as any;
      if (!key) return errorResponse("Not found", 404);
      if (key.user_id !== auth.user_id && !auth.is_admin) return errorResponse("Forbidden", 403);
      db.prepare("UPDATE api_keys SET is_active = 0 WHERE id = ?").run(id);
      return json({ revoked: true, id });
    }

    // ========================================================================
    // SPACE MANAGEMENT
    // ========================================================================

    if (url.pathname === "/spaces" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        if (!body.name) return errorResponse("name is required");
        const result = db.prepare(
          "INSERT INTO spaces (user_id, name, description) VALUES (?, ?, ?) RETURNING id, created_at"
        ).get(auth.user_id, body.name.trim(), body.description || null) as any;
        return json({ id: result.id, name: body.name.trim(), created_at: result.created_at });
      } catch (e: any) {
        if (e.message?.includes("UNIQUE")) return errorResponse("Space name already exists", 409);
        return errorResponse(`Failed: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/spaces" && method === "GET") {
      const spaces = db.prepare(
        `SELECT s.id, s.name, s.description, s.created_at,
           (SELECT COUNT(*) FROM memories WHERE space_id = s.id) as memory_count
         FROM spaces s WHERE s.user_id = ? ORDER BY s.name`
      ).all(auth.user_id);
      return json({ spaces });
    }

    if (url.pathname.match(/^\/spaces\/\d+$/) && method === "DELETE") {
      const id = Number(url.pathname.split("/")[2]);
      const space = db.prepare("SELECT user_id, name FROM spaces WHERE id = ?").get(id) as any;
      if (!space) return errorResponse("Not found", 404);
      if (space.user_id !== auth.user_id && !auth.is_admin) return errorResponse("Forbidden", 403);
      if (space.name === "default") return errorResponse("Cannot delete default space", 400);
      db.prepare("DELETE FROM spaces WHERE id = ?").run(id);
      return json({ deleted: true, id });
    }

    // ========================================================================
    // EXPORT — full dump of user's memories
    // ========================================================================

    if (url.pathname === "/export" && method === "GET") {
      if (!hasScope(auth, "read")) return errorResponse("Read scope required", 403);
      const format = url.searchParams.get("format") || "json";
      const spaceFilter = auth.space_id ? "AND space_id = ?" : "";
      const params: any[] = [auth.user_id];
      if (auth.space_id) params.push(auth.space_id);

      const mems = db.prepare(
        `SELECT id, content, category, source, importance, version, is_latest,
           parent_memory_id, root_memory_id, source_count, is_static, is_forgotten,
           is_archived, forget_after, forget_reason, is_inference, created_at, updated_at
         FROM memories WHERE user_id = ? ${spaceFilter} ORDER BY id`
      ).all(...params);

      const memLinks = db.prepare(
        `SELECT ml.source_id, ml.target_id, ml.similarity, ml.type
         FROM memory_links ml
         JOIN memories m ON ml.source_id = m.id
         WHERE m.user_id = ?`
      ).all(auth.user_id);

      const exportData = {
        version: "engram-v4",
        exported_at: new Date().toISOString(),
        memories: mems,
        links: memLinks,
        stats: { memory_count: mems.length, link_count: memLinks.length },
      };

      if (format === "jsonl") {
        const lines = mems.map(m => JSON.stringify(m)).join("\n");
        return new Response(lines, {
          headers: {
            "Content-Type": "application/x-ndjson",
            "Content-Disposition": "attachment; filename=engram-export.jsonl",
            "Access-Control-Allow-Origin": "*",
          },
        });
      }

      return new Response(JSON.stringify(exportData, null, 2), {
        headers: {
          "Content-Type": "application/json",
          "Content-Disposition": "attachment; filename=engram-export.json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }

    // ========================================================================
    // IMPORT — bulk import memories
    // ========================================================================

    if (url.pathname === "/import" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        const items = body.memories || body.items || body;
        if (!Array.isArray(items)) return errorResponse("Expected memories array");

        let imported = 0, failed = 0;
        const importTransaction = db.transaction(() => {
          for (const item of items) {
            try {
              if (!item.content || typeof item.content !== "string") { failed++; continue; }
              db.prepare(
                `INSERT INTO memories (content, category, source, importance, user_id, space_id,
                   is_static, version, source_count, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, datetime('now')), COALESCE(?, datetime('now')))`
              ).run(
                item.content.trim(),
                item.category || "general",
                item.source || "import",
                Math.max(1, Math.min(10, Number(item.importance) || 5)),
                auth.user_id,
                auth.space_id || null,
                item.is_static ? 1 : 0,
                item.version || 1,
                item.source_count || 1,
                item.created_at || null,
                item.updated_at || null,
              );
              imported++;
            } catch { failed++; }
          }
        });
        importTransaction();

        // Queue embedding backfill for imported memories
        if (imported > 0) {
          backfillEmbeddings(imported).catch(e => console.error("Import backfill error:", e));
        }

        return json({ imported, failed, total: items.length });
      } catch (e: any) {
        return errorResponse(`Import failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // HEALTH
    // ========================================================================
    if (url.pathname === "/health" && method === "GET") {
      const memCount = db.prepare("SELECT COUNT(*) as count FROM memories").get() as { count: number };
      const convCount = db.prepare("SELECT COUNT(*) as count FROM conversations").get() as { count: number };
      const msgCount = db.prepare("SELECT COUNT(*) as count FROM messages").get() as { count: number };
      const linkCount = db.prepare("SELECT COUNT(*) as count FROM memory_links").get() as { count: number };
      const embCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE embedding IS NOT NULL").get() as { count: number };
      const noEmbCount = (countNoEmbedding.get() as { count: number }).count;
      const forgottenCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE is_forgotten = 1").get() as { count: number };
      const staticCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE is_static = 1 AND is_forgotten = 0").get() as { count: number };
      const versionedCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE version > 1").get() as { count: number };
      const archivedCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE is_archived = 1 AND is_forgotten = 0").get() as { count: number };
      const pendingCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE status = 'pending' AND is_forgotten = 0").get() as { count: number };
      const rejectedCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE status = 'rejected'").get() as { count: number };
      const episodeCount = db.prepare("SELECT COUNT(*) as count FROM episodes").get() as { count: number };
      const consolidationCount = db.prepare("SELECT COUNT(*) as count FROM consolidations").get() as { count: number };
      const taggedCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE tags IS NOT NULL AND is_forgotten = 0").get() as { count: number };
      const entityCount = db.prepare("SELECT COUNT(*) as count FROM entities").get() as { count: number };
      const projectCount = db.prepare("SELECT COUNT(*) as count FROM projects").get() as { count: number };
      const dbSize = Bun.file(DB_PATH).size;
      return json({
        status: "ok",
        version: 5.0,
        memories: memCount.count,
        embedded: embCount.count,
        unembedded: noEmbCount,
        links: linkCount.count,
        forgotten: forgottenCount.count,
        archived: archivedCount.count,
          pending: pendingCount.count,
          rejected: rejectedCount.count,
        static: staticCount.count,
        versioned: versionedCount.count,
        tagged: taggedCount.count,
        episodes: episodeCount.count,
        consolidations: consolidationCount.count,
        entities: entityCount.count,
        projects: projectCount.count,
        conversations: convCount.count,
        messages: msgCount.count,
        embedding_model: EMBEDDING_MODEL,
        llm_model: LLM_MODEL,
        llm_configured: !!LLM_API_KEY,
        features: {
          decay: "fsrs6",
          fsrs6: true,
          dual_strength: true,
          tags: true,
          episodes: true,
          consolidation: !!LLM_API_KEY,
          typed_relationships: true,
          access_tracking: true,
          confidence: true,
          webhooks: true,
          sync: true,
          pack: true,
          prompt_templates: true,
          auto_tagging: !!LLM_API_KEY,
          mem0_import: true,
          supermemory_import: true,
          entities: true,
          projects: true,
          scoped_search: true,
          reranker: RERANKER_ENABLED && !!LLM_API_KEY,
          conversation_extraction: !!LLM_API_KEY,
          derived_memories: !!LLM_API_KEY,
          graph: true,
          url_ingest: true,
          contradiction_detection: true,
          contradiction_resolution: !!LLM_API_KEY,
          time_travel: true,
          smart_context: true,
          reflections: !!LLM_API_KEY,
          scheduled_digests: true,
        },
        db_size_mb: Math.round(dbSize / 1048576 * 100) / 100,
      });
    }

    // ========================================================================
    // CONVERSATION EXTRACTION — POST /add (Mem0-compatible)
    // ========================================================================

    if (url.pathname === "/add" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      if (!LLM_API_KEY) return errorResponse("LLM not configured — /add requires fact extraction", 400);
      try {
        const body = await req.json() as any;
        const messages = body.messages as Array<{ role: string; content: string }>;
        if (!Array.isArray(messages) || messages.length === 0) {
          return errorResponse("messages array required: [{role: 'user'|'assistant'|'system', content: '...'}]");
        }

        const category = body.category || "general";
        const source = body.source || "conversation";
        const projectIds = body.project_ids as number[] | undefined;
        const entityIds = body.entity_ids as number[] | undefined;
        const episodeId = body.episode_id as number | undefined;

        // Format conversation for extraction
        const convoText = messages.map(m => `${m.role}: ${m.content}`).join("\n\n");

        const extractionPrompt = `You are a fact extraction engine. Analyze this conversation and extract distinct, atomic facts worth remembering long-term.

Rules:
- Extract facts primarily from USER messages. Only extract from ASSISTANT messages if they contain genuinely novel information (not just rephrasing the user).
- Each fact should be ONE self-contained statement. If you can't summarize it in under 50 words, split it.
- Skip greetings, filler, questions without assertions, and transient information.
- For each fact, classify:
  - category: task|discovery|decision|state|issue|general
  - importance: 1-10 (9-10=critical decisions, 7-8=useful knowledge, 5-6=context, <5=minor)
  - is_static: true if this is a permanent/rarely-changing fact, false if temporal
  - tags: 2-5 lowercase keyword tags
- Detect temporal facts and set forget_after if appropriate (ISO datetime or null)

Return JSON:
{
  "facts": [
    {
      "content": "extracted fact as a clear statement",
      "category": "task",
      "importance": 7,
      "is_static": false,
      "tags": ["keyword1", "keyword2"],
      "forget_after": null
    }
  ]
}

If no meaningful facts, return {"facts": []}`;

        const llmResp = await callLLM(extractionPrompt, convoText);
        if (!llmResp) return json({ added: 0, facts: [] });

        // Parse extracted facts
        let extracted: { facts: Array<any> };
        try {
          const cleaned = llmResp.replace(/```json\n?|\n?```/g, "").trim();
          // Try direct parse first, then extract JSON object
          try {
            extracted = JSON.parse(cleaned);
          } catch {
            const jsonMatch = cleaned.match(/\{[\s\S]*"facts"[\s\S]*\}/);
            if (jsonMatch) {
              extracted = JSON.parse(jsonMatch[0]);
            } else {
              console.error("Conversation extraction: unparseable LLM response:", cleaned.substring(0, 500));
              return errorResponse("LLM returned unparseable response", 500);
            }
          }
        } catch (parseErr: any) {
          console.error("Conversation extraction parse error:", parseErr.message);
          return errorResponse("LLM returned unparseable response", 500);
        }

        if (!extracted.facts?.length) return json({ added: 0, facts: [] });

        // Store each fact as a memory
        const stored: Array<{ id: number; content: string; category: string }> = [];
        for (const fact of extracted.facts) {
          if (!fact.content?.trim()) continue;

          let embBuffer: Buffer | null = null;
          let embArray: Float32Array | null = null;
          try {
            embArray = await embed(fact.content.trim());
            embBuffer = embeddingToBuffer(embArray);
          } catch {}

          const result = insertMemory.get(
            fact.content.trim(), fact.category || category, source, null,
            fact.importance || DEFAULT_IMPORTANCE, embBuffer,
            1, 1, null, null, 1, fact.is_static ? 1 : 0, 0,
            fact.forget_after || null, null, 0
          ) as { id: number; created_at: string };

          const syncId = crypto.randomUUID();
          const tagsJson = fact.tags?.length ? JSON.stringify(fact.tags) : null;
          db.prepare(
            "UPDATE memories SET user_id = ?, space_id = ?, tags = ?, episode_id = ?, sync_id = ?, confidence = 1.0 WHERE id = ?"
          ).run(auth.user_id, auth.space_id || null, tagsJson, episodeId || null, syncId, result.id);

          // Link to entities/projects
          if (entityIds) for (const eid of entityIds) linkMemoryEntity.run(result.id, eid);
          if (projectIds) for (const pid of projectIds) linkMemoryProject.run(result.id, pid);

          // Auto-link
          if (embArray) { writeVec(result.id, embArray); await autoLink(result.id, embArray); }

          // Queue async fact extraction (for relationship detection)
          if (LLM_API_KEY) {
            (async () => {
              try {
                const extraction = await extractFacts(result.id, fact.content.trim(), embArray!);
                if (extraction) processExtractionResult(result.id, extraction, embArray!);
              } catch {}
            })();
          }

          emitWebhookEvent("memory.created", {
            id: result.id, content: fact.content.trim(), category: fact.category || category,
            importance: fact.importance || DEFAULT_IMPORTANCE, source: "conversation",
          }, auth.user_id);

          stored.push({ id: result.id, content: fact.content.trim(), category: fact.category || category });
        }

        return json({
          added: stored.length,
          facts: stored,
          source: "conversation",
          messages_processed: messages.length,
        });
      } catch (e: any) {
        return errorResponse(`Conversation extraction failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // INGEST — Extract memories from URLs or text blobs
    // ========================================================================

    if (url.pathname === "/ingest" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      if (!LLM_API_KEY) return errorResponse("LLM not configured — /ingest requires fact extraction", 400);
      try {
        const body = await req.json() as any;
        const { url: ingestUrl, text: ingestText, entity_ids, project_ids, episode_id, source } = body;

        if (!ingestUrl && !ingestText) {
          return errorResponse("Provide 'url' (string) or 'text' (string)");
        }

        let rawText = "";
        let ingestSource = source || "ingest";
        let title = "";

        // --- Fetch URL ---
        if (ingestUrl) {
          if (typeof ingestUrl !== "string" || !ingestUrl.match(/^https?:\/\//)) {
            return errorResponse("url must be a valid http/https URL");
          }
          try {
            const resp = await fetch(ingestUrl, {
              headers: { "User-Agent": "Engram/4.4 (memory ingest)" },
              redirect: "follow",
              signal: AbortSignal.timeout(15000),
            });
            if (!resp.ok) return errorResponse(`Fetch failed: ${resp.status} ${resp.statusText}`, 502);

            const contentType = resp.headers.get("content-type") || "";
            const raw = await resp.text();

            if (contentType.includes("html")) {
              // Extract title
              const titleMatch = raw.match(/<title[^>]*>([^<]+)<\/title>/i);
              title = titleMatch ? titleMatch[1].trim() : new URL(ingestUrl).hostname;

              // Strip HTML to text
              rawText = raw
                .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
                .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
                .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, "")
                .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, "")
                .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, "")
                .replace(/<aside[^>]*>[\s\S]*?<\/aside>/gi, "")
                .replace(/<!--[\s\S]*?-->/g, "")
                .replace(/<br\s*\/?>/gi, "\n")
                .replace(/<\/p>/gi, "\n\n")
                .replace(/<\/div>/gi, "\n")
                .replace(/<\/li>/gi, "\n")
                .replace(/<\/h[1-6]>/gi, "\n\n")
                .replace(/<[^>]+>/g, " ")
                .replace(/&nbsp;/g, " ")
                .replace(/&amp;/g, "&")
                .replace(/&lt;/g, "<")
                .replace(/&gt;/g, ">")
                .replace(/&quot;/g, '"')
                .replace(/&#39;/g, "'")
                .replace(/\n{3,}/g, "\n\n")
                .replace(/ {2,}/g, " ")
                .trim();
            } else {
              // Plain text, JSON, etc — use as-is
              rawText = raw.trim();
              title = new URL(ingestUrl).pathname.split("/").pop() || ingestUrl;
            }
            ingestSource = `url:${ingestUrl}`;
          } catch (fetchErr: any) {
            return errorResponse(`Fetch error: ${fetchErr.message}`, 502);
          }
        }

        // --- Raw text ---
        if (ingestText) {
          if (typeof ingestText !== "string" || ingestText.trim().length === 0) {
            return errorResponse("text must be a non-empty string");
          }
          rawText = ingestText.trim();
          title = body.title || rawText.substring(0, 60).replace(/\n/g, " ");
          ingestSource = source || "text";
        }

        // Truncate to ~12K chars for LLM context
        const MAX_INGEST = 12000;
        const truncated = rawText.length > MAX_INGEST;
        if (truncated) rawText = rawText.substring(0, MAX_INGEST);

        // --- Chunk into segments for extraction ---
        const CHUNK_SIZE = 3000;
        const CHUNK_OVERLAP = 200;
        const chunks: string[] = [];
        if (rawText.length <= CHUNK_SIZE) {
          chunks.push(rawText);
        } else {
          let pos = 0;
          while (pos < rawText.length) {
            let end = Math.min(pos + CHUNK_SIZE, rawText.length);
            // Try to break at paragraph or sentence boundary
            if (end < rawText.length) {
              const paraBreak = rawText.lastIndexOf("\n\n", end);
              if (paraBreak > pos + CHUNK_SIZE * 0.5) end = paraBreak;
              else {
                const sentBreak = rawText.lastIndexOf(". ", end);
                if (sentBreak > pos + CHUNK_SIZE * 0.5) end = sentBreak + 1;
              }
            }
            chunks.push(rawText.substring(pos, end));
            pos = end > pos ? end - CHUNK_OVERLAP : end + 1;
            if (pos >= rawText.length) break;
          }
        }

        // --- Extract facts from each chunk ---
        const allFacts: Array<{ id: number; content: string; category: string }> = [];
        let chunkNum = 0;

        for (const chunk of chunks) {
          chunkNum++;
          const extractionPrompt = `You are a fact extraction engine. Analyze this text and extract distinct, atomic facts worth remembering long-term.

Source: ${title}${ingestUrl ? ` (${ingestUrl})` : ""}
Chunk ${chunkNum}/${chunks.length}${truncated ? " (document was truncated)" : ""}

Rules:
- Each fact should be ONE self-contained statement. Under 50 words each.
- Skip boilerplate, navigation text, ads, cookie notices, and filler.
- Preserve specific numbers, names, dates, and technical details.
- For each fact, classify:
  - category: task|discovery|decision|state|issue|general
  - importance: 1-10
  - is_static: true if permanent/rarely-changing, false if temporal
  - tags: 2-5 lowercase keyword tags
- If the text has no meaningful facts, return {"facts": []}

Return JSON:
{
  "facts": [
    {
      "content": "extracted fact as a clear statement",
      "category": "discovery",
      "importance": 7,
      "is_static": true,
      "tags": ["keyword1", "keyword2"]
    }
  ]
}`;

          const llmResp = await callLLM(extractionPrompt, chunk);
          if (!llmResp) continue;

          let extracted: { facts: Array<any> };
          try {
            const cleaned = llmResp.replace(/```json\n?|\n?```/g, "").trim();
            try {
              extracted = JSON.parse(cleaned);
            } catch {
              const jsonMatch = cleaned.match(/\{[\s\S]*"facts"[\s\S]*\}/);
              if (jsonMatch) extracted = JSON.parse(jsonMatch[0]);
              else continue;
            }
          } catch { continue; }

          if (!extracted.facts?.length) continue;

          for (const fact of extracted.facts) {
            if (!fact.content?.trim()) continue;

            let embBuffer: Buffer | null = null;
            let embArray: Float32Array | null = null;
            try {
              embArray = await embed(fact.content.trim());
              embBuffer = embeddingToBuffer(embArray);
            } catch {}

            const result = insertMemory.get(
              fact.content.trim(), fact.category || "general", ingestSource, null,
              fact.importance || DEFAULT_IMPORTANCE, embBuffer,
              1, 1, null, null, 1, fact.is_static ? 1 : 0, 0,
              null, null, 0
            ) as { id: number; created_at: string };

            const syncId = crypto.randomUUID();
            const tags = fact.tags?.length ? JSON.stringify(fact.tags) : null;
            db.prepare(
              "UPDATE memories SET user_id = ?, space_id = ?, tags = ?, episode_id = ?, sync_id = ?, confidence = 1.0 WHERE id = ?"
            ).run(auth.user_id, auth.space_id || null, tags, episode_id || null, syncId, result.id);

            if (entity_ids) for (const eid of entity_ids) linkMemoryEntity.run(result.id, eid);
            if (project_ids) for (const pid of project_ids) linkMemoryProject.run(result.id, pid);

            if (embArray) { writeVec(result.id, embArray); await autoLink(result.id, embArray); }

            if (LLM_API_KEY) {
              (async () => {
                try {
                  const extraction = await extractFacts(result.id, fact.content.trim(), embArray!);
                  if (extraction) processExtractionResult(result.id, extraction, embArray!);
                } catch {}
              })();
            }

            emitWebhookEvent("memory.created", {
              id: result.id, content: fact.content.trim(), category: fact.category || "general",
              importance: fact.importance || DEFAULT_IMPORTANCE, source: ingestSource,
            }, auth.user_id);

            allFacts.push({ id: result.id, content: fact.content.trim(), category: fact.category || "general" });
          }
        }

        return json({
          ingested: allFacts.length,
          facts: allFacts,
          source: ingestSource,
          title,
          chunks_processed: chunks.length,
          truncated,
        });
      } catch (e: any) {
        return errorResponse(`Ingest failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // CONTRADICTION DETECTION — find conflicting memories
    // ========================================================================

    if (url.pathname === "/contradictions" && method === "GET") {
      try {
        const threshold = Number(url.searchParams.get("threshold") || 0.6);
        const limitParam = Math.min(Number(url.searchParams.get("limit") || 30), 100);
        const useLLM = url.searchParams.get("verify") === "true" && !!LLM_API_KEY;

        // Get all contradicts-type links first (already detected by fact extraction)
        const knownContradictions = db.prepare(
          `SELECT ml.source_id, ml.target_id, ml.similarity,
             ms.content as source_content, ms.category as source_category, ms.created_at as source_created,
             mt.content as target_content, mt.category as target_category, mt.created_at as target_created
           FROM memory_links ml
           JOIN memories ms ON ml.source_id = ms.id
           JOIN memories mt ON ml.target_id = mt.id
           WHERE ml.type = 'contradicts' AND ms.is_forgotten = 0 AND mt.is_forgotten = 0
           ORDER BY ml.created_at DESC LIMIT ?`
        ).all(limitParam) as any[];

        const contradictions: Array<{
          memory_a: { id: number; content: string; category: string; created_at: string };
          memory_b: { id: number; content: string; category: string; created_at: string };
          similarity: number;
          source: string;
          verified?: boolean;
          explanation?: string;
        }> = [];

        // Add known contradictions
        for (const c of knownContradictions) {
          contradictions.push({
            memory_a: { id: c.source_id, content: c.source_content, category: c.source_category, created_at: c.source_created },
            memory_b: { id: c.target_id, content: c.target_content, category: c.target_category, created_at: c.target_created },
            similarity: c.similarity,
            source: "link",
          });
        }

        // Scan for potential contradictions: high similarity but different content patterns
        // Same category memories with high embedding similarity often contain updates/contradictions
        if (contradictions.length < limitParam) {
          const allMems = getLatestEmbeddings.all() as Array<{
            id: number; content: string; category: string; importance: number;
            embedding: Buffer; is_static: boolean; source_count: number;
          }>;

          const seenPairs = new Set(contradictions.map(c =>
            `${Math.min(c.memory_a.id, c.memory_b.id)}-${Math.max(c.memory_a.id, c.memory_b.id)}`
          ));

          const candidates: Array<{ a: any; b: any; sim: number }> = [];

          for (let i = 0; i < allMems.length && candidates.length < limitParam * 3; i++) {
            const embA = bufferToEmbedding(allMems[i].embedding);
            for (let j = i + 1; j < allMems.length; j++) {
              const pairKey = `${Math.min(allMems[i].id, allMems[j].id)}-${Math.max(allMems[i].id, allMems[j].id)}`;
              if (seenPairs.has(pairKey)) continue;

              // Same or related category + high similarity = potential contradiction
              if (allMems[i].category !== allMems[j].category && threshold < 0.8) continue;

              const embB = bufferToEmbedding(allMems[j].embedding);
              const sim = cosineSimilarity(embA, embB);

              // Sweet spot: similar enough to be about the same thing (>0.6) but not identical (>0.95)
              if (sim >= threshold && sim < 0.95) {
                candidates.push({ a: allMems[i], b: allMems[j], sim });
                seenPairs.add(pairKey);
              }
            }
          }

          candidates.sort((x, y) => y.sim - x.sim);
          const toVerify = candidates.slice(0, limitParam - contradictions.length);

          if (useLLM && toVerify.length > 0) {
            // Batch verify with LLM
            const pairs = toVerify.map((c, i) =>
              `[Pair ${i}]\nA (#${c.a.id}): ${c.a.content.substring(0, 300)}\nB (#${c.b.id}): ${c.b.content.substring(0, 300)}`
            ).join("\n\n");

            const verifyPrompt = `You detect contradictions between memory pairs. For each pair, determine if they CONTRADICT each other (state conflicting facts about the same thing).

NOT contradictions: updates (B supersedes A), extensions (B adds to A), or unrelated.
IS a contradiction: A says X, B says NOT-X or a different value for the same property.

Return JSON array:
[
  { "pair": 0, "contradicts": true/false, "explanation": "brief reason" },
  ...
]

Only include pairs that are actual contradictions.`;

            try {
              const resp = await callLLM(verifyPrompt, pairs);
              const cleaned = resp.replace(/```json\n?|\n?```/g, "").trim();
              const results = JSON.parse(cleaned) as Array<{ pair: number; contradicts: boolean; explanation: string }>;

              for (const r of results) {
                if (r.contradicts && toVerify[r.pair]) {
                  const c = toVerify[r.pair];
                  contradictions.push({
                    memory_a: { id: c.a.id, content: c.a.content, category: c.a.category, created_at: "" },
                    memory_b: { id: c.b.id, content: c.b.content, category: c.b.category, created_at: "" },
                    similarity: Math.round(c.sim * 1000) / 1000,
                    source: "scan",
                    verified: true,
                    explanation: r.explanation,
                  });
                }
              }
            } catch (llmErr: any) {
              // Fall back to returning unverified candidates
              for (const c of toVerify) {
                contradictions.push({
                  memory_a: { id: c.a.id, content: c.a.content, category: c.a.category, created_at: "" },
                  memory_b: { id: c.b.id, content: c.b.content, category: c.b.category, created_at: "" },
                  similarity: Math.round(c.sim * 1000) / 1000,
                  source: "scan",
                  verified: false,
                });
              }
            }
          } else {
            for (const c of toVerify) {
              contradictions.push({
                memory_a: { id: c.a.id, content: c.a.content, category: c.a.category, created_at: "" },
                memory_b: { id: c.b.id, content: c.b.content, category: c.b.category, created_at: "" },
                similarity: Math.round(c.sim * 1000) / 1000,
                source: "scan",
              });
            }
          }
        }

        return json({
          contradictions: contradictions.slice(0, limitParam),
          total: contradictions.length,
          threshold,
          verified: useLLM,
        });
      } catch (e: any) {
        return errorResponse(`Contradiction scan failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // CONTRADICTION RESOLUTION — resolve a specific contradiction
    // ========================================================================

    if (url.pathname === "/contradictions/resolve" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        const { memory_a_id, memory_b_id, resolution } = body;
        // resolution: "keep_a" | "keep_b" | "keep_both" | "merge"

        if (!memory_a_id || !memory_b_id || !resolution) {
          return errorResponse("memory_a_id, memory_b_id, and resolution (keep_a|keep_b|keep_both|merge) required");
        }

        const memA = getMemory.get(memory_a_id) as any;
        const memB = getMemory.get(memory_b_id) as any;
        if (!memA || !memB) return errorResponse("One or both memories not found", 404);

        if (resolution === "keep_a") {
          markArchived.run(memory_b_id);
          insertLink.run(memory_a_id, memory_b_id, 1.0, "resolves");
          // Remove contradicts links
          db.prepare("DELETE FROM memory_links WHERE type = 'contradicts' AND ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))").run(memory_a_id, memory_b_id, memory_b_id, memory_a_id);
          return json({ resolved: true, kept: memory_a_id, archived: memory_b_id });
        }

        if (resolution === "keep_b") {
          markArchived.run(memory_a_id);
          insertLink.run(memory_b_id, memory_a_id, 1.0, "resolves");
          db.prepare("DELETE FROM memory_links WHERE type = 'contradicts' AND ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))").run(memory_a_id, memory_b_id, memory_b_id, memory_a_id);
          return json({ resolved: true, kept: memory_b_id, archived: memory_a_id });
        }

        if (resolution === "keep_both") {
          // Remove the contradiction link, mark as intentional
          db.prepare("DELETE FROM memory_links WHERE type = 'contradicts' AND ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))").run(memory_a_id, memory_b_id, memory_b_id, memory_a_id);
          insertLink.run(memory_a_id, memory_b_id, 0.9, "related");
          return json({ resolved: true, action: "kept_both", linked: true });
        }

        if (resolution === "merge" && LLM_API_KEY) {
          const mergeResp = await callLLM(
            `Merge these two contradicting memories into a single accurate memory. Preserve the most recent/correct information. Return JSON: {"content": "merged text", "category": "category"}`,
            `Memory A (#${memA.id}, created ${memA.created_at}): ${memA.content}\n\nMemory B (#${memB.id}, created ${memB.created_at}): ${memB.content}`
          );
          const cleaned = mergeResp.replace(/```json\n?|\n?```/g, "").trim();
          const merged = JSON.parse(cleaned) as { content: string; category: string };

          let embBuffer: Buffer | null = null;
          let embArray: Float32Array | null = null;
          try { embArray = await embed(merged.content); embBuffer = embeddingToBuffer(embArray); } catch {}

          const result = insertMemory.get(
            merged.content, merged.category || memA.category, "contradiction-merge", null,
            Math.max(memA.importance, memB.importance), embBuffer,
            1, 1, null, null, (memA.source_count || 1) + (memB.source_count || 1), 0, 0, null, null, 0
          ) as { id: number; created_at: string };
          db.prepare("UPDATE memories SET user_id = ? WHERE id = ?").run(auth.user_id, result.id);

          markArchived.run(memory_a_id);
          markArchived.run(memory_b_id);
          insertLink.run(result.id, memory_a_id, 1.0, "resolves");
          insertLink.run(result.id, memory_b_id, 1.0, "resolves");
          db.prepare("DELETE FROM memory_links WHERE type = 'contradicts' AND ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))").run(memory_a_id, memory_b_id, memory_b_id, memory_a_id);

          if (embArray) { writeVec(result.id, embArray); await autoLink(result.id, embArray); }

          return json({ resolved: true, merged_memory_id: result.id, content: merged.content, archived: [memory_a_id, memory_b_id] });
        }

        return errorResponse("Invalid resolution. Use: keep_a, keep_b, keep_both, merge");
      } catch (e: any) {
        return errorResponse(`Resolution failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // TIME-TRAVEL — query memory state at a point in time
    // ========================================================================

    if (url.pathname === "/timetravel" && method === "POST") {
      try {
        const body = await req.json() as any;
        const { as_of, query, category, limit: lim } = body;

        if (!as_of) return errorResponse("as_of (ISO datetime) required");

        const asOfDate = new Date(as_of);
        if (isNaN(asOfDate.getTime())) return errorResponse("as_of must be valid ISO datetime");

        const asOfStr = asOfDate.toISOString().replace("T", " ").replace("Z", "");
        const resultLimit = Math.min(Number(lim) || 50, 200);

        // Get memories that existed at as_of, considering version chains
        // A memory was "current" at time T if:
        // 1. It was created before T
        // 2. It was either still is_latest OR was superseded after T
        let timeMemories: any[];

        if (query) {
          // Semantic search within the time window
          const embQ = await embed(query);
          const allMems = db.prepare(
            `SELECT id, content, category, source, importance, embedding, created_at, updated_at,
               version, is_latest, root_memory_id, parent_memory_id, is_static, is_forgotten, tags,
               is_archived, confidence, decay_score
             FROM memories
             WHERE created_at <= ? AND is_forgotten = 0 AND embedding IS NOT NULL AND user_id = ?
             ORDER BY created_at DESC`
          ).all(asOfStr, auth.user_id) as any[];

          // For version chains, find which version was current at as_of
          const rootLatest = new Map<number, any>(); // root_id -> best version at as_of
          const standalone: any[] = [];

          for (const m of allMems) {
            if (m.root_memory_id) {
              const rootId = m.root_memory_id;
              const existing = rootLatest.get(rootId);
              if (!existing || new Date(m.created_at) > new Date(existing.created_at)) {
                rootLatest.set(rootId, m);
              }
            } else if (!m.parent_memory_id) {
              // Check if this was later superseded — if so, check if superseded after as_of
              const laterVersion = db.prepare(
                `SELECT id, created_at FROM memories WHERE parent_memory_id = ? AND created_at <= ? ORDER BY created_at ASC LIMIT 1`
              ).get(m.id, asOfStr) as any;

              if (!laterVersion) {
                standalone.push(m); // No later version at that time — this was current
              }
              // If there IS a later version, that version will be picked up by the root chain logic
            }
          }

          const candidates = [...standalone, ...Array.from(rootLatest.values())];

          // Score by semantic similarity
          const scored = candidates.map(m => {
            const emb = bufferToEmbedding(m.embedding);
            const sim = cosineSimilarity(embQ, emb);
            return { ...m, similarity: sim, embedding: undefined };
          }).filter(m => m.similarity > 0.3);

          scored.sort((a, b) => b.similarity - a.similarity);
          timeMemories = scored.slice(0, resultLimit);
        } else {
          // Just list memories as of that time
          let catFilter = "";
          const params: any[] = [asOfStr, auth.user_id];
          if (category) {
            catFilter = " AND category = ?";
            params.push(category);
          }
          params.push(resultLimit);

          timeMemories = db.prepare(
            `SELECT id, content, category, source, importance, created_at, updated_at,
               version, is_latest, root_memory_id, is_static, tags, confidence
             FROM memories
             WHERE created_at <= ? AND is_forgotten = 0 AND user_id = ?${catFilter}
             AND (is_latest = 1 OR updated_at > ?)
             ORDER BY created_at DESC LIMIT ?`
          ).all(...[...params.slice(0, -1), asOfStr, ...params.slice(-1)]) as any[];
        }

        // Count stats at that time
        const statsAtTime = db.prepare(
          `SELECT COUNT(*) as total,
             SUM(CASE WHEN is_static = 1 THEN 1 ELSE 0 END) as static_count,
             SUM(CASE WHEN category = 'task' THEN 1 ELSE 0 END) as tasks,
             SUM(CASE WHEN category = 'state' THEN 1 ELSE 0 END) as states,
             SUM(CASE WHEN category = 'decision' THEN 1 ELSE 0 END) as decisions,
             SUM(CASE WHEN category = 'discovery' THEN 1 ELSE 0 END) as discoveries,
             SUM(CASE WHEN category = 'issue' THEN 1 ELSE 0 END) as issues
           FROM memories WHERE created_at <= ? AND is_forgotten = 0 AND user_id = ?`
        ).get(asOfStr, auth.user_id) as any;

        return json({
          as_of: as_of,
          query: query || null,
          memories: timeMemories.map(m => ({
            id: m.id,
            content: m.content,
            category: m.category,
            source: m.source,
            importance: m.importance,
            version: m.version,
            is_static: !!m.is_static,
            tags: m.tags ? JSON.parse(m.tags) : [],
            created_at: m.created_at,
            similarity: m.similarity || undefined,
          })),
          stats: statsAtTime,
          total_returned: timeMemories.length,
        });
      } catch (e: any) {
        return errorResponse(`Time-travel failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // SMART CONTEXT BUILDER — optimal RAG context within token budget
    // ========================================================================

    if (url.pathname === "/context" && method === "POST") {
      try {
        const body = await req.json() as any;
        const { query, max_tokens, include_static, include_recent, strategy } = body;

        if (!query || typeof query !== "string") return errorResponse("query (string) required");

        const tokenBudget = Math.min(Number(max_tokens) || 4000, 32000);
        const includeStatic = include_static !== false; // default true
        const includeRecent = include_recent !== false; // default true
        const contextStrategy = strategy || "balanced"; // balanced | precision | breadth

        // Rough token estimation: ~4 chars per token
        const estimateTokens = (text: string) => Math.ceil(text.length / 4);

        interface ContextBlock {
          id: number;
          content: string;
          category: string;
          score: number;
          source: string; // "static" | "semantic" | "recent" | "linked"
          tokens: number;
        }

        const blocks: ContextBlock[] = [];
        let usedTokens = 0;

        // Phase 1: Static facts (always highest priority — these define identity/config)
        if (includeStatic) {
          const statics = getStaticMemories.all(auth.user_id) as any[];
          for (const s of statics) {
            const tokens = estimateTokens(s.content);
            if (usedTokens + tokens > tokenBudget * 0.4) break; // cap statics at 40% of budget
            blocks.push({
              id: s.id, content: s.content, category: s.category,
              score: 100, source: "static", tokens,
            });
            usedTokens += tokens;
          }
        }

        // Phase 2: Semantic search (core relevance)
        const semanticBudget = contextStrategy === "precision" ? 0.5 : contextStrategy === "breadth" ? 0.3 : 0.4;
        const semanticLimit = contextStrategy === "precision" ? 30 : contextStrategy === "breadth" ? 50 : 40;
        const semanticResults = await hybridSearch(query, semanticLimit, false, true, RERANKER_ENABLED && !!LLM_API_KEY);

        const seenIds = new Set(blocks.map(b => b.id));
        for (const r of semanticResults) {
          if (seenIds.has(r.id)) continue;
          const tokens = estimateTokens(r.content);
          if (usedTokens + tokens > tokenBudget * (0.4 + semanticBudget)) break;
          blocks.push({
            id: r.id, content: r.content, category: r.category,
            score: r.combined_score || r.semantic_score || 0, source: "semantic", tokens,
          });
          seenIds.add(r.id);
          usedTokens += tokens;
        }

        // Phase 3: Linked memories (expand context graph)
        if (contextStrategy !== "precision" && usedTokens < tokenBudget * 0.85) {
          const semanticIds = blocks.filter(b => b.source === "semantic").slice(0, 5).map(b => b.id);
          for (const sid of semanticIds) {
            const linked = getLinksFor.all(sid, sid) as any[];
            for (const l of linked) {
              if (seenIds.has(l.id) || l.is_forgotten) continue;
              const tokens = estimateTokens(l.content);
              if (usedTokens + tokens > tokenBudget * 0.9) break;
              blocks.push({
                id: l.id, content: l.content, category: l.category,
                score: l.similarity * 50, source: "linked", tokens,
              });
              seenIds.add(l.id);
              usedTokens += tokens;
            }
          }
        }

        // Phase 4: Recent memories (temporal context)
        if (includeRecent && usedTokens < tokenBudget * 0.95) {
          const recent = getRecentDynamicMemories.all(auth.user_id, 10) as any[];
          for (const r of recent) {
            if (seenIds.has(r.id)) continue;
            const tokens = estimateTokens(r.content);
            if (usedTokens + tokens > tokenBudget) break;
            blocks.push({
              id: r.id, content: r.content, category: r.category,
              score: 10, source: "recent", tokens,
            });
            seenIds.add(r.id);
            usedTokens += tokens;
          }
        }

        // Track access for all included memories
        for (const b of blocks) trackAccessWithFSRS(b.id);

        // Build formatted context string
        const contextParts: string[] = [];
        const staticBlocks = blocks.filter(b => b.source === "static");
        const semanticBlocks = blocks.filter(b => b.source === "semantic");
        const linkedBlocks = blocks.filter(b => b.source === "linked");
        const recentBlocks = blocks.filter(b => b.source === "recent");

        if (staticBlocks.length > 0) {
          contextParts.push("## Permanent Facts\n" + staticBlocks.map(b => `- ${b.content}`).join("\n"));
        }
        if (semanticBlocks.length > 0) {
          contextParts.push("## Relevant Memories\n" + semanticBlocks.map(b => `- [${b.category}] ${b.content}`).join("\n"));
        }
        if (linkedBlocks.length > 0) {
          contextParts.push("## Related Context\n" + linkedBlocks.map(b => `- ${b.content}`).join("\n"));
        }
        if (recentBlocks.length > 0) {
          contextParts.push("## Recent Activity\n" + recentBlocks.map(b => `- [${b.created_at || ""}] ${b.content}`).join("\n"));
        }

        return json({
          context: contextParts.join("\n\n"),
          blocks: blocks.map(b => ({ id: b.id, category: b.category, source: b.source, score: Math.round(b.score * 100) / 100, tokens: b.tokens })),
          token_estimate: usedTokens,
          token_budget: tokenBudget,
          utilization: Math.round(usedTokens / tokenBudget * 100) / 100,
          strategy: contextStrategy,
          breakdown: {
            static: staticBlocks.length,
            semantic: semanticBlocks.length,
            linked: linkedBlocks.length,
            recent: recentBlocks.length,
          },
        });
      } catch (e: any) {
        return errorResponse(`Context build failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // MEMORY REFLECTIONS — periodic meta-analysis
    // ========================================================================

    if (url.pathname === "/reflect" && method === "POST") {
      if (!LLM_API_KEY) return errorResponse("LLM not configured — /reflect requires inference", 400);
      try {
        const body = await req.json() as any;
        const period = body.period || "week"; // day | week | month
        const force = body.force === true;

        const now = new Date();
        let periodStart: Date;
        if (period === "day") periodStart = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        else if (period === "month") periodStart = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        else periodStart = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

        const periodStartStr = periodStart.toISOString().replace("T", " ").replace("Z", "");
        const periodEndStr = now.toISOString().replace("T", " ").replace("Z", "");

        // Check if we already reflected on this period
        if (!force) {
          const existing = db.prepare(
            `SELECT id, content, themes, created_at FROM reflections
             WHERE user_id = ? AND period_start >= ? ORDER BY created_at DESC LIMIT 1`
          ).get(auth.user_id, periodStartStr) as any;
          if (existing) {
            return json({
              reflection: existing.content,
              themes: existing.themes ? JSON.parse(existing.themes) : [],
              period: { start: periodStartStr, end: periodEndStr },
              cached: true,
              id: existing.id,
            });
          }
        }

        // Gather memories from the period
        const periodMemories = db.prepare(
          `SELECT id, content, category, importance, tags, created_at, is_static, confidence
           FROM memories WHERE created_at >= ? AND created_at <= ? AND is_forgotten = 0 AND user_id = ?
           ORDER BY importance DESC, created_at DESC LIMIT 100`
        ).all(periodStartStr, periodEndStr, auth.user_id) as any[];

        if (periodMemories.length < 3) {
          return json({
            reflection: null,
            message: `Only ${periodMemories.length} memories in the ${period} period — need at least 3 for reflection`,
            period: { start: periodStartStr, end: periodEndStr },
          });
        }

        // Get category distribution
        const categories: Record<string, number> = {};
        for (const m of periodMemories) {
          categories[m.category] = (categories[m.category] || 0) + 1;
        }

        const memoryList = periodMemories.map(m =>
          `[#${m.id} ${m.category} imp=${m.importance}] ${m.content.substring(0, 200)}`
        ).join("\n");

        const reflectPrompt = `You are a reflective intelligence analyzing a collection of memories from a ${period} period. Generate a meta-analysis that identifies:

1. **Key Themes**: The 3-5 dominant themes or areas of focus
2. **Progress Summary**: What was accomplished and what moved forward
3. **Patterns**: Recurring patterns, habits, or tendencies
4. **Unresolved Items**: Things mentioned but not completed or resolved
5. **Insights**: Non-obvious connections or observations

Category distribution: ${JSON.stringify(categories)}
Memory count: ${periodMemories.length}
Period: ${periodStartStr} to ${periodEndStr}

Return JSON:
{
  "reflection": "2-3 paragraph natural language reflection",
  "themes": ["theme1", "theme2", "theme3"],
  "progress": ["completed item 1", "completed item 2"],
  "patterns": ["pattern 1", "pattern 2"],
  "unresolved": ["unresolved item 1"],
  "insight": "one key non-obvious insight"
}`;

        const resp = await callLLM(reflectPrompt, memoryList);
        const cleaned = resp.replace(/```json\n?|\n?```/g, "").trim();
        let result: any;
        try {
          result = JSON.parse(cleaned);
        } catch {
          const jsonMatch = cleaned.match(/\{[\s\S]*"reflection"[\s\S]*\}/);
          if (jsonMatch) result = JSON.parse(jsonMatch[0]);
          else return errorResponse("LLM returned unparseable reflection", 500);
        }

        // Store the reflection
        const reflectionId = db.prepare(
          `INSERT INTO reflections (user_id, content, themes, period_start, period_end, memory_count, source_memory_ids)
           VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id`
        ).get(
          auth.user_id,
          result.reflection,
          JSON.stringify(result.themes || []),
          periodStartStr,
          periodEndStr,
          periodMemories.length,
          JSON.stringify(periodMemories.map((m: any) => m.id))
        ) as { id: number };

        // Also store the reflection as a memory for future recall
        let embBuffer: Buffer | null = null;
        let embArray: Float32Array | null = null;
        try { embArray = await embed(result.reflection); embBuffer = embeddingToBuffer(embArray); } catch {}

        const reflectionMem = insertMemory.get(
          `[Reflection: ${period}ly, ${periodStartStr.substring(0, 10)} to ${periodEndStr.substring(0, 10)}] ${result.reflection}`,
          "discovery", "reflection", null, 7, embBuffer,
          1, 1, null, null, periodMemories.length, 1, 0, null, null, 1
        ) as { id: number; created_at: string };
        db.prepare("UPDATE memories SET user_id = ?, tags = ? WHERE id = ?").run(
          auth.user_id,
          JSON.stringify(["reflection", period, ...(result.themes || []).slice(0, 3)]),
          reflectionMem.id
        );
        if (embArray) await autoLink(reflectionMem.id, embArray);

        emitWebhookEvent("reflection.created", {
          id: reflectionId.id,
          period,
          themes: result.themes,
          memory_count: periodMemories.length,
        }, auth.user_id);

        return json({
          id: reflectionId.id,
          memory_id: reflectionMem.id,
          reflection: result.reflection,
          themes: result.themes || [],
          progress: result.progress || [],
          patterns: result.patterns || [],
          unresolved: result.unresolved || [],
          insight: result.insight || null,
          period: { start: periodStartStr, end: periodEndStr },
          memories_analyzed: periodMemories.length,
          cached: false,
        });
      } catch (e: any) {
        return errorResponse(`Reflection failed: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/reflections" && method === "GET") {
      const limitParam = Math.min(Number(url.searchParams.get("limit") || 10), 50);
      const reflections = db.prepare(
        `SELECT id, content, themes, period_start, period_end, memory_count, created_at
         FROM reflections WHERE user_id = ? ORDER BY created_at DESC LIMIT ?`
      ).all(auth.user_id, limitParam) as any[];

      return json({
        reflections: reflections.map(r => ({
          ...r,
          themes: r.themes ? JSON.parse(r.themes) : [],
        })),
        total: reflections.length,
      });
    }

    // ========================================================================
    // SCHEDULED DIGESTS — webhook delivery of memory summaries
    // ========================================================================

    if (url.pathname === "/digests" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        const { webhook_url, webhook_secret, schedule, include_stats, include_new_memories, include_contradictions, include_reflections } = body;

        if (!webhook_url || typeof webhook_url !== "string" || !webhook_url.match(/^https?:\/\//)) {
          return errorResponse("webhook_url (valid http/https URL) required");
        }

        const sched = ["hourly", "daily", "weekly"].includes(schedule) ? schedule : "daily";

        // Calculate next send time
        const now = new Date();
        let nextSend: Date;
        if (sched === "hourly") nextSend = new Date(now.getTime() + 60 * 60 * 1000);
        else if (sched === "weekly") nextSend = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
        else nextSend = new Date(now.getTime() + 24 * 60 * 60 * 1000);

        const nextSendStr = nextSend.toISOString().replace("T", " ").replace("Z", "");

        const result = db.prepare(
          `INSERT INTO digests (user_id, schedule, webhook_url, webhook_secret, include_stats, include_new_memories, include_contradictions, include_reflections, next_send_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id, created_at`
        ).get(
          auth.user_id, sched, webhook_url, webhook_secret || null,
          include_stats !== false ? 1 : 0,
          include_new_memories !== false ? 1 : 0,
          include_contradictions !== false ? 1 : 0,
          include_reflections !== false ? 1 : 0,
          nextSendStr
        ) as { id: number; created_at: string };

        return json({
          id: result.id,
          schedule: sched,
          webhook_url,
          next_send_at: nextSendStr,
          created_at: result.created_at,
        });
      } catch (e: any) {
        return errorResponse(`Digest creation failed: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/digests" && method === "GET") {
      const digests = db.prepare(
        `SELECT id, schedule, webhook_url, include_stats, include_new_memories,
           include_contradictions, include_reflections, last_sent_at, next_send_at, active, created_at
         FROM digests WHERE user_id = ? ORDER BY created_at DESC`
      ).all(auth.user_id) as any[];
      return json({ digests });
    }

    if (url.pathname.match(/^\/digests\/\d+$/) && method === "DELETE") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const digestId = Number(url.pathname.split("/")[2]);
      db.prepare("DELETE FROM digests WHERE id = ? AND user_id = ?").run(digestId, auth.user_id);
      return json({ deleted: true, id: digestId });
    }

    if (url.pathname === "/digests/send" && method === "POST") {
      // Manually trigger a digest send (for testing)
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        const digestId = body.digest_id;
        if (!digestId) return errorResponse("digest_id required");

        const digest = db.prepare("SELECT * FROM digests WHERE id = ? AND user_id = ?").get(digestId, auth.user_id) as any;
        if (!digest) return errorResponse("Digest not found", 404);

        const payload = await buildDigestPayload(digest, auth.user_id);
        await sendDigestWebhook(digest, payload);

        return json({ sent: true, digest_id: digestId, payload });
      } catch (e: any) {
        return errorResponse(`Digest send failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // STORE — v3 with async fact extraction
    // ========================================================================

    if ((url.pathname === "/store" || url.pathname === "/memory" || url.pathname === "/memories") && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json();
        const { content, category, source, session_id, importance, tags, episode } = body;
        if (!content || typeof content !== "string" || content.trim().length === 0) {
          return errorResponse("content is required and must be a non-empty string");
        }

        const imp = Math.max(1, Math.min(10, Number(importance) || DEFAULT_IMPORTANCE));

        // Validate and serialize tags
        let tagsJson: string | null = null;
        if (tags) {
          if (Array.isArray(tags)) {
            tagsJson = JSON.stringify(tags.map((t: any) => String(t).trim().toLowerCase()).filter(Boolean));
          } else if (typeof tags === "string") {
            tagsJson = JSON.stringify(tags.split(",").map(t => t.trim().toLowerCase()).filter(Boolean));
          }
        }

        // Episode management: auto-create or find existing episode
        let episodeId: number | null = null;
        if (episode !== false && session_id && source) {
          const existing = getEpisodeBySession.get(session_id, source, auth.user_id) as any;
          if (existing) {
            episodeId = existing.id;
          } else if (episode !== "none") {
            // Auto-create episode for this session
            const ep = insertEpisode.get(
              null, session_id, source, auth.user_id
            ) as { id: number; started_at: string };
            episodeId = ep.id;
          }
        }

        let embBuffer: Buffer | null = null;
        let embArray: Float32Array | null = null;
        try {
          embArray = await embed(content.trim());
          embBuffer = embeddingToBuffer(embArray);
        } catch (e: any) {
          console.error("Embedding failed, storing without:", e.message);
        }

        const result = insertMemory.get(
          content.trim(),
          (category || "general").trim(),
          (source || "unknown").trim(),
          session_id || null,
          imp,
          embBuffer,
          1, 1, null, null, 1, 0, 0, null, null, 0
        ) as { id: number; created_at: string };

        // Set user_id, space_id, tags, episode_id, sync_id, confidence
        const syncId = crypto.randomUUID();
        const memStatus = body.status === "pending" ? "pending" : "approved";
        db.prepare(
          "UPDATE memories SET user_id = ?, space_id = ?, tags = ?, episode_id = ?, sync_id = ?, confidence = 1.0, status = ? WHERE id = ?"
        ).run(auth.user_id, auth.space_id || null, tagsJson, episodeId, syncId, memStatus, result.id);

        // Link to entities and projects if provided
        const entityIds = body.entity_ids as number[] | undefined;
        const projectIds = body.project_ids as number[] | undefined;
        if (entityIds && Array.isArray(entityIds)) {
          for (const eid of entityIds) linkMemoryEntity.run(result.id, eid);
        }
        if (projectIds && Array.isArray(projectIds)) {
          for (const pid of projectIds) linkMemoryProject.run(result.id, pid);
        }

        // Update episode memory count
        if (episodeId) {
          updateEpisode.run(null, null, null, episodeId);
        }

        // Calculate initial decay score + FSRS state
        const initFSRS = fsrsProcessReview(null, FSRSRating.Good, 0);
        const decayScore = calculateDecayScore(imp, result.created_at, 0, null, false, 1, initFSRS.stability);
        db.prepare("UPDATE memories SET decay_score = ?, fsrs_stability = ?, fsrs_difficulty = ?, fsrs_storage_strength = ?, fsrs_retrieval_strength = ?, fsrs_learning_state = ?, fsrs_reps = ?, fsrs_lapses = ?, fsrs_last_review_at = ? WHERE id = ?").run(
          Math.round(decayScore * 1000) / 1000,
          initFSRS.stability, initFSRS.difficulty, initFSRS.storage_strength,
          initFSRS.retrieval_strength, initFSRS.learning_state, initFSRS.reps, initFSRS.lapses,
          initFSRS.last_review_at, result.id
        );

        let linked = 0;
        if (embArray) {
          writeVec(result.id, embArray);
          linked = await autoLink(result.id, embArray);
        }

        // Async fact extraction (non-blocking)
        if (LLM_API_KEY && embArray) {
          const capturedEmbArray = embArray;
          (async () => {
            try {
              const allMems = getLatestEmbeddings.all() as Array<{
                id: number; content: string; category: string; importance: number; embedding: Buffer;
              }>;
              const similarities: Array<{ id: number; content: string; category: string; score: number }> = [];
              for (const mem of allMems) {
                if (mem.id === result.id || !mem.embedding) continue;
                const memEmb = bufferToEmbedding(mem.embedding);
                const sim = cosineSimilarity(capturedEmbArray, memEmb);
                if (sim > 0.4) {
                  similarities.push({ id: mem.id, content: mem.content, category: mem.category, score: sim });
                }
              }
              similarities.sort((a, b) => b.score - a.score);
              const top3 = similarities.slice(0, 3);

              const extraction = await extractFacts(content.trim(), category || "general", top3);
              if (extraction) {
                processExtractionResult(result.id, extraction, capturedEmbArray);
                console.log(`Fact extraction complete for memory #${result.id}: ${extraction.relation_to_existing.type}`);
              }
            } catch (e: any) {
              console.error(`Async fact extraction failed for #${result.id}:`, e.message);
            }
          })();
        }

        // Emit webhook event
        emitWebhookEvent("memory.created", {
          id: result.id, content: content.trim(), category: category || "general",
          importance: imp, tags: tagsJson ? JSON.parse(tagsJson) : [], episode_id: episodeId,
        }, auth.user_id);

        return json({
          stored: true,
          id: result.id,
          created_at: result.created_at,
          importance: imp,
          linked,
          embedded: !!embBuffer,
          tags: tagsJson ? JSON.parse(tagsJson) : [],
          episode_id: episodeId,
          decay_score: decayScore,
          fact_extraction: LLM_API_KEY ? "queued" : "disabled",
          status: memStatus,
        });
      } catch (e: any) {
        return errorResponse(`Failed to store: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // SEARCH — v3
    // ========================================================================

    if ((url.pathname === "/search" || url.pathname === "/memories/search") && method === "POST") {
      try {
        const body = await req.json();
        const { query, limit, include_links, expand_relationships, latest_only, tag, episode_id: filterEpisode } = body;
        if (!query || typeof query !== "string") return errorResponse("query is required");
        let results = await hybridSearch(
          query,
          Math.min(limit || 10, 50),
          include_links || false,
          expand_relationships ?? true,
          latest_only ?? true
        );

        // Filter by tag if specified
        if (tag) {
          results = results.filter(r => {
            const mem = getMemoryWithoutEmbedding.get(r.id) as any;
            if (!mem?.tags) return false;
            try { return JSON.parse(mem.tags).includes(tag); } catch { return false; }
          });
        }

        // Filter by episode if specified
        if (filterEpisode) {
          results = results.filter(r => {
            const mem = getMemoryWithoutEmbedding.get(r.id) as any;
            return mem?.episode_id === filterEpisode;
          });
        }

        // Rerank results for better precision
        const doRerank = body.rerank !== false; // opt-out with rerank: false
        if (doRerank && results.length > 3) {
          results = await rerank(query, results);
          // Re-trim to requested limit after reranking
          results = results.slice(0, Math.min(limit || 10, 50));
        }

        // Track access on returned results
        for (const r of results) {
          trackAccessWithFSRS(r.id);
        }

        // Episodic expansion: if a result belongs to an episode, include episode context
        const episodeContext: Array<{ episode_id: number; title: string; memories: any[] }> = [];
        const seenEpisodes = new Set<number>();
        for (const r of results) {
          const mem = getMemoryWithoutEmbedding.get(r.id) as any;
          if (mem?.episode_id && !seenEpisodes.has(mem.episode_id)) {
            seenEpisodes.add(mem.episode_id);
            const ep = getEpisode.get(mem.episode_id) as any;
            if (ep) {
              const epMems = getEpisodeMemories.all(mem.episode_id) as any[];
              episodeContext.push({
                episode_id: mem.episode_id,
                title: ep.title || `Session ${ep.session_id || ep.id}`,
                memories: epMems.map(m => ({ id: m.id, content: m.content, category: m.category })),
              });
            }
          }
          // Attach tags to results
          if (mem?.tags) {
            try { r.tags = JSON.parse(mem.tags); } catch {}
          }
          r.episode_id = mem?.episode_id;
          r.access_count = mem?.access_count;
        }

        return json({
          results,
          ...(episodeContext.length > 0 ? { episodes: episodeContext } : {}),
        });
      } catch (e: any) {
        return errorResponse(`Search failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // LIST
    // ========================================================================

    if (url.pathname === "/list" && method === "GET") {
      const limit = Math.min(Number(url.searchParams.get("limit") || 20), 100);
      const category = url.searchParams.get("category");
      const results = category ? listByCategory.all(category, auth.user_id, limit) : listRecent.all(auth.user_id, limit);
      return json({ results });
    }

    // ========================================================================
    // MEMORY — get / delete / forget
    // ========================================================================

    if (url.pathname.match(/^\/memory\/\d+\/forget$/) && method === "POST") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      const body = await req.json().catch(() => ({})) as any;
      markForgotten.run(id);
      if (body.reason) {
        db.prepare("UPDATE memories SET forget_reason = ? WHERE id = ?").run(body.reason, id);
      }
      return json({ forgotten: true, id });
    }

    // ========================================================================
    // ARCHIVE / UNARCHIVE
    // ========================================================================

    if (url.pathname.match(/^\/memory\/\d+\/archive$/) && method === "POST") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      const mem = getMemoryWithoutEmbedding.get(id) as any;
      if (!mem) return errorResponse("Not found", 404);
      markArchived.run(id);
      return json({ archived: true, id });
    }

    if (url.pathname.match(/^\/memory\/\d+\/unarchive$/) && method === "POST") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      const mem = getMemoryWithoutEmbedding.get(id) as any;
      if (!mem) return errorResponse("Not found", 404);
      markUnarchived.run(id);
      return json({ unarchived: true, id });
    }

    // ========================================================================
    // UPDATE (versioned) — creates new version via version chain
    // ========================================================================

    if (url.pathname.match(/^\/memory\/\d+\/update$/) && method === "POST") {
      try {
        const id = Number(url.pathname.split("/")[2]);
        if (isNaN(id)) return errorResponse("Invalid id");
        const existing = getMemoryWithoutEmbedding.get(id) as any;
        if (!existing) return errorResponse("Not found", 404);
        if (existing.is_forgotten) return errorResponse("Cannot update a forgotten memory", 400);

        const body = await req.json() as any;
        const newContent = body.content;
        if (!newContent || typeof newContent !== "string" || newContent.trim().length === 0) {
          return errorResponse("content is required and must be a non-empty string");
        }

        const category = body.category || existing.category;
        const imp = body.importance ? Math.max(1, Math.min(10, Number(body.importance))) : existing.importance;

        // Embed the new content
        let embBuffer: Buffer | null = null;
        let embArray: Float32Array | null = null;
        try {
          embArray = await embed(newContent.trim());
          embBuffer = embeddingToBuffer(embArray);
        } catch (e: any) {
          console.error("Embedding failed for update:", e.message);
        }

        // Determine version chain
        const rootId = existing.root_memory_id || existing.id;
        const newVersion = (existing.version || 1) + 1;

        // Mark the old memory as superseded
        markSuperseded.run(id);

        // Insert the new version
        const result = insertMemory.get(
          newContent.trim(),
          category,
          existing.source,
          existing.session_id,
          imp,
          embBuffer,
          newVersion,      // version
          1,               // is_latest
          id,              // parent_memory_id
          rootId,          // root_memory_id
          existing.source_count || 1,
          existing.is_static ? 1 : 0,
          0,               // is_forgotten
          null,            // forget_after
          null,            // forget_reason
          existing.is_inference ? 1 : 0
        ) as { id: number; created_at: string };

        // Link old -> new as "updates"
        insertLink.run(result.id, id, 1.0, "updates");

        // Auto-link new version
        let linked = 0;
        if (embArray) {
          writeVec(result.id, embArray);
          linked = await autoLink(result.id, embArray);
        }

        console.log(`Memory #${id} updated -> #${result.id} (v${newVersion}, root=#${rootId})`);

        return json({
          updated: true,
          old_id: id,
          new_id: result.id,
          version: newVersion,
          root_id: rootId,
          linked,
          embedded: !!embBuffer,
        });
      } catch (e: any) {
        return errorResponse(`Update failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // DUPLICATES — find near-duplicate memory clusters
    // ========================================================================

    if (url.pathname === "/duplicates" && method === "GET") {
      try {
        const threshold = Number(url.searchParams.get("threshold") || 0.85);
        const limitParam = Math.min(Number(url.searchParams.get("limit") || 50), 200);

        const allMems = getLatestEmbeddings.all() as Array<{
          id: number; content: string; category: string; importance: number;
          embedding: Buffer; is_static: boolean; source_count: number;
        }>;

        // Find clusters of similar memories
        const clusters: Array<{
          anchor: { id: number; content: string; category: string };
          duplicates: Array<{ id: number; content: string; category: string; similarity: number }>;
        }> = [];
        const seen = new Set<number>();

        for (let i = 0; i < allMems.length; i++) {
          if (seen.has(allMems[i].id)) continue;
          const embA = bufferToEmbedding(allMems[i].embedding);
          const dupes: Array<{ id: number; content: string; category: string; similarity: number }> = [];

          for (let j = i + 1; j < allMems.length; j++) {
            if (seen.has(allMems[j].id)) continue;
            const embB = bufferToEmbedding(allMems[j].embedding);
            const sim = cosineSimilarity(embA, embB);
            if (sim >= threshold) {
              dupes.push({
                id: allMems[j].id,
                content: allMems[j].content.substring(0, 200),
                category: allMems[j].category,
                similarity: Math.round(sim * 1000) / 1000,
              });
              seen.add(allMems[j].id);
            }
          }

          if (dupes.length > 0) {
            seen.add(allMems[i].id);
            clusters.push({
              anchor: {
                id: allMems[i].id,
                content: allMems[i].content.substring(0, 200),
                category: allMems[i].category,
              },
              duplicates: dupes,
            });
            if (clusters.length >= limitParam) break;
          }
        }

        return json({ threshold, clusters, total_clusters: clusters.length });
      } catch (e: any) {
        return errorResponse(`Duplicate scan failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // DEDUPLICATE — merge duplicate clusters (keep anchor, archive dupes)
    // ========================================================================

    if (url.pathname === "/deduplicate" && method === "POST") {
      try {
        const body = await req.json() as any;
        const threshold = Number(body.threshold || 0.85);
        const dryRun = body.dry_run !== false; // default to dry run for safety
        const maxMerge = Math.min(Number(body.max_merge || 50), 500);

        const allMems = getLatestEmbeddings.all() as Array<{
          id: number; content: string; category: string; importance: number;
          embedding: Buffer; is_static: boolean; source_count: number;
        }>;

        const merged: Array<{ kept: number; archived: number[]; similarity: number }> = [];
        const seen = new Set<number>();
        let totalArchived = 0;

        for (let i = 0; i < allMems.length && merged.length < maxMerge; i++) {
          if (seen.has(allMems[i].id)) continue;
          const embA = bufferToEmbedding(allMems[i].embedding);
          const dupes: Array<{ id: number; similarity: number; source_count: number }> = [];

          for (let j = i + 1; j < allMems.length; j++) {
            if (seen.has(allMems[j].id)) continue;
            const embB = bufferToEmbedding(allMems[j].embedding);
            const sim = cosineSimilarity(embA, embB);
            if (sim >= threshold) {
              dupes.push({ id: allMems[j].id, similarity: sim, source_count: allMems[j].source_count || 1 });
              seen.add(allMems[j].id);
            }
          }

          if (dupes.length > 0) {
            seen.add(allMems[i].id);

            if (!dryRun) {
              // Aggregate source_count to the kept memory
              let totalSourceCount = allMems[i].source_count || 1;
              for (const d of dupes) {
                totalSourceCount += (d.source_count || 1) - 1;
                markArchived.run(d.id);
                insertLink.run(allMems[i].id, d.id, d.similarity, "derives");
                totalArchived++;
              }
              db.prepare("UPDATE memories SET source_count = ?, updated_at = datetime('now') WHERE id = ?")
                .run(totalSourceCount, allMems[i].id);
            }

            merged.push({
              kept: allMems[i].id,
              archived: dupes.map(d => d.id),
              similarity: Math.round(dupes[0].similarity * 1000) / 1000,
            });
          }
        }

        return json({
          dry_run: dryRun,
          threshold,
          clusters_found: merged.length,
          total_archived: dryRun ? 0 : totalArchived,
          merges: merged,
        });
      } catch (e: any) {
        return errorResponse(`Dedup failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // SMART RECALL — context-aware memory retrieval for plugin
    // ========================================================================

    if (url.pathname === "/recall" && method === "POST") {
      try {
        const body = await req.json() as any;
        const context = body.context || body.query || ""; // 'query' for BotMemory compat
        const limit = Math.min(Number(body.limit) || 20, 50);
        const includeTags = body.tags as string[] | undefined;

        const results: Map<number, { memory: any; score: number; source: string }> = new Map();

        // 1. Static facts (always included, highest priority)
        const staticFacts = getStaticMemories.all(auth.user_id) as Array<any>;
        for (const sf of staticFacts) {
          results.set(sf.id, { memory: sf, score: 100, source: "static" });
        }

        // 2. Semantic search against context (if provided)
        if (context.trim()) {
          const semanticResults = await hybridSearch(context, limit, false, true, true);
          for (const sr of semanticResults) {
            if (!results.has(sr.id)) {
              // Use decay_score instead of raw search score
              const decayMultiplier = sr.decay_score ? (sr.decay_score / sr.importance) : 1;
              results.set(sr.id, { memory: sr, score: sr.score * 50 * decayMultiplier, source: "semantic" });
            }
          }
        }

        // 3. High-importance memories weighted by decay (not just raw importance)
        const recentImportant = db.prepare(
          `SELECT id, content, category, source, importance, created_at, source_count, is_static,
             access_count, last_accessed_at, decay_score, tags, episode_id
           FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
           ORDER BY COALESCE(decay_score, importance) DESC, created_at DESC LIMIT ?`
        ).all(auth.user_id, limit) as Array<any>;
        for (const ri of recentImportant) {
          if (!results.has(ri.id)) {
            const effectiveScore = ri.decay_score || ri.importance;
            results.set(ri.id, { memory: ri, score: effectiveScore * 2, source: "important" });
          }
        }

        // 4. Recent activity (fill any remaining)
        const recent = listRecent.all(auth.user_id, Math.min(limit, 15)) as Array<any>;
        for (const r of recent) {
          if (!results.has(r.id)) {
            results.set(r.id, { memory: r, score: 1, source: "recent" });
          }
        }

        // 5. Tag-based boost: if caller specifies tags, boost matching memories
        if (includeTags && includeTags.length > 0) {
          for (const [id, entry] of results) {
            const mem = getMemoryWithoutEmbedding.get(id) as any;
            if (mem?.tags) {
              try {
                const memTags = JSON.parse(mem.tags) as string[];
                const overlap = includeTags.filter(t => memTags.includes(t)).length;
                if (overlap > 0) entry.score *= (1 + overlap * 0.2);
              } catch {}
            }
          }
        }

        // Sort by score descending, limit
        const sorted = Array.from(results.values())
          .sort((a, b) => b.score - a.score)
          .slice(0, limit);

        // Track access on recalled memories
        for (const s of sorted) {
          trackAccessWithFSRS(s.memory.id);
        }

        // Episodic expansion: find episodes referenced by recalled memories
        const episodeContext: Array<{ episode_id: number; title: string; memory_count: number }> = [];
        const seenEpisodes = new Set<number>();
        for (const s of sorted) {
          const mem = getMemoryWithoutEmbedding.get(s.memory.id) as any;
          if (mem?.episode_id && !seenEpisodes.has(mem.episode_id)) {
            seenEpisodes.add(mem.episode_id);
            const ep = getEpisode.get(mem.episode_id) as any;
            if (ep) {
              episodeContext.push({
                episode_id: mem.episode_id,
                title: ep.title || `Session ${ep.session_id || ep.id}`,
                memory_count: ep.memory_count,
              });
            }
          }
        }

        return json({
          // Standard Engram format
          memories: sorted.map(s => ({
            ...s.memory,
            recall_source: s.source,
            recall_score: Math.round(s.score * 100) / 100,
            tags: s.memory.tags ? (() => { try { return JSON.parse(s.memory.tags); } catch { return []; } })() : [],
          })),
          breakdown: {
            static: sorted.filter(s => s.source === "static").length,
            semantic: sorted.filter(s => s.source === "semantic").length,
            important: sorted.filter(s => s.source === "important").length,
            recent: sorted.filter(s => s.source === "recent").length,
          },
          ...(episodeContext.length > 0 ? { episodes: episodeContext } : {}),
          // BotMemory v1 compat fields (for Discord bots)
          profile: sorted.filter(s => s.source === "static").map(s => s.memory.content),
          recent: sorted.filter(s => s.source === "recent").map(s => ({
            id: s.memory.id, content: s.memory.content, category: s.memory.category,
            source: s.memory.source, createdAt: s.memory.created_at,
          })),
          results: sorted.filter(s => s.source === "semantic" || s.source === "important").map(s => ({
            id: s.memory.id, content: s.memory.content, category: s.memory.category,
            source: s.memory.source, score: s.score, createdAt: s.memory.created_at,
          })),
          count: sorted.length,
        });
      } catch (e: any) {
        return errorResponse(`Smart recall failed: ${e.message}`, 500);
      }
    }

    if (url.pathname.startsWith("/memory/") && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      const memory = getMemoryWithoutEmbedding.get(id) as any;
      if (!memory) return errorResponse("Not found", 404);

      // Track access
      trackAccessWithFSRS(id);

      const links = getLinksFor.all(id, id) as Array<{
        id: number; similarity: number; type: string; content: string; category: string;
      }>;

      const rootId = memory.root_memory_id || memory.id;
      const chain = getVersionChain.all(rootId, rootId) as Array<{
        id: number; content: string; version: number; is_latest: boolean;
        created_at: string; source_count: number;
      }>;

      // Parse tags and include episode info
      let tags: string[] = [];
      try { tags = memory.tags ? JSON.parse(memory.tags) : []; } catch {}
      let episode = null;
      if (memory.episode_id) {
        episode = getEpisode.get(memory.episode_id) as any;
      }

      return json({
        ...memory,
        tags,
        episode: episode ? { id: episode.id, title: episode.title, session_id: episode.session_id } : null,
        decay_score: calculateDecayScore(
          memory.importance, memory.created_at, memory.access_count || 0,
          memory.last_accessed_at, !!memory.is_static, memory.source_count || 1
        ),
        links: links.map(l => ({ ...l })),
        version_chain: chain.length > 1 ? chain : undefined,
      });
    }

    if (url.pathname.startsWith("/memory/") && method === "DELETE") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      deleteMemory.run(id);
      return json({ deleted: true, id });
    }

    // ========================================================================
    // BACKFILL
    // ========================================================================

    if (url.pathname === "/backfill" && method === "POST") {
      try {
        const body = await req.json().catch(() => ({}));
        const batch = Math.min(Number((body as any).batch) || 50, 200);
        const count = await backfillEmbeddings(batch);
        const remaining = (countNoEmbedding.get() as { count: number }).count;
        return json({ backfilled: count, remaining });
      } catch (e: any) {
        return errorResponse(`Backfill failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // LINKS
    // ========================================================================

    if (url.pathname.startsWith("/links/") && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      const links = getLinksFor.all(id, id);
      return json({ memory_id: id, links });
    }

    // ========================================================================
    // PROFILE
    // ========================================================================

    if (url.pathname === "/profile" && method === "GET") {
      try {
        const summary = url.searchParams.get("summary") === "true";
        const profile = await generateProfile(auth.user_id, summary);
        return json(profile);
      } catch (e: any) {
        return errorResponse(`Profile generation failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // RAW GRAPH DATA (legacy — use GET /graph with params instead)
    // ========================================================================

    if (url.pathname === "/graph/raw" && method === "GET") {
      const memories = getAllMemoriesForGraph.all(auth.user_id);
      const links = getAllLinksForGraph.all(auth.user_id);
      return json({ memories, links });
    }

    // ========================================================================
    // VERSION CHAIN
    // ========================================================================

    if (url.pathname.startsWith("/versions/") && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      const mem = getMemoryWithoutEmbedding.get(id) as any;
      if (!mem) return errorResponse("Not found", 404);
      const rootId = mem.root_memory_id || mem.id;
      const chain = getVersionChain.all(rootId, rootId);
      return json({ root_id: rootId, chain });
    }

    // ========================================================================
    // SWEEP
    // ========================================================================

    if (url.pathname === "/sweep" && method === "POST") {
      const count = sweepExpiredMemories();
      return json({ swept: count });
    }

    // ========================================================================
    // CONVERSATIONS
    // ========================================================================

    if (url.pathname === "/conversations" && method === "POST") {
      try {
        const body = await req.json();
        const { agent, session_id, title, metadata } = body;
        if (!agent || typeof agent !== "string") return errorResponse("agent is required");
        const result = insertConversation.get(
          agent.trim(), session_id || null, title || null,
          metadata ? JSON.stringify(metadata) : null
        ) as { id: number; started_at: string };
        db.prepare("UPDATE conversations SET user_id = ? WHERE id = ?").run(auth.user_id, result.id);
        return json({ id: result.id, started_at: result.started_at });
      } catch (e: any) {
        return errorResponse(`Failed to create conversation: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/conversations" && method === "GET") {
      const limit = Math.min(Number(url.searchParams.get("limit") || 50), 500);
      const agent = url.searchParams.get("agent");
      const results = agent
        ? listConversationsByAgent.all(agent, limit)
        : listConversations.all(limit);
      return json({ results });
    }

    if (/^\/conversations\/\d+$/.test(url.pathname) && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      const conv = getConversation.get(id);
      if (!conv) return errorResponse("Not found", 404);
      const limit = Math.min(Number(url.searchParams.get("limit") || 10000), 100000);
      const offset = Number(url.searchParams.get("offset") || 0);
      const msgs = getMessages.all(id, limit, offset);
      return json({ conversation: conv, messages: msgs });
    }

    if (/^\/conversations\/\d+$/.test(url.pathname) && method === "PATCH") {
      try {
        const id = Number(url.pathname.split("/")[2]);
        const body = await req.json();
        updateConversation.run(
          body.title || null,
          body.metadata ? JSON.stringify(body.metadata) : null,
          id
        );
        return json({ updated: true, id });
      } catch (e: any) {
        return errorResponse(`Failed to update: ${e.message}`, 500);
      }
    }

    if (/^\/conversations\/\d+$/.test(url.pathname) && method === "DELETE") {
      const id = Number(url.pathname.split("/")[2]);
      deleteConversation.run(id);
      return json({ deleted: true, id });
    }

    // ========================================================================
    // MESSAGES
    // ========================================================================

    if (/^\/conversations\/\d+\/messages$/.test(url.pathname) && method === "POST") {
      try {
        const convId = Number(url.pathname.split("/")[2]);
        const conv = getConversation.get(convId);
        if (!conv) return errorResponse("Conversation not found", 404);
        const body = await req.json();
        const msgs = Array.isArray(body) ? body : [body];
        const results: Array<{ id: number; created_at: string }> = [];
        for (const msg of msgs) {
          if (!msg.role || !msg.content) continue;
          const result = insertMessage.get(
            convId, msg.role, msg.content, msg.metadata ? JSON.stringify(msg.metadata) : null
          ) as { id: number; created_at: string };
          results.push(result);
        }
        touchConversation.run(convId);
        return json({ added: results.length, messages: results });
      } catch (e: any) {
        return errorResponse(`Failed to add messages: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // BULK + UPSERT
    // ========================================================================

    if (url.pathname === "/conversations/bulk" && method === "POST") {
      try {
        const body = await req.json();
        const { agent, session_id, title, metadata, messages: msgs } = body;
        if (!agent) return errorResponse("agent is required");
        if (!msgs || !Array.isArray(msgs) || msgs.length === 0) {
          return errorResponse("messages array is required and must not be empty");
        }
        const conv = bulkInsertConvo(
          agent.trim(), session_id || null, title || null,
          metadata ? JSON.stringify(metadata) : null,
          msgs.map((m: any) => ({
            role: m.role || "user",
            content: m.content || "",
            metadata: m.metadata ? JSON.stringify(m.metadata) : null,
          }))
        );
        return json({ id: conv.id, started_at: conv.started_at, messages: msgs.length });
      } catch (e: any) {
        return errorResponse(`Bulk store failed: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/conversations/upsert" && method === "POST") {
      try {
        const body = await req.json();
        const { agent, session_id, title, metadata, messages: msgs } = body;
        if (!agent) return errorResponse("agent is required");
        if (!session_id) return errorResponse("session_id is required for upsert");

        let conv = getConversationBySession.get(agent, session_id) as any;
        let created = false;
        if (!conv) {
          const result = insertConversation.get(
            agent, session_id, title || null,
            metadata ? JSON.stringify(metadata) : null
          ) as { id: number; started_at: string };
          conv = { id: result.id };
          created = true;
        } else if (title || metadata) {
          updateConversation.run(
            title || null,
            metadata ? JSON.stringify(metadata) : null,
            conv.id
          );
        }

        let added = 0;
        if (msgs && Array.isArray(msgs)) {
          for (const msg of msgs) {
            if (!msg.role || !msg.content) continue;
            insertMessage.run(
              conv.id, msg.role, msg.content,
              msg.metadata ? JSON.stringify(msg.metadata) : null
            );
            added++;
          }
          if (added > 0) touchConversation.run(conv.id);
        }
        return json({ id: conv.id, created, added });
      } catch (e: any) {
        return errorResponse(`Upsert failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // SEARCH MESSAGES
    // ========================================================================

    if (url.pathname === "/messages/search" && method === "POST") {
      try {
        const body = await req.json();
        const { query, limit } = body;
        if (!query || typeof query !== "string") return errorResponse("query is required");
        const sanitized = sanitizeFTS(query);
        if (!sanitized) return json({ results: [] });
        const results = searchMessages.all(sanitized, Math.min(limit || 30, 200));
        return json({ results });
      } catch (e: any) {
        return errorResponse(`Search failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // GUI CRUD — create/edit/delete memories from the web interface
    // ========================================================================

    if (url.pathname === "/gui/memories" && method === "POST") {
      if (!guiAuthed(req)) return errorResponse("GUI auth required", 401);
      try {
        const body = await req.json() as any;
        if (!body.content?.trim()) return errorResponse("content is required");
        const imp = Math.max(1, Math.min(10, Number(body.importance) || 5));
        let tagsJson: string | null = null;
        if (body.tags) {
          const tags = Array.isArray(body.tags) ? body.tags : body.tags.split(",");
          tagsJson = JSON.stringify(tags.map((t: any) => String(t).trim().toLowerCase()).filter(Boolean));
        }
        let embBuffer: Buffer | null = null;
        try {
          const embArray = await embed(body.content.trim());
          embBuffer = embeddingToBuffer(embArray);
          const result = insertMemory.get(
            body.content.trim(), body.category || "general", "gui", null,
            imp, embBuffer, 1, 1, null, null, 1, body.is_static ? 1 : 0, 0, null, null, 0
          ) as { id: number; created_at: string };
          db.prepare("UPDATE memories SET user_id = ?, tags = ? WHERE id = ?").run(1, tagsJson, result.id);
          const initFSRS2 = fsrsProcessReview(null, FSRSRating.Good, 0);
          const decayScore = calculateDecayScore(imp, result.created_at, 0, null, !!body.is_static, 1, initFSRS2.stability);
          db.prepare("UPDATE memories SET decay_score = ?, fsrs_stability = ?, fsrs_difficulty = ?, fsrs_storage_strength = ?, fsrs_retrieval_strength = ?, fsrs_learning_state = ?, fsrs_reps = ?, fsrs_lapses = ?, fsrs_last_review_at = ? WHERE id = ?").run(
            Math.round(decayScore * 1000) / 1000,
            initFSRS2.stability, initFSRS2.difficulty, initFSRS2.storage_strength,
            initFSRS2.retrieval_strength, initFSRS2.learning_state, initFSRS2.reps, initFSRS2.lapses,
            initFSRS2.last_review_at, result.id
          );
          writeVec(result.id, embArray);
          await autoLink(result.id, embArray);
          return json({ created: true, id: result.id });
        } catch (e: any) {
          return errorResponse(`Failed: ${e.message}`, 500);
        }
      } catch (e: any) { return errorResponse(`Bad request: ${e.message}`, 400); }
    }

    if (url.pathname.match(/^\/gui\/memories\/\d+$/) && method === "PATCH") {
      if (!guiAuthed(req)) return errorResponse("GUI auth required", 401);
      try {
        const id = Number(url.pathname.split("/")[3]);
        const body = await req.json() as any;
        const sets: string[] = [];
        const vals: any[] = [];
        if (body.content !== undefined) { sets.push("content = ?"); vals.push(body.content.trim()); }
        if (body.category !== undefined) { sets.push("category = ?"); vals.push(body.category); }
        if (body.importance !== undefined) { sets.push("importance = ?"); vals.push(Math.max(1, Math.min(10, Number(body.importance)))); }
        if (body.is_static !== undefined) { sets.push("is_static = ?"); vals.push(body.is_static ? 1 : 0); }
        if (sets.length === 0) return errorResponse("Nothing to update");
        sets.push("updated_at = datetime('now')");
        vals.push(id);
        db.prepare(`UPDATE memories SET ${sets.join(", ")} WHERE id = ?`).run(...vals);
        // Re-embed if content changed
        if (body.content !== undefined) {
          try {
            const emb = await embed(body.content.trim());
            updateMemoryEmbedding.run(embeddingToBuffer(emb), id); try { updateMemoryVec.run(embeddingToVectorJSON(emb), id); } catch {}
          } catch {}
        }
        return json({ updated: true, id });
      } catch (e: any) { return errorResponse(`Failed: ${e.message}`, 500); }
    }

    if (url.pathname.match(/^\/gui\/memories\/\d+$/) && method === "DELETE") {
      if (!guiAuthed(req)) return errorResponse("GUI auth required", 401);
      const id = Number(url.pathname.split("/")[3]);
      deleteMemory.run(id);
      return json({ deleted: true, id });
    }

    if (url.pathname === "/gui/memories/bulk-archive" && method === "POST") {
      if (!guiAuthed(req)) return errorResponse("GUI auth required", 401);
      try {
        const body = await req.json() as any;
        const ids = body.ids;
        if (!Array.isArray(ids)) return errorResponse("ids array required");
        let count = 0;
        for (const id of ids) { markArchived.run(id); count++; }
        return json({ archived: count });
      } catch (e: any) { return errorResponse(`Failed: ${e.message}`, 500); }
    }

    // ========================================================================
    // TAGS — v4.1
    // ========================================================================

    if (url.pathname === "/tags" && method === "GET") {
      const rows = getAllTags.all(auth.user_id) as Array<{ tags: string }>;
      const tagSet = new Set<string>();
      for (const row of rows) {
        try {
          const parsed = JSON.parse(row.tags) as string[];
          for (const t of parsed) tagSet.add(t);
        } catch {}
      }
      return json({ tags: Array.from(tagSet).sort() });
    }

    if (url.pathname === "/tags/search" && method === "POST") {
      try {
        const body = await req.json() as any;
        const tag = body.tag?.trim().toLowerCase();
        if (!tag) return errorResponse("tag is required");
        const limit = Math.min(Number(body.limit) || 20, 100);
        const results = getByTag.all(`%"${tag}"%`, auth.user_id, limit) as any[];
        for (const r of results) {
          try { r.tags = JSON.parse(r.tags); } catch { r.tags = []; }
          trackAccessWithFSRS(r.id);
        }
        return json({ results, tag });
      } catch (e: any) {
        return errorResponse(`Tag search failed: ${e.message}`, 500);
      }
    }

    if (url.pathname.match(/^\/memory\/\d+\/tags$/) && method === "PUT") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const id = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        let tags: string[] = [];
        if (Array.isArray(body.tags)) {
          tags = body.tags.map((t: any) => String(t).trim().toLowerCase()).filter(Boolean);
        }
        db.prepare("UPDATE memories SET tags = ?, updated_at = datetime('now') WHERE id = ?")
          .run(JSON.stringify(tags), id);
        return json({ updated: true, id, tags });
      } catch (e: any) {
        return errorResponse(`Failed to update tags: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // EPISODES — v4.1
    // ========================================================================

    if (url.pathname === "/episodes" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        const ep = insertEpisode.get(
          body.title || null, body.session_id || null, body.agent || null, auth.user_id
        ) as { id: number; started_at: string };
        return json({ created: true, id: ep.id, started_at: ep.started_at });
      } catch (e: any) {
        return errorResponse(`Failed to create episode: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/episodes" && method === "GET") {
      const limit = Math.min(Number(url.searchParams.get("limit") || 20), 100);
      const episodes = listEpisodes.all(auth.user_id, limit) as any[];
      return json({ episodes });
    }

    if (/^\/episodes\/\d+$/.test(url.pathname) && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      const episode = getEpisode.get(id) as any;
      if (!episode) return errorResponse("Episode not found", 404);
      const memories = getEpisodeMemories.all(id) as any[];
      for (const m of memories) {
        try { m.tags = JSON.parse(m.tags); } catch { m.tags = []; }
      }
      return json({ ...episode, memories });
    }

    if (/^\/episodes\/\d+$/.test(url.pathname) && method === "PATCH") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const id = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        updateEpisode.run(body.title || null, body.summary || null, body.ended_at || null, id);
        return json({ updated: true, id });
      } catch (e: any) {
        return errorResponse(`Failed to update episode: ${e.message}`, 500);
      }
    }

    if (/^\/episodes\/\d+\/memories$/.test(url.pathname) && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const episodeId = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        const memoryIds = body.memory_ids;
        if (!Array.isArray(memoryIds)) return errorResponse("memory_ids array required");
        let assigned = 0;
        for (const mid of memoryIds) {
          assignToEpisode.run(episodeId, mid);
          assigned++;
        }
        updateEpisode.run(null, null, null, episodeId);
        return json({ assigned, episode_id: episodeId });
      } catch (e: any) {
        return errorResponse(`Failed to assign memories: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // CONSOLIDATION — v4.1
    // ========================================================================

    if (url.pathname === "/consolidate" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json().catch(() => ({})) as any;
        const memoryId = body.memory_id; // optional: consolidate specific cluster

        if (memoryId) {
          const result = await consolidateCluster(memoryId, auth.user_id);
          if (!result) return json({ consolidated: false, reason: "Cluster too small or already consolidated" });
          return json({ consolidated: true, summary_id: result.summaryId, archived: result.archivedCount });
        } else {
          const total = await runConsolidationSweep(auth.user_id);
          return json({ consolidated: total > 0, archived: total });
        }
      } catch (e: any) {
        return errorResponse(`Consolidation failed: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/consolidations" && method === "GET") {
      const rows = db.prepare(
        `SELECT c.id, c.summary_memory_id, c.source_memory_ids, c.cluster_label, c.created_at,
          m.content as summary_content
         FROM consolidations c JOIN memories m ON c.summary_memory_id = m.id
         ORDER BY c.created_at DESC LIMIT 50`
      ).all() as any[];
      for (const r of rows) {
        try { r.source_memory_ids = JSON.parse(r.source_memory_ids); } catch {}
      }
      return json({ consolidations: rows });
    }

    // ========================================================================
    // DECAY — v4.1
    // ========================================================================

    if (url.pathname === "/decay/refresh" && method === "POST") {
      const updated = updateDecayScores();
      return json({ refreshed: updated });
    }

    if (url.pathname === "/decay/scores" && method === "GET") {
      const limit = Math.min(Number(url.searchParams.get("limit") || 20), 100);
      const order = url.searchParams.get("order") === "asc" ? "ASC" : "DESC";
      const rows = db.prepare(
        `SELECT id, content, category, importance, decay_score, access_count, last_accessed_at,
           created_at, is_static, source_count, confidence,
           fsrs_stability, fsrs_difficulty, fsrs_storage_strength, fsrs_retrieval_strength,
           fsrs_learning_state, fsrs_reps, fsrs_lapses, fsrs_last_review_at
         FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1
         ORDER BY COALESCE(decay_score, importance) ${order} LIMIT ?`
      ).all(limit) as any[];
      return json({ memories: rows });
    }

    // FSRS-6 endpoints
    if (url.pathname === "/fsrs/review" && method === "POST") {
      try {
        const body = await req.json() as any;
        const id = Number(body.id);
        const grade = Number(body.grade || 3) as FSRSRating;
        if (!id || grade < 1 || grade > 4) return errorResponse("id required, grade 1-4", 400);
        trackAccessWithFSRS(id, grade);
        const updated = getFSRS.get(id) as any;
        return json({ id, fsrs: updated });
      } catch (e: any) { return errorResponse(e.message, 400); }
    }

    if (url.pathname === "/fsrs/state" && method === "GET") {
      const id = Number(url.searchParams.get("id"));
      if (!id) return errorResponse("id required", 400);
      const row = getFSRS.get(id) as any;
      if (!row) return errorResponse("not found", 404);
      const elapsed = row.fsrs_last_review_at
        ? (Date.now() - new Date(row.fsrs_last_review_at + "Z").getTime()) / 86400000
        : (Date.now() - new Date(row.created_at + "Z").getTime()) / 86400000;
      const retrievability = row.fsrs_stability
        ? fsrsRetrievability(row.fsrs_stability, elapsed)
        : null;
      const nextReview = row.fsrs_stability
        ? fsrsNextInterval(row.fsrs_stability)
        : null;
      return json({ id, retrievability, next_review_days: nextReview, ...row });
    }

    if (url.pathname === "/fsrs/init" && method === "POST") {
      // Backfill FSRS state for all memories that don't have it
      const uninitialized = db.prepare(
        `SELECT id, created_at FROM memories WHERE fsrs_stability IS NULL AND is_forgotten = 0 AND is_latest = 1`
      ).all() as any[];
      let count = 0;
      const batch = db.transaction(() => {
        for (const m of uninitialized) {
          const init = fsrsProcessReview(null, FSRSRating.Good, 0);
          updateFSRS.run(init.stability, init.difficulty, init.storage_strength,
            init.retrieval_strength, init.learning_state, init.reps, init.lapses,
            init.last_review_at, m.id);
          count++;
        }
      });
      batch();
      return json({ initialized: count });
    }

    // ========================================================================
    // CONTEXT WINDOW OPTIMIZER — POST /pack
    // ========================================================================

    if (url.pathname === "/pack" && method === "POST") {
      try {
        const body = await req.json() as any;
        const context = body.context || "";
        const tokenBudget = Math.max(100, Math.min(Number(body.tokens) || 4000, 128000));
        const format = body.format || "text"; // text, json, xml

        // Run recall to get candidate memories
        const candidates: Array<{ content: string; category: string; importance: number; decay_score: number; confidence: number; score: number; source: string; id: number }> = [];

        // Static facts first
        const staticFacts = getStaticMemories.all(auth.user_id) as Array<any>;
        for (const sf of staticFacts) {
          candidates.push({ ...sf, score: 100, source: "static", decay_score: sf.importance, confidence: sf.confidence || 1 });
        }

        // Semantic search
        if (context.trim()) {
          const semantic = await hybridSearch(context, 50, false, true, true);
          for (const sr of semantic) {
            if (!candidates.find(c => c.id === sr.id)) {
              candidates.push({
                id: sr.id, content: sr.content, category: sr.category,
                importance: sr.importance, decay_score: sr.decay_score || sr.importance,
                confidence: 1, score: sr.score * 50, source: "semantic",
              });
            }
          }
        }

        // High importance
        const important = db.prepare(
          `SELECT id, content, category, importance, decay_score, confidence
           FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
           ORDER BY COALESCE(decay_score, importance) DESC LIMIT 30`
        ).all(auth.user_id) as Array<any>;
        for (const m of important) {
          if (!candidates.find(c => c.id === m.id)) {
            candidates.push({ ...m, score: (m.decay_score || m.importance) * 2, source: "important" });
          }
        }

        // Sort by effective score (score * confidence)
        candidates.sort((a, b) => (b.score * (b.confidence || 1)) - (a.score * (a.confidence || 1)));

        // Greedy packing within token budget (~4 chars per token)
        const packed: typeof candidates = [];
        let tokensUsed = 0;
        for (const c of candidates) {
          const memTokens = Math.ceil(c.content.length / 4) + 10; // overhead for formatting
          if (tokensUsed + memTokens > tokenBudget) continue;
          packed.push(c);
          tokensUsed += memTokens;
        }

        // Track access
        for (const p of packed) trackAccessWithFSRS(p.id);

        // Format output
        let output: string;
        if (format === "xml") {
          output = packed.map(p =>
            `<memory id="${p.id}" category="${p.category}" importance="${p.importance}">\n${p.content}\n</memory>`
          ).join("\n");
        } else if (format === "json") {
          output = JSON.stringify(packed.map(p => ({
            id: p.id, content: p.content, category: p.category, importance: p.importance,
          })));
        } else {
          output = packed.map(p => `[${p.category}] ${p.content}`).join("\n\n");
        }

        return json({
          packed: output,
          memories_included: packed.length,
          tokens_estimated: tokensUsed,
          token_budget: tokenBudget,
          utilization: Math.round((tokensUsed / tokenBudget) * 100) + "%",
        });
      } catch (e: any) {
        return errorResponse(`Pack failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // PROMPT TEMPLATE ENGINE — GET /prompt
    // ========================================================================

    if (url.pathname === "/prompt" && method === "GET") {
      try {
        const format = url.searchParams.get("format") || "raw"; // raw, anthropic, openai, llamaindex
        const tokenBudget = Math.max(100, Math.min(Number(url.searchParams.get("tokens") || 4000), 128000));
        const context = url.searchParams.get("context") || "";

        // Use pack logic internally
        const packReq = new Request("http://localhost/pack", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ context, tokens: tokenBudget, format: "text" }),
        });
        // Run pack inline
        const candidates: Array<any> = [];
        const staticFacts = getStaticMemories.all(auth.user_id) as Array<any>;
        for (const sf of staticFacts) candidates.push({ ...sf, score: 100 });
        if (context.trim()) {
          const semantic = await hybridSearch(context, 30, false, true, true);
          for (const sr of semantic) {
            if (!candidates.find((c: any) => c.id === sr.id)) candidates.push({ ...sr, score: sr.score * 50 });
          }
        }
        const important = db.prepare(
          `SELECT id, content, category, importance, decay_score, confidence
           FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
           ORDER BY COALESCE(decay_score, importance) DESC LIMIT 1000`
        ).all(auth.user_id) as Array<any>;
        for (const m of important) {
          if (!candidates.find((c: any) => c.id === m.id)) candidates.push({ ...m, score: (m.decay_score || m.importance) * 2 });
        }
        candidates.sort((a: any, b: any) => b.score - a.score);

        const packed: string[] = [];
        let tokensUsed = 0;
        for (const c of candidates) {
          const t = Math.ceil(c.content.length / 4) + 5;
          if (tokensUsed + t > tokenBudget) continue;
          packed.push(`[${c.category}] ${c.content}`);
          tokensUsed += t;
          trackAccessWithFSRS(c.id);
        }

        const memoryBlock = packed.join("\n\n");

        let prompt: string;
        if (format === "anthropic") {
          prompt = `<context>
<engram-memories count="${packed.length}" tokens="~${tokensUsed}">
${memoryBlock}
</engram-memories>
</context>

The above are persistent memories from previous sessions. Use them to maintain continuity. If a memory contradicts the current conversation, prefer the conversation.`;
        } else if (format === "openai") {
          prompt = `# Persistent Memory (Engram)
The following are ${packed.length} memories from previous sessions (~${tokensUsed} tokens):

${memoryBlock}

Use these memories for context. If they conflict with the current conversation, prefer the conversation.`;
        } else if (format === "llamaindex") {
          prompt = `[MEMORY CONTEXT]
${memoryBlock}
[/MEMORY CONTEXT]`;
        } else {
          prompt = memoryBlock;
        }

        return json({
          prompt,
          format,
          memories_included: packed.length,
          tokens_estimated: tokensUsed,
        });
      } catch (e: any) {
        return errorResponse(`Prompt generation failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // WEBHOOKS — v4.2
    // ========================================================================

    if (url.pathname === "/webhooks" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        if (!body.url) return errorResponse("url is required");
        const events = body.events || ["*"];
        const secret = body.secret || null;
        const result = insertWebhook.get(body.url, JSON.stringify(events), secret, auth.user_id) as { id: number; created_at: string };
        return json({ created: true, id: result.id, url: body.url, events });
      } catch (e: any) {
        return errorResponse(`Failed to create webhook: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/webhooks" && method === "GET") {
      const hooks = listWebhooks.all(auth.user_id) as any[];
      for (const h of hooks) {
        try { h.events = JSON.parse(h.events); } catch {}
      }
      return json({ webhooks: hooks });
    }

    if (url.pathname.match(/^\/webhooks\/\d+$/) && method === "DELETE") {
      const id = Number(url.pathname.split("/")[2]);
      deleteWebhook.run(id, auth.user_id);
      return json({ deleted: true, id });
    }

    // ========================================================================
    // SYNC — v4.2 (Multi-instance replication)
    // ========================================================================

    if (url.pathname === "/sync/changes" && method === "GET") {
      const since = url.searchParams.get("since") || "1970-01-01T00:00:00";
      const limit = Math.min(Number(url.searchParams.get("limit") || 100), 1000);
      const changes = getChangesSince.all(since, auth.user_id, limit) as any[];
      for (const c of changes) {
        try { c.tags = c.tags ? JSON.parse(c.tags) : []; } catch { c.tags = []; }
      }
      return json({
        changes,
        count: changes.length,
        since,
        server_time: new Date().toISOString(),
      });
    }

    if (url.pathname === "/sync/receive" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        const memories = body.memories;
        if (!Array.isArray(memories)) return errorResponse("memories array required");

        let created = 0, updated = 0, skipped = 0;
        for (const mem of memories) {
          if (!mem.sync_id || !mem.content) { skipped++; continue; }

          const existing = getMemoryBySyncId.get(mem.sync_id) as any;
          if (existing) {
            // Conflict resolution: last-write-wins
            if (mem.updated_at > existing.updated_at) {
              db.prepare(
                `UPDATE memories SET content = ?, category = ?, importance = ?, tags = ?,
                 confidence = ?, is_static = ?, is_forgotten = ?, is_archived = ?,
                 updated_at = ? WHERE id = ?`
              ).run(
                mem.content, mem.category || "general", mem.importance || 5,
                mem.tags ? JSON.stringify(mem.tags) : null,
                mem.confidence ?? 1.0, mem.is_static ? 1 : 0,
                mem.is_forgotten ? 1 : 0, mem.is_archived ? 1 : 0,
                mem.updated_at, existing.id
              );
              // Re-embed on content change
              try {
                const emb = await embed(mem.content);
                updateMemoryEmbedding.run(embeddingToBuffer(emb), existing.id); try { updateMemoryVec.run(embeddingToVectorJSON(emb), existing.id); } catch {}
              } catch {}
              updated++;
            } else {
              skipped++;
            }
          } else {
            // New memory from remote
            let embBuffer: Buffer | null = null;
            try { embBuffer = embeddingToBuffer(await embed(mem.content)); } catch {}
            const result = insertMemory.get(
              mem.content, mem.category || "general", mem.source || "sync", mem.session_id || null,
              mem.importance || 5, embBuffer, mem.version || 1, 1, null, null, 1,
              mem.is_static ? 1 : 0, mem.is_forgotten ? 1 : 0, null, null, 0
            ) as { id: number; created_at: string };
            db.prepare(
              "UPDATE memories SET user_id = ?, sync_id = ?, tags = ?, confidence = ?, is_archived = ? WHERE id = ?"
            ).run(
              auth.user_id, mem.sync_id, mem.tags ? JSON.stringify(mem.tags) : null,
              mem.confidence ?? 1.0, mem.is_archived ? 1 : 0, result.id
            );
            if (embBuffer) {
              const embArray = await embed(mem.content);
              writeVec(result.id, embArray);
              await autoLink(result.id, embArray);
            }
            created++;
          }
        }

        return json({ synced: true, created, updated, skipped });
      } catch (e: any) {
        return errorResponse(`Sync receive failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // DERIVE — Infer new facts from memory clusters
    // ========================================================================

    if (url.pathname === "/derive" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      if (!LLM_API_KEY) return errorResponse("LLM not configured — /derive requires inference", 400);
      try {
        const body = await req.json() as any;
        const context = body.context || "";
        const limit = Math.min(Number(body.limit || 30), 100);
        const minCluster = Number(body.min_cluster || 3);

        // Gather candidate memories to derive from
        let candidates: any[];
        if (context.trim()) {
          candidates = await hybridSearch(context, limit, false, true, true);
        } else {
          candidates = db.prepare(
            `SELECT id, content, category, importance, tags, created_at
             FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
             ORDER BY COALESCE(decay_score, importance) DESC LIMIT ?`
          ).all(auth.user_id, limit) as any[];
        }

        if (candidates.length < minCluster) {
          return json({ derived: 0, message: `Need at least ${minCluster} memories, found ${candidates.length}` });
        }

        // Format memories for LLM inference
        const memoryList = candidates.map((c, i) =>
          `[${c.id}] (${c.category}) ${c.content}`
        ).join("\n");

        const derivePrompt = `You are an inference engine for a memory system. Given a collection of memories, identify patterns, connections, and inferences that are NOT explicitly stated but can be logically derived.

Rules:
- Only derive facts that are NOT already stored — don't repeat existing memories
- Each derived fact must cite which memory IDs it was inferred from (source_ids)
- Confidence should reflect how certain the inference is (0.3-0.9, never 1.0)
- Prefer actionable insights over trivial observations
- Maximum 5 derived facts per batch

Return JSON:
{
  "derived": [
    {
      "content": "inferred fact",
      "category": "discovery",
      "importance": 6,
      "confidence": 0.7,
      "source_ids": [123, 456],
      "reasoning": "brief explanation of the inference"
    }
  ]
}

If no meaningful inferences, return {"derived": []}`;

        const resp = await callLLM(derivePrompt, `Here are the memories:\n\n${memoryList}`);
        if (!resp) return json({ derived: 0, facts: [] });

        let parsed: { derived: Array<any> };
        try {
          const cleaned = resp.replace(/```json\n?|\n?```/g, "").trim();
          try {
            parsed = JSON.parse(cleaned);
          } catch {
            const jsonMatch = cleaned.match(/\{[\s\S]*"derived"[\s\S]*\}/);
            if (jsonMatch) {
              parsed = JSON.parse(jsonMatch[0]);
            } else {
              console.error("Derive: unparseable LLM response:", cleaned.substring(0, 500));
              return json({ derived: 0, facts: [], error: "LLM returned unparseable response" });
            }
          }
        } catch {
          return json({ derived: 0, facts: [], error: "LLM returned unparseable response" });
        }

        if (!parsed.derived?.length) return json({ derived: 0, facts: [] });

        const stored: Array<{ id: number; content: string; confidence: number; source_ids: number[] }> = [];
        for (const d of parsed.derived) {
          if (!d.content?.trim()) continue;

          let embBuffer: Buffer | null = null;
          let embArray: Float32Array | null = null;
          try {
            embArray = await embed(d.content.trim());
            embBuffer = embeddingToBuffer(embArray);
          } catch {}

          const result = insertMemory.get(
            d.content.trim(), d.category || "discovery", "derived", null,
            d.importance || 5, embBuffer, 1, 1, null, null, 1, 0, 0, null, null, 0
          ) as { id: number; created_at: string };

          const syncId = crypto.randomUUID();
          db.prepare(
            "UPDATE memories SET user_id = ?, sync_id = ?, confidence = ?, tags = ? WHERE id = ?"
          ).run(auth.user_id, syncId, d.confidence || 0.7, JSON.stringify(["derived", ...(d.tags || [])]), result.id);

          // Link to source memories
          if (d.source_ids && Array.isArray(d.source_ids)) {
            for (const srcId of d.source_ids) {
              try { insertLink.run(result.id, srcId, d.confidence || 0.7, "derived_from"); } catch {}
            }
          }

          if (embArray) { writeVec(result.id, embArray); await autoLink(result.id, embArray); }

          emitWebhookEvent("memory.derived", {
            id: result.id, content: d.content.trim(), confidence: d.confidence,
            source_ids: d.source_ids, reasoning: d.reasoning,
          }, auth.user_id);

          stored.push({ id: result.id, content: d.content.trim(), confidence: d.confidence || 0.7, source_ids: d.source_ids || [] });
        }

        return json({ derived: stored.length, facts: stored });
      } catch (e: any) {
        return errorResponse(`Derive failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // MEM0 IMPORT — v4.2
    // ========================================================================

    if (url.pathname === "/import/mem0" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        const memories = body.memories || body.results || body;
        if (!Array.isArray(memories)) return errorResponse("Expected array of mem0 memories");

        let imported = 0;
        for (const mem of memories) {
          // Mem0 format: { id, memory/text/content, metadata?, created_at?, updated_at?, user_id? }
          const content = mem.memory || mem.text || mem.content;
          if (!content) continue;

          const category = mem.metadata?.category || mem.category || "general";
          const source = mem.metadata?.source || mem.source || "mem0-import";
          const importance = mem.metadata?.importance || 5;
          const tags = mem.metadata?.tags || ["mem0-import"];

          let embBuffer: Buffer | null = null;
          let embArray: Float32Array | null = null;
          try {
            embArray = await embed(content.trim());
            embBuffer = embeddingToBuffer(embArray);
          } catch {}

          const result = insertMemory.get(
            content.trim(), category, source, null, importance, embBuffer,
            1, 1, null, null, 1, 0, 0, null, null, 0
          ) as { id: number; created_at: string };

          db.prepare(
            "UPDATE memories SET user_id = ?, tags = ?, sync_id = ?, confidence = 1.0 WHERE id = ?"
          ).run(auth.user_id, JSON.stringify(tags), crypto.randomUUID(), result.id);

          if (embArray) { writeVec(result.id, embArray); await autoLink(result.id, embArray); }
          imported++;
        }

        return json({ imported, source: "mem0" });
      } catch (e: any) {
        return errorResponse(`Mem0 import failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // SUPERMEMORY IMPORT — v4.2
    // ========================================================================

    if (url.pathname === "/import/supermemory" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        // Supermemory formats:
        //   v1 API: { documents: [{ content, spaces?, type?, createdAt?, metadata? }] }
        //   Export: { memories: [{ content/text, space?, tags?, ...}] }
        //   Raw array: [{ content, ... }]
        const items = body.documents || body.memories || body.data || (Array.isArray(body) ? body : null);
        if (!items || !Array.isArray(items)) {
          return errorResponse("Expected documents/memories array. Accepted shapes: { documents: [...] }, { memories: [...] }, or raw array");
        }

        let imported = 0, skipped = 0;
        for (const item of items) {
          const content = item.content || item.text || item.description || item.raw;
          if (!content?.trim()) { skipped++; continue; }

          // Map supermemory type → engram category
          const typeMap: Record<string, string> = {
            note: "general", tweet: "discovery", page: "discovery",
            document: "task", bookmark: "discovery", conversation: "state",
          };
          const category = item.category
            || typeMap[item.type?.toLowerCase()]
            || (item.space?.toLowerCase() === "work" ? "task" : null)
            || "general";

          // Supermemory spaces → tags
          const tags: string[] = ["supermemory-import"];
          if (item.spaces && Array.isArray(item.spaces)) {
            for (const s of item.spaces) tags.push(String(s).toLowerCase());
          } else if (item.space) {
            tags.push(String(item.space).toLowerCase());
          }
          if (item.tags && Array.isArray(item.tags)) {
            for (const t of item.tags) tags.push(String(t).toLowerCase());
          }
          if (item.type) tags.push(item.type.toLowerCase());

          const importance = item.importance || item.metadata?.importance || 5;
          const source = item.source || item.metadata?.source || "supermemory-import";

          let embBuffer: Buffer | null = null;
          let embArray: Float32Array | null = null;
          try {
            embArray = await embed(content.trim());
            embBuffer = embeddingToBuffer(embArray);
          } catch {}

          const result = insertMemory.get(
            content.trim(), category, source, null, importance, embBuffer,
            1, 1, null, null, 1, 0, 0, null, null, 0
          ) as { id: number; created_at: string };

          db.prepare(
            "UPDATE memories SET user_id = ?, tags = ?, sync_id = ?, confidence = 1.0 WHERE id = ?"
          ).run(auth.user_id, JSON.stringify([...new Set(tags)]), crypto.randomUUID(), result.id);

          if (embArray) { writeVec(result.id, embArray); await autoLink(result.id, embArray); }
          imported++;
        }

        return json({ imported, skipped, source: "supermemory" });
      } catch (e: any) {
        return errorResponse(`Supermemory import failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // MEMORY GRAPH — v4.3
    // ========================================================================

    if (url.pathname === "/graph" && method === "GET") {
      try {
        const center = url.searchParams.get("center"); // memory ID to center on
        const depth = Math.min(Number(url.searchParams.get("depth") || 2), 4);
        const maxNodes = Math.min(Number(url.searchParams.get("max") || 1000), 2000);
        const includeEntities = url.searchParams.get("entities") !== "0";
        const context = url.searchParams.get("q");

        type GNode = { id: string; label: string; type: string; category?: string; importance?: number; confidence?: number; group?: string; size?: number };
        type GEdge = { source: string; target: string; type: string; weight: number };

        const nodes: Map<string, GNode> = new Map();
        const edges: GEdge[] = [];
        const visited = new Set<number>();

        // BFS from center or top memories
        const queue: Array<{ id: number; currentDepth: number }> = [];

        if (center) {
          queue.push({ id: Number(center), currentDepth: 0 });
        } else if (context) {
          // Semantic search to find starting nodes
          const results = await hybridSearch(context, 10, false, true, true);
          for (const r of results) queue.push({ id: r.id, currentDepth: 0 });
        } else {
          // Top memories by decay score
          const top = db.prepare(
            `SELECT id FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
             ORDER BY COALESCE(decay_score, importance) DESC LIMIT 1000`
          ).all(auth.user_id) as any[];
          for (const t of top) queue.push({ id: t.id, currentDepth: 0 });
        }

        while (queue.length > 0 && nodes.size < maxNodes) {
          const { id, currentDepth } = queue.shift()!;
          if (visited.has(id)) continue;
          visited.add(id);

          const mem = getMemoryWithoutEmbedding.get(id) as any;
          if (!mem || mem.is_forgotten) continue;

          const nodeId = `m${id}`;
          nodes.set(nodeId, {
            id: nodeId,
            label: mem.content.substring(0, 60) + (mem.content.length > 60 ? "…" : ""),
            type: "memory",
            category: mem.category,
            importance: mem.importance,
            confidence: mem.confidence,
            group: mem.category,
            size: Math.max(3, mem.importance * 1.5),
            source: mem.source,
            created_at: mem.created_at,
            is_static: mem.is_static,
            is_forgotten: mem.is_forgotten,
            is_archived: mem.is_archived,
            parent_memory_id: mem.parent_memory_id,
            source_count: mem.source_count,
            content: mem.content,
            version: mem.version,
            tags: mem.tags,
            forget_after: mem.forget_after,
          } as any);

          // Get links
          if (currentDepth < depth) {
            const links = db.prepare(
              `SELECT ml.source_id, ml.target_id, ml.similarity, ml.type,
                m.id as linked_id, m.content, m.category, m.importance, m.confidence
               FROM memory_links ml
               JOIN memories m ON m.id = CASE WHEN ml.source_id = ? THEN ml.target_id ELSE ml.source_id END
               WHERE (ml.source_id = ? OR ml.target_id = ?) AND m.is_forgotten = 0`
            ).all(id, id, id) as any[];

            for (const link of links) {
              const linkedNodeId = `m${link.linked_id}`;
              edges.push({
                source: nodeId,
                target: linkedNodeId,
                type: link.type || "related",
                weight: link.similarity,
              });
              if (!visited.has(link.linked_id)) {
                queue.push({ id: link.linked_id, currentDepth: currentDepth + 1 });
              }
            }
          }

          // Entity connections
          if (includeEntities) {
            const memEntities = db.prepare(
              `SELECT e.id, e.name, e.type FROM entities e
               JOIN memory_entities me ON me.entity_id = e.id WHERE me.memory_id = ?`
            ).all(id) as any[];
            for (const ent of memEntities) {
              const entNodeId = `e${ent.id}`;
              if (!nodes.has(entNodeId)) {
                nodes.set(entNodeId, {
                  id: entNodeId,
                  label: ent.name,
                  type: "entity",
                  group: ent.type,
                  size: 8,
                });
              }
              edges.push({ source: nodeId, target: entNodeId, type: "about", weight: 1.0 });
            }
          }
        }

        // Add entity-to-entity relationships
        if (includeEntities) {
          const entityIds = [...nodes.entries()].filter(([k]) => k.startsWith("e")).map(([k]) => Number(k.slice(1)));
          if (entityIds.length > 0) {
            const placeholders = entityIds.map(() => "?").join(",");
            const rels = db.prepare(
              `SELECT source_entity_id, target_entity_id, relationship FROM entity_relationships
               WHERE source_entity_id IN (${placeholders}) OR target_entity_id IN (${placeholders})`
            ).all(...entityIds, ...entityIds) as any[];
            for (const r of rels) {
              edges.push({
                source: `e${r.source_entity_id}`,
                target: `e${r.target_entity_id}`,
                type: r.relationship,
                weight: 0.9,
              });
            }
          }
        }

        // Add project clusters
        const projectNodes = db.prepare(
          `SELECT DISTINCT p.id, p.name, p.status FROM projects p
           JOIN memory_projects mp ON mp.project_id = p.id
           JOIN memories m ON m.id = mp.memory_id
           WHERE p.user_id = ? AND m.is_forgotten = 0`
        ).all(auth.user_id) as any[];
        for (const proj of projectNodes) {
          const projNodeId = `p${proj.id}`;
          nodes.set(projNodeId, {
            id: projNodeId,
            label: proj.name,
            type: "project",
            group: "project",
            size: 10,
          });
          // Link project to its memories that are in the graph
          const projMems = db.prepare(
            "SELECT memory_id FROM memory_projects WHERE project_id = ?"
          ).all(proj.id) as any[];
          for (const pm of projMems) {
            if (nodes.has(`m${pm.memory_id}`)) {
              edges.push({ source: projNodeId, target: `m${pm.memory_id}`, type: "contains", weight: 0.8 });
            }
          }
        }

        return json({
          nodes: [...nodes.values()],
          edges: edges,
          links: edges.slice(),
          node_count: nodes.size,
          edge_count: edges.length,
        });
      } catch (e: any) {
        return errorResponse(`Graph failed: ${e.message}`, 500);
      }
    }

    // Graph visualization page
    if (url.pathname === "/graph/view" && method === "GET") {
      const graphHtml = await Bun.file(resolve(import.meta.dir, "engram-graph.html")).text().catch(() => null);
      if (!graphHtml) return errorResponse("Graph view not found. Place engram-graph.html alongside server.ts", 404);
      return new Response(graphHtml, { headers: { "Content-Type": "text/html" } });
    }

    // ========================================================================
    // ENTITIES — v4.3
    // ========================================================================

    // Create entity
    if (url.pathname === "/entities" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        if (!body.name?.trim()) return errorResponse("name is required");
        const validTypes = ["person", "organization", "team", "device", "product", "service", "generic"];
        const type = validTypes.includes(body.type) ? body.type : "generic";
        const result = insertEntity.get(
          body.name.trim(), type, body.description || null,
          body.aka || null, body.metadata ? JSON.stringify(body.metadata) : null,
          auth.user_id
        ) as { id: number; created_at: string };
        return json({ created: true, id: result.id, name: body.name.trim(), type, created_at: result.created_at });
      } catch (e: any) {
        return errorResponse(`Failed to create entity: ${e.message}`, 500);
      }
    }

    // List entities
    if (url.pathname === "/entities" && method === "GET") {
      const type = url.searchParams.get("type");
      const q = url.searchParams.get("q");
      let entities: any[];
      if (q) {
        const like = `%${q}%`;
        entities = searchEntities.all(auth.user_id, like, like, like, 100) as any[];
      } else if (type) {
        entities = listEntitiesByType.all(auth.user_id, type) as any[];
      } else {
        entities = listEntities.all(auth.user_id) as any[];
      }
      for (const e of entities) {
        try { if (e.metadata) e.metadata = JSON.parse(e.metadata); } catch {}
      }
      return json({ entities, count: entities.length });
    }

    // Get single entity with details
    if (url.pathname.match(/^\/entities\/\d+$/) && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      const entity = getEntity.get(id) as any;
      if (!entity) return errorResponse("Entity not found", 404);
      try { if (entity.metadata) entity.metadata = JSON.parse(entity.metadata); } catch {}
      entity.memory_ids = entity.memory_ids ? entity.memory_ids.split(",").map(Number) : [];
      entity.relationships = getEntityRelationships.all(id, id, id, id, id) as any[];
      const limit = Math.min(Number(url.searchParams.get("limit") || 20), 100);
      entity.memories = getEntityMemories.all(id, limit) as any[];
      for (const m of entity.memories) {
        try { if (m.tags) m.tags = JSON.parse(m.tags); } catch { m.tags = []; }
      }
      return json(entity);
    }

    // Update entity
    if (url.pathname.match(/^\/entities\/\d+$/) && method === "PUT") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const id = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        updateEntity.run(
          body.name || null, body.type || null, body.description || null,
          body.aka || null, body.metadata ? JSON.stringify(body.metadata) : null,
          id, auth.user_id
        );
        return json({ updated: true, id });
      } catch (e: any) {
        return errorResponse(`Update failed: ${e.message}`, 500);
      }
    }

    // Delete entity
    if (url.pathname.match(/^\/entities\/\d+$/) && method === "DELETE") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const id = Number(url.pathname.split("/")[2]);
      deleteEntity.run(id, auth.user_id);
      return json({ deleted: true, id });
    }

    // Link memory ↔ entity
    if (url.pathname.match(/^\/entities\/\d+\/memories\/\d+$/) && method === "PUT") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const parts = url.pathname.split("/");
      const entityId = Number(parts[2]);
      const memoryId = Number(parts[4]);
      linkMemoryEntity.run(memoryId, entityId);
      return json({ linked: true, entity_id: entityId, memory_id: memoryId });
    }

    // Unlink memory ↔ entity
    if (url.pathname.match(/^\/entities\/\d+\/memories\/\d+$/) && method === "DELETE") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const parts = url.pathname.split("/");
      const entityId = Number(parts[2]);
      const memoryId = Number(parts[4]);
      unlinkMemoryEntity.run(memoryId, entityId);
      return json({ unlinked: true, entity_id: entityId, memory_id: memoryId });
    }

    // Entity relationships
    if (url.pathname.match(/^\/entities\/\d+\/relationships$/) && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const entityId = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        if (!body.target_id || !body.relationship) return errorResponse("target_id and relationship required");
        insertEntityRelationship.run(entityId, body.target_id, body.relationship);
        return json({ linked: true, source: entityId, target: body.target_id, relationship: body.relationship });
      } catch (e: any) {
        return errorResponse(`Relationship failed: ${e.message}`, 500);
      }
    }

    if (url.pathname.match(/^\/entities\/\d+\/relationships$/) && method === "DELETE") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const entityId = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        if (!body.target_id || !body.relationship) return errorResponse("target_id and relationship required");
        deleteEntityRelationship.run(entityId, body.target_id, body.relationship);
        return json({ unlinked: true, source: entityId, target: body.target_id, relationship: body.relationship });
      } catch (e: any) {
        return errorResponse(`Unlink failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // PROJECTS — v4.3
    // ========================================================================

    // Create project
    if (url.pathname === "/projects" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        if (!body.name?.trim()) return errorResponse("name is required");
        const validStatuses = ["active", "paused", "completed", "archived"];
        const status = validStatuses.includes(body.status) ? body.status : "active";
        const result = insertProject.get(
          body.name.trim(), body.description || null, status,
          body.metadata ? JSON.stringify(body.metadata) : null, auth.user_id
        ) as { id: number; created_at: string };
        return json({ created: true, id: result.id, name: body.name.trim(), status, created_at: result.created_at });
      } catch (e: any) {
        return errorResponse(`Failed to create project: ${e.message}`, 500);
      }
    }

    // List projects
    if (url.pathname === "/projects" && method === "GET") {
      const status = url.searchParams.get("status");
      const projects = status
        ? listProjectsByStatus.all(auth.user_id, status) as any[]
        : listProjects.all(auth.user_id) as any[];
      for (const p of projects) {
        try { if (p.metadata) p.metadata = JSON.parse(p.metadata); } catch {}
      }
      return json({ projects, count: projects.length });
    }

    // Get single project
    if (url.pathname.match(/^\/projects\/\d+$/) && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      const project = getProject.get(id) as any;
      if (!project) return errorResponse("Project not found", 404);
      try { if (project.metadata) project.metadata = JSON.parse(project.metadata); } catch {}
      project.memory_ids = project.memory_ids ? project.memory_ids.split(",").map(Number) : [];
      const limit = Math.min(Number(url.searchParams.get("limit") || 50), 200);
      project.memories = getProjectMemories.all(id, limit) as any[];
      for (const m of project.memories) {
        try { if (m.tags) m.tags = JSON.parse(m.tags); } catch { m.tags = []; }
      }
      return json(project);
    }

    // Update project
    if (url.pathname.match(/^\/projects\/\d+$/) && method === "PUT") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const id = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        updateProject.run(
          body.name || null, body.description || null,
          body.status || null, body.metadata ? JSON.stringify(body.metadata) : null,
          id, auth.user_id
        );
        return json({ updated: true, id });
      } catch (e: any) {
        return errorResponse(`Update failed: ${e.message}`, 500);
      }
    }

    // Delete project
    if (url.pathname.match(/^\/projects\/\d+$/) && method === "DELETE") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const id = Number(url.pathname.split("/")[2]);
      deleteProject.run(id, auth.user_id);
      return json({ deleted: true, id });
    }

    // Link memory ↔ project
    if (url.pathname.match(/^\/projects\/\d+\/memories\/\d+$/) && method === "PUT") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const parts = url.pathname.split("/");
      const projectId = Number(parts[2]);
      const memoryId = Number(parts[4]);
      linkMemoryProject.run(memoryId, projectId);
      return json({ linked: true, project_id: projectId, memory_id: memoryId });
    }

    // Unlink memory ↔ project
    if (url.pathname.match(/^\/projects\/\d+\/memories\/\d+$/) && method === "DELETE") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const parts = url.pathname.split("/");
      const projectId = Number(parts[2]);
      const memoryId = Number(parts[4]);
      unlinkMemoryProject.run(memoryId, projectId);
      return json({ unlinked: true, project_id: projectId, memory_id: memoryId });
    }

    // Scoped search — search memories within a project
    if (url.pathname.match(/^\/projects\/\d+\/search$/) && method === "POST") {
      try {
        const projectId = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        const query = body.query;
        if (!query) return errorResponse("query is required");
        const limit = Math.min(Number(body.limit || 20), 100);

        // Get all memory IDs in this project
        const projectMemIds = (db.prepare(
          "SELECT memory_id FROM memory_projects WHERE project_id = ?"
        ).all(projectId) as any[]).map(r => r.memory_id);

        if (projectMemIds.length === 0) return json({ results: [], count: 0, project_id: projectId });

        // Run normal search then filter to project scope
        const allResults = await hybridSearch(query, limit * 3, false, true, true);
        const scoped = allResults.filter(r => projectMemIds.includes(r.id)).slice(0, limit);
        for (const r of scoped) trackAccessWithFSRS(r.id);
        return json({ results: scoped, count: scoped.length, project_id: projectId });
      } catch (e: any) {
        return errorResponse(`Project search failed: ${e.message}`, 500);
      }
    }

    // Entity-scoped search
    if (url.pathname.match(/^\/entities\/\d+\/search$/) && method === "POST") {
      try {
        const entityId = Number(url.pathname.split("/")[2]);
        const body = await req.json() as any;
        const query = body.query;
        if (!query) return errorResponse("query is required");
        const limit = Math.min(Number(body.limit || 20), 100);

        const entityMemIds = (db.prepare(
          "SELECT memory_id FROM memory_entities WHERE entity_id = ?"
        ).all(entityId) as any[]).map(r => r.memory_id);

        if (entityMemIds.length === 0) return json({ results: [], count: 0, entity_id: entityId });

        const allResults = await hybridSearch(query, limit * 3, false, true, true);
        const scoped = allResults.filter(r => entityMemIds.includes(r.id)).slice(0, limit);
        for (const r of scoped) trackAccessWithFSRS(r.id);
        return json({ results: scoped, count: scoped.length, entity_id: entityId });
      } catch (e: any) {
        return errorResponse(`Entity search failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // STATS — v4
    // ========================================================================

    if (url.pathname === "/stats" && method === "GET") {
      const memCount = db.prepare("SELECT COUNT(*) as count FROM memories").get() as { count: number };
      const embCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE embedding IS NOT NULL").get() as { count: number };
      const linkCount = db.prepare("SELECT COUNT(*) as count FROM memory_links").get() as { count: number };
      const convCount = db.prepare("SELECT COUNT(*) as count FROM conversations").get() as { count: number };
      const msgCount = db.prepare("SELECT COUNT(*) as count FROM messages").get() as { count: number };
      const forgottenCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE is_forgotten = 1").get() as { count: number };
      const staticCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE is_static = 1 AND is_forgotten = 0").get() as { count: number };
      const dynamicCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE is_static = 0 AND is_forgotten = 0").get() as { count: number };
      const versionedCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE version > 1").get() as { count: number };
      const archivedCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE is_archived = 1 AND is_forgotten = 0").get() as { count: number };
      const pendingCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE status = 'pending' AND is_forgotten = 0").get() as { count: number };
      const rejectedCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE status = 'rejected'").get() as { count: number };
      const inferenceCount = db.prepare("SELECT COUNT(*) as count FROM memories WHERE is_inference = 1").get() as { count: number };

      const linkTypes = db.prepare(
        `SELECT type, COUNT(*) as count FROM memory_links GROUP BY type ORDER BY count DESC`
      ).all();

      const categories = db.prepare(
        `SELECT category, COUNT(*) as count FROM memories WHERE is_forgotten = 0 GROUP BY category ORDER BY count DESC`
      ).all();

      const agents = db.prepare(
        `SELECT agent, COUNT(*) as conversations,
          (SELECT COUNT(*) FROM messages m JOIN conversations c2 ON m.conversation_id = c2.id WHERE c2.agent = c.agent) as total_messages
         FROM conversations c GROUP BY agent ORDER BY total_messages DESC`
      ).all();

      const dbSize = Bun.file(DB_PATH).size;
      return json({
        memories: {
          total: memCount.count,
          embedded: embCount.count,
          forgotten: forgottenCount.count,
          archived: archivedCount.count,
          pending: pendingCount.count,
          rejected: rejectedCount.count,
          static: staticCount.count,
          dynamic: dynamicCount.count,
          versioned: versionedCount.count,
          inferences: inferenceCount.count,
          categories,
        },
        links: {
          total: linkCount.count,
          by_type: linkTypes,
        },
        conversations: convCount.count,
        messages: msgCount.count,
        agents,
        embedding_model: EMBEDDING_MODEL,
        llm_model: LLM_MODEL,
        llm_configured: !!LLM_API_KEY,
        db_size_mb: Math.round(dbSize / 1048576 * 100) / 100,
        db_path: DB_PATH,
      });
    }

    // ========================================================================
    // INBOX / REVIEW QUEUE — v5.1
    // ========================================================================

    // List pending memories
    if (url.pathname === "/inbox" && method === "GET") {
      const limit = Math.min(Number(url.searchParams.get("limit") || 50), 200);
      const offset = Number(url.searchParams.get("offset") || 0);
      const pending = listPending.all(auth.user_id, limit, offset) as any[];
      const total = (countPending.get(auth.user_id) as { count: number }).count;
      for (const p of pending) {
        try { if (p.tags) p.tags = JSON.parse(p.tags); } catch { p.tags = []; }
      }
      return json({ pending, count: pending.length, total, offset, limit });
    }

    // Approve a pending memory
    if (url.pathname.match(/^\/inbox\/\d+\/approve$/) && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const id = Number(url.pathname.split("/")[2]);
      const mem = getMemoryWithoutEmbedding.get(id) as any;
      if (!mem) return errorResponse("Not found", 404);
      if (mem.status !== "pending") return errorResponse(`Memory is already ${mem.status}`, 400);
      approveMemory.run(id, auth.user_id);
      emitWebhookEvent("memory.approved", { id }, auth.user_id);
      return json({ approved: true, id });
    }

    // Reject a pending memory
    if (url.pathname.match(/^\/inbox\/\d+\/reject$/) && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      const id = Number(url.pathname.split("/")[2]);
      const mem = getMemoryWithoutEmbedding.get(id) as any;
      if (!mem) return errorResponse("Not found", 404);
      if (mem.status !== "pending") return errorResponse(`Memory is already ${mem.status}`, 400);
      const body = await req.json().catch(() => ({})) as any;
      rejectMemory.run(id, auth.user_id);
      if (body.reason) {
        db.prepare("UPDATE memories SET forget_reason = ? WHERE id = ?").run(body.reason, id);
      }
      emitWebhookEvent("memory.rejected", { id, reason: body.reason || null }, auth.user_id);
      return json({ rejected: true, id });
    }

    // Edit + approve in one shot
    if (url.pathname.match(/^\/inbox\/\d+\/edit$/) && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const id = Number(url.pathname.split("/")[2]);
        const mem = getMemoryWithoutEmbedding.get(id) as any;
        if (!mem) return errorResponse("Not found", 404);
        const body = await req.json() as any;

        const sets: string[] = ["status = 'approved'", "updated_at = datetime('now')"];
        const vals: any[] = [];
        if (body.content?.trim()) { sets.push("content = ?"); vals.push(body.content.trim()); }
        if (body.category) { sets.push("category = ?"); vals.push(body.category); }
        if (body.importance) { sets.push("importance = ?"); vals.push(Math.max(1, Math.min(10, Number(body.importance)))); }
        if (body.tags) {
          const tags = Array.isArray(body.tags) ? body.tags : body.tags.split(",");
          sets.push("tags = ?");
          vals.push(JSON.stringify(tags.map((t: any) => String(t).trim().toLowerCase()).filter(Boolean)));
        }
        vals.push(id);
        db.prepare(`UPDATE memories SET ${sets.join(", ")} WHERE id = ?`).run(...vals);

        // Re-embed if content changed
        if (body.content?.trim()) {
          try {
            const emb = await embed(body.content.trim());
            updateMemoryEmbedding.run(embeddingToBuffer(emb), id);
            try { updateMemoryVec.run(embeddingToVectorJSON(emb), id); } catch {}
          } catch {}
        }

        emitWebhookEvent("memory.approved", { id, edited: true }, auth.user_id);
        return json({ approved: true, edited: true, id });
      } catch (e: any) {
        return errorResponse(`Edit failed: ${e.message}`, 500);
      }
    }

    // Bulk approve/reject
    if (url.pathname === "/inbox/bulk" && method === "POST") {
      if (!hasScope(auth, "write")) return errorResponse("Write scope required", 403);
      try {
        const body = await req.json() as any;
        const ids = body.ids;
        const action = body.action; // "approve" or "reject"
        if (!Array.isArray(ids) || !ids.length) return errorResponse("ids array required");
        if (action !== "approve" && action !== "reject") return errorResponse("action must be 'approve' or 'reject'");

        let count = 0;
        const stmt = action === "approve" ? approveMemory : rejectMemory;
        for (const id of ids) {
          stmt.run(id, auth.user_id);
          count++;
        }
        emitWebhookEvent(`memory.bulk_${action}`, { ids, count }, auth.user_id);
        return json({ action, count, ids });
      } catch (e: any) {
        return errorResponse(`Bulk ${(await req.json().catch(() => ({}))).action || "action"} failed: ${e.message}`, 500);
      }
    }

        return errorResponse("Not found", 404);
  },
});

// ============================================================================
// STARTUP
// ============================================================================

const noEmb = (countNoEmbedding.get() as { count: number }).count;
if (noEmb > 0) {
  console.log(`Found ${noEmb} memories without embeddings, backfilling...`);
  backfillEmbeddings(200).then((n) => {
    console.log(`Backfilled ${n} memories. Remaining: ${noEmb - n}`);
  }).catch(e => console.error("Backfill error:", e));
}

// Auto-forget sweep timer
setInterval(() => {
  const swept = sweepExpiredMemories();
  if (swept > 0) console.log(`Auto-forget sweep: ${swept} memories forgotten`);
}, FORGET_SWEEP_INTERVAL);

// Decay score refresh (every 15 minutes)
setInterval(() => {
  const updated = updateDecayScores();
  if (updated > 0) console.log(`Decay refresh: ${updated} scores updated`);
}, 15 * 60 * 1000);

// Auto-consolidation sweep (if LLM configured)
if (LLM_API_KEY) {
  setInterval(async () => {
    try {
      const consolidated = await runConsolidationSweep();
      if (consolidated > 0) console.log(`Auto-consolidation: ${consolidated} memories consolidated`);
    } catch (e: any) {
      console.error("Auto-consolidation error:", e.message);
    }
  }, CONSOLIDATION_INTERVAL);
}

sweepExpiredMemories();
updateDecayScores(); // Initial decay score calculation

// Migrate BLOB embeddings → FLOAT32 vector column (one-time, idempotent)
{
  const unmigrated = db.prepare(
    "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND embedding_vec IS NULL"
  ).all() as Array<{ id: number; embedding: ArrayBuffer | Buffer }>;
  if (unmigrated.length > 0) {
    let migrated = 0;
    const batch = db.transaction(() => {
      for (const m of unmigrated) {
        try {
          // .all() returns ArrayBuffer, .get() returns Buffer — handle both
          const ab = m.embedding instanceof ArrayBuffer ? m.embedding
            : m.embedding.buffer.slice(m.embedding.byteOffset, m.embedding.byteOffset + m.embedding.byteLength);
          const emb = new Float32Array(ab);
          const vecJson = embeddingToVectorJSON(emb);
          updateMemoryVec.run(vecJson, m.id);
          migrated++;
        } catch {}
      }
    });
    batch();
    console.log(`Vector migration: ${migrated}/${unmigrated.length} embeddings → FLOAT32`);
  }
}

// Digest scheduler — check every 5 minutes for due digests
setInterval(async () => {
  try {
    const sent = await processScheduledDigests();
    if (sent > 0) console.log(`Digest scheduler: sent ${sent} digest(s)`);
  } catch (e: any) {
    console.error("Digest scheduler error:", e.message);
  }
}, 5 * 60 * 1000);

console.log(`Engram v5.1 listening on ${HOST}:${PORT}`);
console.log(`Database: ${DB_PATH}`);
console.log(`Embedding model: ${EMBEDDING_MODEL} (${EMBEDDING_DIM}d)`);
console.log(`LLM: ${LLM_API_KEY ? `${LLM_MODEL} via ${LLM_URL}` : "not configured"}`);
console.log(`Auto-link: threshold=${AUTO_LINK_THRESHOLD}, max=${AUTO_LINK_MAX}`);
console.log(`FSRS-6: w20=${FSRS6_WEIGHTS[20]}, retention=${FSRS_DEFAULT_RETENTION}, consolidation=${LLM_API_KEY ? `threshold=${CONSOLIDATION_THRESHOLD}` : "disabled (no LLM)"}`);
console.log(`Auto-forget sweep: every ${FORGET_SWEEP_INTERVAL / 1000}s`);
