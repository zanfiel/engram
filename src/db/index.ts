// ============================================================================
// DATABASE — Schema, migrations, prepared statements, audit
// ============================================================================

import Database from 'libsql';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { mkdirSync } from 'fs';
import { log } from '../config/logger.ts';
import { DB_PATH, DATA_DIR, DEFAULT_RATE_LIMIT, DEFAULT_IMPORTANCE } from '../config/index.ts';
import { FSRSRating, type FSRSMemoryState, fsrsProcessReview, calculateDecayScore } from '../fsrs/index.ts';

export function embeddingToVectorJSON(emb: Float32Array): string {
  return "[" + Array.from(emb).join(",") + "]";
}

mkdirSync(DATA_DIR, { recursive: true });

export const db = new Database(DB_PATH);
db.exec('PRAGMA journal_mode=WAL');
db.exec('PRAGMA foreign_keys=ON');
db.exec('PRAGMA busy_timeout=5000');


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

// Migration helper - logs unexpected errors instead of swallowing them silently
function migrate(sql: string) {
  try {
    db.exec(sql);
  } catch (e: any) {
    const msg = String(e);
    if (msg.includes("duplicate column") || msg.includes("already exists")) return;
    log.warn({ msg: "migration_error", sql: sql.slice(0, 120), error: msg });
  }
}

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
  migrate(`ALTER TABLE memories ADD COLUMN ${col} ${def}`);
}
migrate("ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 5");
migrate("ALTER TABLE memories ADD COLUMN model TEXT");
migrate("ALTER TABLE memories ADD COLUMN embedding BLOB");

// v3.1 indexes
migrate("CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(is_archived) WHERE is_archived = 1");

// v3 indexes
migrate("CREATE INDEX IF NOT EXISTS idx_memories_root ON memories(root_memory_id)");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_parent ON memories(parent_memory_id)");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_latest ON memories(is_latest) WHERE is_latest = 1");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(is_forgotten)");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_forget_after ON memories(forget_after) WHERE forget_after IS NOT NULL");

// v4.1 — Access tracking, tags, episodes
const v41Columns: [string, string][] = [
  ["last_accessed_at", "TEXT"],
  ["access_count", "INTEGER NOT NULL DEFAULT 0"],
  ["tags", "TEXT"],  // JSON array: ["tag1", "tag2"]
  ["episode_id", "INTEGER"],
  ["decay_score", "REAL"],  // cached effective score
];
for (const [col, def] of v41Columns) {
  migrate(`ALTER TABLE memories ADD COLUMN ${col} ${def}`);
}
migrate("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags) WHERE tags IS NOT NULL");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_episode ON memories(episode_id) WHERE episode_id IS NOT NULL");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(access_count DESC)");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_decay ON memories(decay_score DESC)");

// Episodes table
migrate(`
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


// v5.7 — Episode embeddings, FSRS, FTS
migrate("ALTER TABLE episodes ADD COLUMN embedding BLOB");
migrate("ALTER TABLE episodes ADD COLUMN embedding_vec_1024 FLOAT32(1024)");
migrate("ALTER TABLE episodes ADD COLUMN duration_seconds INTEGER");
migrate("ALTER TABLE episodes ADD COLUMN fsrs_stability REAL");
migrate("ALTER TABLE episodes ADD COLUMN fsrs_difficulty REAL");
migrate("ALTER TABLE episodes ADD COLUMN fsrs_last_review_at TEXT");
migrate("ALTER TABLE episodes ADD COLUMN fsrs_reps INTEGER DEFAULT 0");
migrate("ALTER TABLE episodes ADD COLUMN decay_score REAL DEFAULT 1.0");

migrate(`CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
  title, summary, content='episodes', content_rowid='id',
  tokenize='porter unicode61'
)`);

migrate(`CREATE TRIGGER IF NOT EXISTS episodes_fts_ai AFTER INSERT ON episodes BEGIN
  INSERT INTO episodes_fts(rowid, title, summary) VALUES (new.id, new.title, new.summary);
END`);
migrate(`CREATE TRIGGER IF NOT EXISTS episodes_fts_ad AFTER DELETE ON episodes BEGIN
  INSERT INTO episodes_fts(episodes_fts, rowid, title, summary) VALUES ('delete', old.id, old.title, old.summary);
END`);
migrate(`CREATE TRIGGER IF NOT EXISTS episodes_fts_au AFTER UPDATE ON episodes BEGIN
  INSERT INTO episodes_fts(episodes_fts, rowid, title, summary) VALUES ('delete', old.id, old.title, old.summary);
  INSERT INTO episodes_fts(rowid, title, summary) VALUES (new.id, new.title, new.summary);
END`);

migrate("CREATE INDEX IF NOT EXISTS episodes_vec_1024_idx ON episodes(libsql_vector_idx(embedding_vec_1024))");

// Consolidation tracking
migrate(`
    CREATE TABLE IF NOT EXISTS consolidations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      summary_memory_id INTEGER NOT NULL REFERENCES memories(id),
      source_memory_ids TEXT NOT NULL,
      cluster_label TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);
migrate("ALTER TABLE consolidations ADD COLUMN user_id INTEGER NOT NULL DEFAULT 1");
migrate("CREATE INDEX IF NOT EXISTS idx_consolidations_user ON consolidations(user_id)");


// v4.2 — Confidence, sync, webhooks
const v42Columns: [string, string][] = [
  ["confidence", "REAL NOT NULL DEFAULT 1.0"],
  ["sync_id", "TEXT"],  // UUID for cross-instance sync
];
for (const [col, def] of v42Columns) {
  migrate(`ALTER TABLE memories ADD COLUMN ${col} ${def}`);
}
migrate("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_sync_id ON memories(sync_id) WHERE sync_id IS NOT NULL");

// v5.1 — Review queue: status column (pending/approved/rejected)
migrate("ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'approved'");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)");

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
  migrate(`ALTER TABLE memories ADD COLUMN ${col} ${def}`);
}
migrate("CREATE INDEX IF NOT EXISTS idx_memories_fsrs_stability ON memories(fsrs_stability) WHERE fsrs_stability IS NOT NULL");

// v5.0 — Native vector column (libsql FLOAT32)
migrate("ALTER TABLE memories ADD COLUMN embedding_vec FLOAT32(384)");
migrate("CREATE INDEX IF NOT EXISTS memories_vec_idx ON memories(libsql_vector_idx(embedding_vec))");

// v5.7 — BGE-large 1024-dim vector column
migrate("ALTER TABLE memories ADD COLUMN embedding_vec_1024 FLOAT32(1024)");
migrate("CREATE INDEX IF NOT EXISTS memories_vec_1024_idx ON memories(libsql_vector_idx(embedding_vec_1024))");

// Webhooks table
migrate(`
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


// Audit log table
migrate(`
    CREATE TABLE IF NOT EXISTS audit_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      action TEXT NOT NULL,
      target_type TEXT,
      target_id INTEGER,
      details TEXT,
      ip TEXT,
      request_id TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
  `);
migrate("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC)");
migrate("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)");
migrate("CREATE INDEX IF NOT EXISTS idx_audit_target ON audit_log(target_type, target_id)");

// v5.8 — Agent Identity & Trust
migrate(`
  CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    category TEXT,
    description TEXT,
    code_hash TEXT,
    trust_score REAL NOT NULL DEFAULT 50,
    total_ops INTEGER NOT NULL DEFAULT 0,
    successful_ops INTEGER NOT NULL DEFAULT 0,
    failed_ops INTEGER NOT NULL DEFAULT 0,
    guard_allows INTEGER NOT NULL DEFAULT 0,
    guard_warns INTEGER NOT NULL DEFAULT 0,
    guard_blocks INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    revoked_at TEXT,
    revoke_reason TEXT,
    last_seen_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, name)
  );
  CREATE INDEX IF NOT EXISTS idx_agents_user ON agents(user_id);
  CREATE INDEX IF NOT EXISTS idx_agents_active ON agents(is_active);
`);

// Link API keys to agent identities
migrate("ALTER TABLE api_keys ADD COLUMN agent_id INTEGER REFERENCES agents(id)");

// Add agent_id + execution signing columns to audit_log
migrate("ALTER TABLE audit_log ADD COLUMN agent_id INTEGER");
migrate("ALTER TABLE audit_log ADD COLUMN execution_hash TEXT");
migrate("ALTER TABLE audit_log ADD COLUMN signature TEXT");
migrate("CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id)");

export const insertAudit = db.prepare(
  "INSERT INTO audit_log (user_id, action, target_type, target_id, details, ip, request_id, agent_id, execution_hash, signature) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
);
export function audit(userId: number | null, action: string, targetType: string | null, targetId: number | null, details: string | null, ip: string | null, requestId: string | null, agentId?: number | null, executionHash?: string | null, signature?: string | null) {
  try { insertAudit.run(userId, action, targetType, targetId, details, ip, requestId, agentId ?? null, executionHash ?? null, signature ?? null); } catch (e: any) { log.warn({ msg: "audit_write_fail", action, error: String(e).slice(0, 200) }); }
}

// v4.3 — Entities, Projects
migrate(`
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

// v5.5 — Structured Intelligence Tables (bench-driven improvements)
migrate(`
    CREATE TABLE IF NOT EXISTS structured_facts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
      subject TEXT NOT NULL,
      verb TEXT NOT NULL,
      object TEXT,
      quantity REAL,
      unit TEXT,
      date_ref TEXT,
      date_approx TEXT,
      confidence REAL NOT NULL DEFAULT 1.0,
      user_id INTEGER DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_sf_memory ON structured_facts(memory_id);
    CREATE INDEX IF NOT EXISTS idx_sf_subject ON structured_facts(subject COLLATE NOCASE);
    CREATE INDEX IF NOT EXISTS idx_sf_verb ON structured_facts(verb);
    CREATE INDEX IF NOT EXISTS idx_sf_date ON structured_facts(date_approx);
    CREATE INDEX IF NOT EXISTS idx_sf_user ON structured_facts(user_id);
`);

migrate(`
    CREATE TABLE IF NOT EXISTS current_state (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      key TEXT NOT NULL,
      value TEXT NOT NULL,
      memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
      previous_value TEXT,
      previous_memory_id INTEGER,
      updated_count INTEGER NOT NULL DEFAULT 1,
      user_id INTEGER DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(key, user_id)
    );
    CREATE INDEX IF NOT EXISTS idx_cs_key ON current_state(key COLLATE NOCASE);
    CREATE INDEX IF NOT EXISTS idx_cs_user ON current_state(user_id);
`);

migrate(`
    CREATE TABLE IF NOT EXISTS user_preferences (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      domain TEXT NOT NULL,
      preference TEXT NOT NULL,
      strength REAL NOT NULL DEFAULT 1.0,
      evidence_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
      user_id INTEGER DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(domain, preference, user_id)
    );
    CREATE INDEX IF NOT EXISTS idx_up_domain ON user_preferences(domain COLLATE NOCASE);
    CREATE INDEX IF NOT EXISTS idx_up_user ON user_preferences(user_id);
`);



// v4.5 — Digests, Reflections, Contradiction tracking
migrate(`
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


// ============================================================================
// SCHEMA v4 — Multi-tenant: users, API keys, spaces
// ============================================================================

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    role TEXT NOT NULL DEFAULT 'admin',
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

// Scratchpad — short-term working memory with TTL
migrate(`
  CREATE TABLE IF NOT EXISTS scratchpad (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id) ON DELETE CASCADE,
    session TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    entry_key TEXT NOT NULL,
    value TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL DEFAULT (datetime('now', '+30 minutes')),
    UNIQUE(user_id, session, entry_key)
  );
  CREATE INDEX IF NOT EXISTS idx_scratchpad_user_expires ON scratchpad(user_id, expires_at);
  CREATE INDEX IF NOT EXISTS idx_scratchpad_session ON scratchpad(user_id, session);
  CREATE INDEX IF NOT EXISTS idx_scratchpad_agent ON scratchpad(user_id, agent);
`);

// v4 migrations — add user_id and space_id columns
for (const [tbl, col, def] of [
  ["memories", "user_id", "INTEGER NOT NULL DEFAULT 1"],
  ["memories", "space_id", "INTEGER"],
  ["conversations", "user_id", "INTEGER NOT NULL DEFAULT 1"],
] as const) {
  migrate(`ALTER TABLE ${tbl} ADD COLUMN ${col} ${def}`);
}
migrate("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_space ON memories(space_id)");
migrate("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)");

// RBAC: add role column (admin/writer/reader)
migrate("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'admin'");

// Ensure default user exists (backwards compat — all existing data is user_id=1)
const defaultUser = db.prepare("SELECT id FROM users WHERE id = 1").get();
if (!defaultUser) {
  db.exec("INSERT INTO users (id, username, is_admin) VALUES (1, 'owner', 1)");
  log.info({ msg: "created_default_user", id: 1 });
}

// Ensure default space for owner
const defaultSpace = db.prepare("SELECT id FROM spaces WHERE user_id = 1 AND name = 'default'").get();
if (!defaultSpace) {
  db.exec("INSERT INTO spaces (user_id, name, description) VALUES (1, 'default', 'Default memory space')");
  log.info({ msg: "created_default_space" });
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
migrate("ALTER TABLE memory_links ADD COLUMN type TEXT NOT NULL DEFAULT 'similarity'");

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

export const insertMemory = db.prepare(
  `INSERT INTO memories (content, category, source, session_id, importance, embedding,
    version, is_latest, parent_memory_id, root_memory_id, source_count, is_static,
    is_forgotten, forget_after, forget_reason, is_inference, model)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
   RETURNING id, created_at`
);

export const updateMemoryEmbedding = db.prepare(
  `UPDATE memories SET embedding = ? WHERE id = ?`
);

export const updateMemoryVec = db.prepare(
  `UPDATE memories SET embedding_vec_1024 = vector(?) WHERE id = ?`
);

/** Write vector column for a newly inserted memory (call after insertMemory) */
export function writeVec(memoryId: number, embArray: Float32Array | null): void {
  if (!embArray) return;
  try { updateMemoryVec.run(embeddingToVectorJSON(embArray), memoryId); } catch {}
}

export const getAllEmbeddings = db.prepare(
  `SELECT id, user_id, content, category, importance, embedding, is_latest, is_forgotten, is_static, source_count
   FROM memories WHERE embedding IS NOT NULL AND is_forgotten = 0`
);

export const getLatestEmbeddings = db.prepare(
  `SELECT id, content, category, importance, embedding, is_static, source_count
   FROM memories WHERE embedding IS NOT NULL AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND status = 'approved'`
);

export const searchMemoriesFTS = db.prepare(
  `SELECT m.id, m.content, m.category, m.source, m.session_id, m.importance, m.created_at,
     m.version, m.is_latest, m.parent_memory_id, m.root_memory_id, m.source_count,
     m.is_static, m.is_forgotten, m.is_inference, m.model,
     rank as fts_rank
   FROM memories_fts f
   JOIN memories m ON f.rowid = m.id
   WHERE memories_fts MATCH ? AND m.is_forgotten = 0 AND m.user_id = ?
   ORDER BY rank
   LIMIT ?`
);

export const listRecent = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at,
     version, is_latest, parent_memory_id, root_memory_id, source_count,
     is_static, is_forgotten, is_inference, forget_after, is_archived, status, model
   FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND status != 'pending' AND user_id = ? ORDER BY created_at DESC LIMIT ?`
);

export const listByCategory = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at,
     version, is_latest, parent_memory_id, root_memory_id, source_count,
     is_static, is_forgotten, is_inference, forget_after, is_archived, status, model
   FROM memories WHERE category = ? AND is_forgotten = 0 AND is_archived = 0 AND status != 'pending' AND user_id = ? ORDER BY created_at DESC LIMIT ?`
);

export const deleteMemory = db.prepare(`DELETE FROM memories WHERE id = ?`);
export const getMemory = db.prepare(`SELECT * FROM memories WHERE id = ?`);

export const getMemoryWithoutEmbedding = db.prepare(
  `SELECT id, user_id, content, category, source, session_id, importance, created_at, updated_at,
     version, is_latest, parent_memory_id, root_memory_id, source_count,
     is_static, is_forgotten, forget_after, forget_reason, is_inference, is_archived, status, model,
     tags, episode_id, access_count, last_accessed_at, confidence
   FROM memories WHERE id = ?`
);

// Version chain queries
export const getVersionChain = db.prepare(
  `SELECT id, content, category, version, is_latest, created_at, source_count
   FROM memories WHERE root_memory_id = ? OR id = ?
   ORDER BY version ASC`
);

export const markSuperseded = db.prepare(
  `UPDATE memories SET is_latest = 0, updated_at = datetime('now') WHERE id = ?`
);

export const incrementSourceCount = db.prepare(
  `UPDATE memories SET source_count = source_count + 1, updated_at = datetime('now') WHERE id = ?`
);

// Forgetting
export const markForgotten = db.prepare(
  `UPDATE memories SET is_forgotten = 1, updated_at = datetime('now') WHERE id = ?`
);

export const markArchived = db.prepare(
  `UPDATE memories SET is_archived = 1, updated_at = datetime('now') WHERE id = ?`
);

export const markUnarchived = db.prepare(
  `UPDATE memories SET is_archived = 0, updated_at = datetime('now') WHERE id = ?`
);

export const getExpiredMemories = db.prepare(
  `SELECT id, content, forget_reason FROM memories
   WHERE forget_after IS NOT NULL AND forget_after <= datetime('now')
   AND is_forgotten = 0`
);

// Memory links — v3 typed
export const insertLink = db.prepare(
  `INSERT OR IGNORE INTO memory_links (source_id, target_id, similarity, type) VALUES (?, ?, ?, ?)`
);

export const getLinksFor = db.prepare(
  `SELECT ml.target_id as id, ml.similarity, ml.type, m.content, m.category, m.importance, m.created_at,
     m.is_latest, m.is_forgotten, m.version, m.source_count, m.model, m.source
   FROM memory_links ml
   JOIN memories m ON ml.target_id = m.id
   WHERE ml.source_id = ?
   UNION
   SELECT ml.source_id as id, ml.similarity, ml.type, m.content, m.category, m.importance, m.created_at,
     m.is_latest, m.is_forgotten, m.version, m.source_count, m.model, m.source
   FROM memory_links ml
   JOIN memories m ON ml.source_id = m.id
   WHERE ml.target_id = ?
   ORDER BY similarity DESC`
);

// User-scoped variants for search module
export const getLinksForUser = db.prepare(
  `SELECT ml.target_id as id, ml.similarity, ml.type, m.content, m.category, m.importance, m.created_at,
     m.is_latest, m.is_forgotten, m.version, m.source_count, m.model, m.source
   FROM memory_links ml
   JOIN memories m ON ml.target_id = m.id
   WHERE ml.source_id = ? AND m.user_id = ?
   UNION
   SELECT ml.source_id as id, ml.similarity, ml.type, m.content, m.category, m.importance, m.created_at,
     m.is_latest, m.is_forgotten, m.version, m.source_count, m.model, m.source
   FROM memory_links ml
   JOIN memories m ON ml.source_id = m.id
   WHERE ml.target_id = ? AND m.user_id = ?
   ORDER BY similarity DESC`
);

export const getVersionChainForUser = db.prepare(
  `SELECT id, content, category, version, is_latest, created_at, source_count
   FROM memories WHERE (root_memory_id = ? OR id = ?) AND user_id = ?
   ORDER BY version ASC`
);

export const countNoEmbedding = db.prepare(
  `SELECT COUNT(*) as count FROM memories WHERE embedding IS NULL`
);
export const countNoEmbeddingForUser = db.prepare(
  `SELECT COUNT(*) as count FROM memories WHERE embedding IS NULL AND user_id = ?`
);
export const getNoEmbedding = db.prepare(
  `SELECT id, content FROM memories WHERE embedding IS NULL LIMIT ?`
);
export const getNoEmbeddingForUser = db.prepare(
  `SELECT id, content FROM memories WHERE embedding IS NULL AND user_id = ? LIMIT ?`
);

// Profile queries
export const getStaticMemories = db.prepare(
  `SELECT id, content, category, source_count, created_at, updated_at, model, source
   FROM memories WHERE is_static = 1 AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND status = 'approved' AND user_id = ?
   ORDER BY source_count DESC, updated_at DESC`
);

// Access tracking + FSRS review processing
export const trackAccess = db.prepare(
  `UPDATE memories SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?`
);
export const updateFSRS = db.prepare(
  `UPDATE memories SET fsrs_stability = ?, fsrs_difficulty = ?, fsrs_storage_strength = ?,
   fsrs_retrieval_strength = ?, fsrs_learning_state = ?, fsrs_reps = ?, fsrs_lapses = ?,
   fsrs_last_review_at = ? WHERE id = ?`
);
export const getFSRS = db.prepare(
  `SELECT fsrs_stability, fsrs_difficulty, fsrs_storage_strength, fsrs_retrieval_strength,
   fsrs_learning_state, fsrs_reps, fsrs_lapses, fsrs_last_review_at, last_accessed_at, created_at
   FROM memories WHERE id = ?`
);

/** Track access AND process as FSRS review (Grade: Good=recall, Again=forget) */
export function trackAccessWithFSRS(memoryId: number, grade: FSRSRating = FSRSRating.Good): void {
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

export function updateDecayScores(userId?: number): number {
  const query = userId != null
    ? `SELECT id, importance, created_at, access_count, last_accessed_at, is_static, source_count, fsrs_stability
       FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?`
    : `SELECT id, importance, created_at, access_count, last_accessed_at, is_static, source_count, fsrs_stability
       FROM memories WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1`;
  const memories = (userId != null
    ? db.prepare(query).all(userId)
    : db.prepare(query).all()) as Array<any>;

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

// Tags
export const getByTag = db.prepare(
  `SELECT id, content, category, source, importance, created_at, tags, access_count, episode_id
   FROM memories WHERE tags LIKE ? ESCAPE '\\' AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
   ORDER BY created_at DESC LIMIT ?`
);

export const getAllTags = db.prepare(
  `SELECT DISTINCT tags FROM memories WHERE tags IS NOT NULL AND is_forgotten = 0 AND is_archived = 0 AND user_id = ?`
);

// Inbox / Review queue prepared statements
export const listPending = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at, tags, confidence, decay_score, status, model
   FROM memories WHERE status = 'pending' AND is_forgotten = 0 AND user_id = ?
   ORDER BY created_at DESC LIMIT ? OFFSET ?`
);
export const countPending = db.prepare(
  `SELECT COUNT(*) as count FROM memories WHERE status = 'pending' AND is_forgotten = 0 AND user_id = ?`
);
export const approveMemory = db.prepare(
  `UPDATE memories SET status = 'approved', updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);
export const rejectMemory = db.prepare(
  `UPDATE memories SET status = 'rejected', is_archived = 1, updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);

// Episodes
export const insertEpisode = db.prepare(
  `INSERT INTO episodes (title, session_id, agent, user_id) VALUES (?, ?, ?, ?) RETURNING id, started_at`
);
export const updateEpisode = db.prepare(
  `UPDATE episodes SET title = COALESCE(?, title), summary = COALESCE(?, summary),
   ended_at = COALESCE(?, ended_at), memory_count = (SELECT COUNT(*) FROM memories WHERE episode_id = episodes.id)
   WHERE id = ?`
);
export const updateEpisodeForUser = db.prepare(
  `UPDATE episodes SET title = COALESCE(?, title), summary = COALESCE(?, summary),
   ended_at = COALESCE(?, ended_at), memory_count = (SELECT COUNT(*) FROM memories WHERE episode_id = episodes.id)
   WHERE id = ? AND user_id = ?`
);
export const getEpisode = db.prepare(`SELECT * FROM episodes WHERE id = ?`);
export const getEpisodeBySession = db.prepare(
  `SELECT * FROM episodes WHERE session_id = ? AND agent = ? AND user_id = ? ORDER BY started_at DESC LIMIT 1`
);
export const listEpisodes = db.prepare(
  `SELECT * FROM episodes WHERE user_id = ? ORDER BY started_at DESC LIMIT ?`
);
export const getEpisodeMemories = db.prepare(
  `SELECT id, content, category, source, importance, created_at, tags, access_count
   FROM memories WHERE episode_id = ? AND user_id = ? AND is_forgotten = 0 ORDER BY created_at ASC`
);
export const assignToEpisode = db.prepare(
  `UPDATE memories SET episode_id = ? WHERE id = ?`
);
export const assignToEpisodeForUser = db.prepare(
  `UPDATE memories SET episode_id = ? WHERE id = ? AND user_id = ?`
);

// Episode embedding + search
export const updateEpisodeEmbedding = db.prepare(
  `UPDATE episodes SET embedding = ? WHERE id = ?`
);
export const updateEpisodeVec = db.prepare(
  `UPDATE episodes SET embedding_vec_1024 = vector(?) WHERE id = ?`
);
export const searchEpisodesFTS = db.prepare(
  `SELECT e.*, rank FROM episodes_fts f JOIN episodes e ON e.id = f.rowid
   WHERE episodes_fts MATCH ? AND e.user_id = ? ORDER BY rank LIMIT ?`
);
export const listEpisodesByTimeRange = db.prepare(
  `SELECT * FROM episodes WHERE user_id = ? AND started_at >= ? AND started_at <= ? ORDER BY started_at DESC LIMIT ?`
);
export const getAllEpisodeEmbeddings = db.prepare(
  `SELECT id, user_id, summary, embedding FROM episodes WHERE embedding IS NOT NULL`
);

// Consolidation queries
export const getClusterCandidates = db.prepare(
  `SELECT source_id, COUNT(*) as link_count FROM memory_links
   JOIN memories m ON memory_links.source_id = m.id
   WHERE m.user_id = ? AND m.is_forgotten = 0 AND m.is_archived = 0 AND m.is_latest = 1
   GROUP BY source_id HAVING link_count >= ?
   ORDER BY link_count DESC LIMIT 10`
);

export const getClusterMembers = db.prepare(
  `SELECT DISTINCT m.id, m.content, m.category, m.importance, m.created_at, m.access_count
   FROM memory_links ml
   JOIN memories center ON center.id = ?
   JOIN memories m ON (ml.target_id = m.id OR ml.source_id = m.id)
   WHERE center.user_id = ? AND (ml.source_id = center.id OR ml.target_id = center.id)
     AND m.user_id = ? AND m.is_forgotten = 0 AND m.is_archived = 0
   ORDER BY m.importance DESC, m.created_at DESC`
);

// Confidence updates
export const updateConfidence = db.prepare(
  `UPDATE memories SET confidence = ?, updated_at = datetime('now') WHERE id = ?`
);

// Webhook queries
export const insertWebhook = db.prepare(
  `INSERT INTO webhooks (url, events, secret, user_id) VALUES (?, ?, ?, ?) RETURNING id, created_at`
);
export const listWebhooks = db.prepare(
  `SELECT id, url, events, active, last_triggered_at, failure_count, created_at
   FROM webhooks WHERE user_id = ? ORDER BY created_at DESC`
);
export const deleteWebhook = db.prepare(`DELETE FROM webhooks WHERE id = ? AND user_id = ?`);
export const getActiveWebhooks = db.prepare(
  `SELECT id, url, events, secret FROM webhooks WHERE active = 1 AND user_id = ?`
);
export const webhookTriggered = db.prepare(
  `UPDATE webhooks SET last_triggered_at = datetime('now') WHERE id = ?`
);
export const webhookFailed = db.prepare(
  `UPDATE webhooks SET failure_count = failure_count + 1,
   active = CASE WHEN failure_count >= 9 THEN 0 ELSE active END WHERE id = ?`
);

// Sync queries
export const getChangesSince = db.prepare(
  `SELECT id, content, category, source, session_id, importance, tags, confidence,
     sync_id, is_static, is_forgotten, is_archived, version, created_at, updated_at
   FROM memories WHERE updated_at > ? AND user_id = ?
   ORDER BY updated_at ASC LIMIT ?`
);
export const getMemoryBySyncId = db.prepare(
  `SELECT id, updated_at FROM memories WHERE sync_id = ? AND user_id = ?`
);

// Entity queries
export const insertEntity = db.prepare(
  `INSERT INTO entities (name, type, description, aka, metadata, user_id)
   VALUES (?, ?, ?, ?, ?, ?) RETURNING id, created_at`
);
export const getEntity = db.prepare(
  `SELECT e.*, GROUP_CONCAT(DISTINCT me.memory_id) as memory_ids
   FROM entities e LEFT JOIN memory_entities me ON me.entity_id = e.id
   WHERE e.id = ? GROUP BY e.id`
);
export const listEntities = db.prepare(
  `SELECT e.id, e.name, e.type, e.description, e.aka, e.created_at,
     (SELECT COUNT(*) FROM memory_entities WHERE entity_id = e.id) as memory_count
   FROM entities e WHERE e.user_id = ? ORDER BY e.name COLLATE NOCASE`
);
export const listEntitiesByType = db.prepare(
  `SELECT e.id, e.name, e.type, e.description, e.aka, e.created_at,
     (SELECT COUNT(*) FROM memory_entities WHERE entity_id = e.id) as memory_count
   FROM entities e WHERE e.user_id = ? AND e.type = ? ORDER BY e.name COLLATE NOCASE`
);
export const searchEntities = db.prepare(
  `SELECT e.id, e.name, e.type, e.description, e.aka, e.created_at,
     (SELECT COUNT(*) FROM memory_entities WHERE entity_id = e.id) as memory_count
   FROM entities e WHERE e.user_id = ? AND (e.name LIKE ? OR e.aka LIKE ? OR e.description LIKE ?)
   ORDER BY e.name COLLATE NOCASE LIMIT ?`
);
export const updateEntity = db.prepare(
  `UPDATE entities SET name = COALESCE(?, name), type = COALESCE(?, type),
   description = COALESCE(?, description), aka = COALESCE(?, aka),
   metadata = COALESCE(?, metadata), updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);
export const deleteEntity = db.prepare(`DELETE FROM entities WHERE id = ? AND user_id = ?`);
export const linkMemoryEntity = db.prepare(
  `INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)`
);
export const unlinkMemoryEntity = db.prepare(
  `DELETE FROM memory_entities WHERE memory_id = ? AND entity_id = ?`
);
export const getEntityMemories = db.prepare(
  `SELECT m.id, m.content, m.category, m.importance, m.tags, m.created_at, m.decay_score, m.confidence
   FROM memories m JOIN memory_entities me ON me.memory_id = m.id
   WHERE me.entity_id = ? AND m.user_id = ? AND m.is_forgotten = 0 AND m.is_archived = 0
   ORDER BY m.created_at DESC LIMIT ?`
);
export const insertEntityRelationship = db.prepare(
  `INSERT OR IGNORE INTO entity_relationships (source_entity_id, target_entity_id, relationship) VALUES (?, ?, ?)`
);
export const deleteEntityRelationship = db.prepare(
  `DELETE FROM entity_relationships WHERE source_entity_id = ? AND target_entity_id = ? AND relationship = ?`
);
export const getEntityRelationships = db.prepare(
  `SELECT er.id, er.relationship, er.created_at,
     CASE WHEN er.source_entity_id = ? THEN er.target_entity_id ELSE er.source_entity_id END as related_entity_id,
     CASE WHEN er.source_entity_id = ? THEN 'outgoing' ELSE 'incoming' END as direction,
     e.name as related_entity_name, e.type as related_entity_type
   FROM entity_relationships er
   JOIN entities e ON e.id = CASE WHEN er.source_entity_id = ? THEN er.target_entity_id ELSE er.source_entity_id END
   WHERE er.source_entity_id = ? OR er.target_entity_id = ?`
);

// Project queries
export const insertProject = db.prepare(
  `INSERT INTO projects (name, description, status, metadata, user_id)
   VALUES (?, ?, ?, ?, ?) RETURNING id, created_at`
);
export const getProject = db.prepare(
  `SELECT p.*, GROUP_CONCAT(DISTINCT mp.memory_id) as memory_ids
   FROM projects p LEFT JOIN memory_projects mp ON mp.project_id = p.id
   WHERE p.id = ? GROUP BY p.id`
);
export const listProjects = db.prepare(
  `SELECT p.id, p.name, p.description, p.status, p.created_at,
     (SELECT COUNT(*) FROM memory_projects WHERE project_id = p.id) as memory_count
   FROM projects p WHERE p.user_id = ? ORDER BY p.status = 'active' DESC, p.name COLLATE NOCASE`
);
export const listProjectsByStatus = db.prepare(
  `SELECT p.id, p.name, p.description, p.status, p.created_at,
     (SELECT COUNT(*) FROM memory_projects WHERE project_id = p.id) as memory_count
   FROM projects p WHERE p.user_id = ? AND p.status = ? ORDER BY p.name COLLATE NOCASE`
);
export const updateProject = db.prepare(
  `UPDATE projects SET name = COALESCE(?, name), description = COALESCE(?, description),
   status = COALESCE(?, status), metadata = COALESCE(?, metadata),
   updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);
export const deleteProject = db.prepare(`DELETE FROM projects WHERE id = ? AND user_id = ?`);
export const linkMemoryProject = db.prepare(
  `INSERT OR IGNORE INTO memory_projects (memory_id, project_id) VALUES (?, ?)`
);
export const unlinkMemoryProject = db.prepare(
  `DELETE FROM memory_projects WHERE memory_id = ? AND project_id = ?`
);
export const getProjectMemories = db.prepare(
  `SELECT m.id, m.content, m.category, m.importance, m.tags, m.created_at, m.decay_score, m.confidence
   FROM memories m JOIN memory_projects mp ON mp.memory_id = m.id
   WHERE mp.project_id = ? AND m.user_id = ? AND m.is_forgotten = 0 AND m.is_archived = 0
   ORDER BY m.created_at DESC LIMIT ?`
);

export const getRecentDynamicMemories = db.prepare(
  `SELECT id, content, category, source_count, created_at, model, source
   FROM memories WHERE is_static = 0 AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
   ORDER BY created_at DESC LIMIT ?`
);

function changeCount(result: any): number {
  return Number(result?.rowsAffected ?? result?.changes ?? 0);
}

export const upsertScratchEntry = db.prepare(
  `INSERT INTO scratchpad (user_id, session, agent, model, entry_key, value, expires_at)
   VALUES (?, ?, ?, ?, ?, ?, datetime('now', '+30 minutes'))
   ON CONFLICT(user_id, session, entry_key) DO UPDATE SET
     agent = excluded.agent,
     model = excluded.model,
     value = excluded.value,
     updated_at = datetime('now'),
     expires_at = datetime('now', '+30 minutes')`
);

export const listScratchEntries = db.prepare(
  `SELECT session, agent, model, entry_key, value, created_at, updated_at, expires_at
   FROM scratchpad
   WHERE user_id = ?
     AND expires_at > datetime('now')
     AND (? IS NULL OR agent = ?)
     AND (? IS NULL OR model = ?)
     AND (? IS NULL OR session = ?)
   ORDER BY updated_at DESC, agent, session, entry_key`
);

export const listScratchEntriesForContext = db.prepare(
  `SELECT session, agent, model, entry_key, value, updated_at
   FROM scratchpad
   WHERE user_id = ?
     AND expires_at > datetime('now')
     AND (? IS NULL OR session != ?)
   ORDER BY updated_at DESC, agent, session, entry_key
   LIMIT 20`
);

export const deleteScratchSession = db.prepare(
  `DELETE FROM scratchpad WHERE user_id = ? AND session = ?`
);

export const deleteScratchSessionKey = db.prepare(
  `DELETE FROM scratchpad WHERE user_id = ? AND session = ? AND entry_key = ?`
);

const purgeExpiredScratchEntriesStmt = db.prepare(
  `DELETE FROM scratchpad WHERE expires_at <= datetime('now')`
);

export function purgeExpiredScratchpad(): number {
  return changeCount(purgeExpiredScratchEntriesStmt.run());
}

// Graph data
export const getAllMemoriesForGraph = db.prepare(
  `SELECT id, content, category, importance, is_latest, is_forgotten, is_static,
     is_inference, version, parent_memory_id, root_memory_id, source_count,
     forget_after, created_at, tags, access_count, episode_id, decay_score
   FROM memories WHERE user_id = ? ORDER BY created_at DESC`
);

export const getAllLinksForGraph = db.prepare(
  `SELECT ml.source_id, ml.target_id, ml.similarity, ml.type FROM memory_links ml
   JOIN memories m ON ml.source_id = m.id WHERE m.user_id = ?`
);

// ============================================================================
// PREPARED STATEMENTS — conversations (unchanged)
// ============================================================================

export const insertConversation = db.prepare(
  `INSERT INTO conversations (agent, session_id, title, metadata, user_id) VALUES (?, ?, ?, ?, ?) RETURNING id, started_at`
);
export const updateConversation = db.prepare(
  `UPDATE conversations SET title = COALESCE(?, title), metadata = COALESCE(?, metadata), updated_at = datetime('now') WHERE id = ? AND user_id = ?`
);
export const getConversation = db.prepare(`SELECT * FROM conversations WHERE id = ?`);
export const getConversationForUser = db.prepare(`SELECT * FROM conversations WHERE id = ? AND user_id = ?`);
export const getConversationBySession = db.prepare(
  `SELECT * FROM conversations WHERE agent = ? AND session_id = ? AND user_id = ? ORDER BY started_at DESC LIMIT 1`
);
export const listConversations = db.prepare(
  `SELECT c.id, c.agent, c.session_id, c.title, c.metadata, c.started_at, c.updated_at,
     (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
   FROM conversations c WHERE c.user_id = ? ORDER BY c.updated_at DESC LIMIT ?`
);
export const listConversationsByAgent = db.prepare(
  `SELECT c.id, c.agent, c.session_id, c.title, c.metadata, c.started_at, c.updated_at,
     (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
   FROM conversations c WHERE c.user_id = ? AND c.agent = ? ORDER BY c.updated_at DESC LIMIT ?`
);
export const deleteConversation = db.prepare(`DELETE FROM conversations WHERE id = ? AND user_id = ?`);
export const insertMessage = db.prepare(
  `INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?) RETURNING id, created_at`
);
export const getMessages = db.prepare(
  `SELECT id, role, content, metadata, created_at FROM messages
   WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?`
);
export const searchMessages = db.prepare(
  `SELECT m.id, m.conversation_id, m.role, m.content, m.metadata, m.created_at,
     c.agent, c.title as conv_title
   FROM messages_fts f
   JOIN messages m ON f.rowid = m.id
   JOIN conversations c ON m.conversation_id = c.id
   WHERE messages_fts MATCH ? AND c.user_id = ?
   ORDER BY m.created_at DESC
   LIMIT ?`
);
export const touchConversation = db.prepare(
  `UPDATE conversations SET updated_at = datetime('now') WHERE id = ?`
);

// Transaction-safe inserts (no RETURNING — avoids libsql "statements in progress" bug)
export const insertConversationTx = db.prepare(
  `INSERT INTO conversations (agent, session_id, title, metadata, user_id) VALUES (?, ?, ?, ?, ?)`
);
export const insertMessageTx = db.prepare(
  `INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)`
);
const getLastRowId = db.prepare(`SELECT last_insert_rowid() as id`);

export const bulkInsertConvo = db.transaction(
  (agent: string, sessionId: string | null, title: string | null, metadata: string | null, userId: number,
   msgs: Array<{ role: string; content: string; metadata?: string | null }>) => {
    insertConversationTx.run(agent, sessionId, title, metadata, userId);
    const { id } = getLastRowId.get() as { id: number };
    for (const msg of msgs) {
      insertMessageTx.run(id, msg.role, msg.content, msg.metadata || null);
    }
    const conv = getConversation.get(id) as { id: number; started_at: string };
    return conv;
  }
);

// ============================================================================
// TIER 4 SCHEMA — Novel features
// ============================================================================

// Causal chains — temporal cause-effect relationships between memories
migrate(`
  CREATE TABLE IF NOT EXISTS causal_chains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    user_id INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_cc_user ON causal_chains(user_id);
`);

migrate(`
  CREATE TABLE IF NOT EXISTS causal_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_id INTEGER NOT NULL REFERENCES causal_chains(id) ON DELETE CASCADE,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT 'event',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(chain_id, memory_id)
  );
  CREATE INDEX IF NOT EXISTS idx_cl_chain ON causal_links(chain_id);
  CREATE INDEX IF NOT EXISTS idx_cl_memory ON causal_links(memory_id);
`);

// Emotional valence — sentiment/affect tracking per memory
migrate("ALTER TABLE memories ADD COLUMN valence REAL");
migrate("ALTER TABLE memories ADD COLUMN arousal REAL");
migrate("ALTER TABLE memories ADD COLUMN dominant_emotion TEXT");
migrate("CREATE INDEX IF NOT EXISTS idx_memories_valence ON memories(valence) WHERE valence IS NOT NULL");

// Reconsolidation tracking
migrate(`
  CREATE TABLE IF NOT EXISTS reconsolidations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    old_importance INTEGER,
    new_importance INTEGER,
    old_confidence REAL,
    new_confidence REAL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_recon_memory ON reconsolidations(memory_id);
`);

// Adaptive importance
migrate("ALTER TABLE memories ADD COLUMN recall_hits INTEGER NOT NULL DEFAULT 0");
migrate("ALTER TABLE memories ADD COLUMN recall_misses INTEGER NOT NULL DEFAULT 0");
migrate("ALTER TABLE memories ADD COLUMN adaptive_score REAL");

// Temporal patterns
migrate(`
  CREATE TABLE IF NOT EXISTS temporal_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL DEFAULT 1,
    day_of_week INTEGER NOT NULL,
    hour_of_day INTEGER NOT NULL,
    category TEXT,
    project_id INTEGER,
    access_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(user_id, day_of_week, hour_of_day, category, project_id)
  );
  CREATE INDEX IF NOT EXISTS idx_tp_user ON temporal_patterns(user_id, day_of_week, hour_of_day);
`);

// Prepared statements for Tier 4 features
export const insertCausalChain = db.prepare(
  "INSERT INTO causal_chains (name, user_id) VALUES (?, ?) RETURNING id"
);
export const insertCausalLink = db.prepare(
  "INSERT OR IGNORE INTO causal_links (chain_id, memory_id, position, role) VALUES (?, ?, ?, ?)"
);
export const getCausalChainForMemory = db.prepare(
  `SELECT cc.id, cc.name, cc.created_at,
     GROUP_CONCAT(cl.memory_id || ':' || cl.position || ':' || cl.role, '|') as links
   FROM causal_chains cc
   JOIN causal_links cl ON cl.chain_id = cc.id
   WHERE cc.id IN (SELECT chain_id FROM causal_links WHERE memory_id = ?)
   GROUP BY cc.id ORDER BY cc.created_at DESC`
);
export const getCausalChainMemories = db.prepare(
  `SELECT cl.position, cl.role, m.id, m.content, m.category, m.importance, m.created_at,
     m.valence, m.dominant_emotion
   FROM causal_links cl
   JOIN memories m ON cl.memory_id = m.id
   WHERE cl.chain_id = ?
   ORDER BY cl.position ASC`
);

export const updateValence = db.prepare(
  "UPDATE memories SET valence = ?, arousal = ?, dominant_emotion = ? WHERE id = ?"
);

export const insertReconsolidation = db.prepare(
  "INSERT INTO reconsolidations (memory_id, old_importance, new_importance, old_confidence, new_confidence, reason) VALUES (?, ?, ?, ?, ?, ?)"
);
export const getReconsolidationHistory = db.prepare(
  "SELECT * FROM reconsolidations WHERE memory_id = ? ORDER BY created_at DESC LIMIT ?"
);

export const updateAdaptiveScore = db.prepare(
  "UPDATE memories SET adaptive_score = ?, recall_hits = ?, recall_misses = ? WHERE id = ?"
);
export const getMemoriesForReconsolidation = db.prepare(
  `SELECT id, content, category, importance, confidence, created_at, access_count,
     fsrs_stability, fsrs_retrieval_strength, recall_hits, recall_misses, adaptive_score
   FROM memories
   WHERE is_forgotten = 0 AND is_archived = 0 AND is_latest = 1 AND user_id = ?
     AND (
       (adaptive_score IS NOT NULL AND adaptive_score < 0.3)
       OR (access_count > 5 AND recall_misses > recall_hits)
       OR (created_at < datetime('now', '-7 days') AND fsrs_stability IS NOT NULL AND fsrs_stability < 1.0)
     )
   ORDER BY RANDOM() LIMIT ?`
);

export const insertTemporalPattern = db.prepare(
  `INSERT INTO temporal_patterns (user_id, day_of_week, hour_of_day, category, project_id, access_count)
   VALUES (?, ?, ?, ?, ?, 1)
   ON CONFLICT(user_id, day_of_week, hour_of_day, category, project_id)
   DO UPDATE SET access_count = access_count + 1`
);
export const getTemporalPatterns = db.prepare(
  "SELECT * FROM temporal_patterns WHERE user_id = ? ORDER BY access_count DESC LIMIT ?"
);
export const getTemporalPatternsForNow = db.prepare(
  `SELECT tp.category, tp.project_id, tp.access_count, p.name as project_name
   FROM temporal_patterns tp
   LEFT JOIN projects p ON tp.project_id = p.id
   WHERE tp.user_id = ? AND tp.day_of_week = ? AND tp.hour_of_day = ?
   ORDER BY tp.access_count DESC LIMIT ?`
);

// ============================================================================
// AGENT IDENTITY — prepared statements
// ============================================================================

export const insertAgent = db.prepare(
  `INSERT INTO agents (user_id, name, category, description, code_hash) VALUES (?, ?, ?, ?, ?) RETURNING id, trust_score, created_at`
);

export const getAgent = db.prepare(
  `SELECT * FROM agents WHERE id = ? AND user_id = ?`
);

export const getAgentByName = db.prepare(
  `SELECT * FROM agents WHERE name = ? AND user_id = ?`
);

export const listAgents = db.prepare(
  `SELECT id, name, category, description, trust_score, total_ops, successful_ops, failed_ops, guard_allows, guard_warns, guard_blocks, is_active, last_seen_at, created_at FROM agents WHERE user_id = ? ORDER BY created_at DESC`
);

export const updateAgentTrust = db.prepare(
  `UPDATE agents SET trust_score = ?, total_ops = ?, successful_ops = ?, failed_ops = ?, guard_allows = ?, guard_warns = ?, guard_blocks = ?, last_seen_at = datetime('now') WHERE id = ?`
);

export const revokeAgent = db.prepare(
  `UPDATE agents SET is_active = 0, revoked_at = datetime('now'), revoke_reason = ?, trust_score = 0 WHERE id = ? AND user_id = ?`
);

export const getAgentByKeyId = db.prepare(
  `SELECT a.* FROM agents a JOIN api_keys ak ON ak.agent_id = a.id WHERE ak.id = ? AND a.is_active = 1`
);

export const linkKeyToAgent = db.prepare(
  `UPDATE api_keys SET agent_id = ? WHERE id = ? AND user_id = ?`
);

export const getAgentExecutions = db.prepare(
  `SELECT id, action, target_type, target_id, details, execution_hash, signature, created_at FROM audit_log WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?`
);

// ============================================================================
// USER-SCOPED LOOKUPS
// ============================================================================

export const getEpisodeForUser = db.prepare(`SELECT * FROM episodes WHERE id = ? AND user_id = ?`);

export const getFSRSForUser = db.prepare(
  `SELECT fsrs_stability, fsrs_difficulty, fsrs_storage_strength, fsrs_retrieval_strength,
   fsrs_learning_state, fsrs_reps, fsrs_lapses, fsrs_last_review_at, last_accessed_at, created_at
   FROM memories WHERE id = ? AND user_id = ?`
);

export const getEntityForUser = db.prepare(
  `SELECT e.*, GROUP_CONCAT(DISTINCT me.memory_id) as memory_ids
   FROM entities e LEFT JOIN memory_entities me ON me.entity_id = e.id
   WHERE e.id = ? AND e.user_id = ? GROUP BY e.id`
);

export const getProjectForUser = db.prepare(
  `SELECT p.*, GROUP_CONCAT(DISTINCT mp.memory_id) as memory_ids
   FROM projects p LEFT JOIN memory_projects mp ON mp.project_id = p.id
   WHERE p.id = ? AND p.user_id = ? GROUP BY p.id`
);
