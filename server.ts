import { Database } from "bun:sqlite";
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

function bufferToEmbedding(buf: Buffer | Uint8Array): Float32Array {
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return new Float32Array(ab);
}

// ============================================================================
// DATABASE
// ============================================================================

const db = new Database(DB_PATH, { create: true });
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
];
for (const [col, def] of v3Columns) {
  try { db.exec(`ALTER TABLE memories ADD COLUMN ${col} ${def}`); } catch {}
}
try { db.exec("ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 5"); } catch {}
try { db.exec("ALTER TABLE memories ADD COLUMN embedding BLOB"); } catch {}

// v3 indexes
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_root ON memories(root_memory_id)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_parent ON memories(parent_memory_id)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_latest ON memories(is_latest) WHERE is_latest = 1"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(is_forgotten)"); } catch {}
try { db.exec("CREATE INDEX IF NOT EXISTS idx_memories_forget_after ON memories(forget_after) WHERE forget_after IS NOT NULL"); } catch {}

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

const getAllEmbeddings = db.prepare(
  `SELECT id, content, category, importance, embedding, is_latest, is_forgotten, is_static, source_count
   FROM memories WHERE embedding IS NOT NULL AND is_forgotten = 0`
);

const getLatestEmbeddings = db.prepare(
  `SELECT id, content, category, importance, embedding, is_static, source_count
   FROM memories WHERE embedding IS NOT NULL AND is_forgotten = 0 AND is_latest = 1`
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
     is_static, is_forgotten, is_inference, forget_after
   FROM memories WHERE is_forgotten = 0 ORDER BY created_at DESC LIMIT ?`
);

const listByCategory = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at,
     version, is_latest, parent_memory_id, root_memory_id, source_count,
     is_static, is_forgotten, is_inference, forget_after
   FROM memories WHERE category = ? AND is_forgotten = 0 ORDER BY created_at DESC LIMIT ?`
);

const deleteMemory = db.prepare(`DELETE FROM memories WHERE id = ?`);
const getMemory = db.prepare(`SELECT * FROM memories WHERE id = ?`);

const getMemoryWithoutEmbedding = db.prepare(
  `SELECT id, content, category, source, session_id, importance, created_at, updated_at,
     version, is_latest, parent_memory_id, root_memory_id, source_count,
     is_static, is_forgotten, forget_after, forget_reason, is_inference
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
   FROM memories WHERE is_static = 1 AND is_forgotten = 0 AND is_latest = 1
   ORDER BY source_count DESC, updated_at DESC`
);

const getRecentDynamicMemories = db.prepare(
  `SELECT id, content, category, source_count, created_at
   FROM memories WHERE is_static = 0 AND is_forgotten = 0 AND is_latest = 1
   ORDER BY created_at DESC LIMIT ?`
);

// Graph data
const getAllMemoriesForGraph = db.prepare(
  `SELECT id, content, category, importance, is_latest, is_forgotten, is_static,
     is_inference, version, parent_memory_id, root_memory_id, source_count,
     forget_after, created_at
   FROM memories ORDER BY created_at DESC`
);

const getAllLinksForGraph = db.prepare(
  `SELECT source_id, target_id, similarity, type FROM memory_links`
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
    type: "none" | "updates" | "extends" | "duplicate";
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
  "relation_to_existing": {
    "type": "none|updates|extends|duplicate",
    "existing_memory_id": number_or_null,
    "reason": "why this relation was determined"
  }
}

Rules:
- "updates" = new content contradicts or supersedes an existing memory
- "extends" = new content adds to/enriches an existing memory without contradicting it
- "duplicate" = new content says essentially the same thing as an existing memory
- "none" = no meaningful relation to any existing memory
- For forget_after: use ISO 8601 datetime. Tasks/events might expire in days-weeks. Permanent facts = null.
- The "facts" array should contain the KEY discrete facts from the content (1-3 facts usually)
- Category should match the original content's category if it makes sense`;

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
      console.log(`Memory #${newMemoryId} updates #${rel.existing_memory_id} (v${newVersion}, root=#${rootId})`);
    }
  }

  if (rel.type === "extends" && rel.existing_memory_id) {
    insertLink.run(newMemoryId, rel.existing_memory_id, 0.9, "extends");
    console.log(`Memory #${newMemoryId} extends #${rel.existing_memory_id}`);
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
  version?: number;
  is_latest?: boolean;
  is_static?: boolean;
  source_count?: number;
  root_memory_id?: number;
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

  // 1. Vector search
  try {
    const queryEmb = await embed(query);
    const allMems = (latestOnly ? getLatestEmbeddings : getAllEmbeddings).all() as Array<{
      id: number; content: string; category: string; importance: number; embedding: Buffer;
      is_latest?: boolean; is_forgotten?: boolean; is_static?: boolean; source_count?: number;
    }>;

    for (const mem of allMems) {
      if (!mem.embedding) continue;
      const memEmb = bufferToEmbedding(mem.embedding);
      const sim = cosineSimilarity(queryEmb, memEmb);
      if (sim > 0.25) {
        results.set(mem.id, {
          id: mem.id,
          content: mem.content,
          category: mem.category,
          importance: mem.importance,
          created_at: "",
          score: sim * 0.55,
          is_static: !!mem.is_static,
          source_count: mem.source_count || 1,
        });
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

  // 3. Boost by importance + source_count + static priority
  for (const r of results.values()) {
    r.score += (r.importance / 10) * 0.05;
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
  const allMems = getLatestEmbeddings.all() as Array<{
    id: number; content: string; category: string; importance: number; embedding: Buffer;
  }>;

  const similarities: Array<{ id: number; similarity: number }> = [];

  for (const mem of allMems) {
    if (mem.id === memoryId || !mem.embedding) continue;
    const memEmb = bufferToEmbedding(mem.embedding);
    const sim = cosineSimilarity(embedding, memEmb);
    if (sim >= AUTO_LINK_THRESHOLD) {
      similarities.push({ id: mem.id, similarity: sim });
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

async function generateProfile(generateSummary: boolean = false): Promise<UserProfile> {
  const staticFacts = getStaticMemories.all() as Array<{
    id: number; content: string; category: string; source_count: number; created_at: string; updated_at: string;
  }>;

  const recentDynamic = getRecentDynamicMemories.all(20) as Array<{
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
      updateMemoryEmbedding.run(embeddingToBuffer(emb), mem.id);
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
// WEB GUI HTML
// ============================================================================

const GUI_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MegaMind v3</title>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"><\/script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; overflow: hidden; height: 100vh; }
#app { display: grid; grid-template-columns: 1fr 360px; grid-template-rows: 48px 1fr; height: 100vh; }
#toolbar { grid-column: 1 / -1; background: #12121a; border-bottom: 1px solid #2a2a3a; display: flex; align-items: center; padding: 0 16px; gap: 12px; }
#toolbar h1 { font-size: 16px; font-weight: 600; color: #8b5cf6; margin-right: 16px; }
#toolbar input { background: #1a1a2a; border: 1px solid #2a2a3a; border-radius: 6px; padding: 6px 12px; color: #e0e0e0; font-size: 13px; width: 280px; outline: none; }
#toolbar input:focus { border-color: #8b5cf6; }
#toolbar button { background: #8b5cf6; color: white; border: none; border-radius: 6px; padding: 6px 14px; font-size: 13px; cursor: pointer; }
#toolbar button:hover { background: #7c3aed; }
#toolbar .stats { margin-left: auto; font-size: 12px; color: #888; display: flex; gap: 14px; }
#toolbar .stats span { color: #8b5cf6; font-weight: 600; }
.filter-group { display: flex; gap: 4px; }
.filter-btn { background: #1a1a2a; border: 1px solid #2a2a3a; border-radius: 4px; padding: 3px 8px; font-size: 11px; cursor: pointer; color: #aaa; }
.filter-btn.active { background: #8b5cf6; border-color: #8b5cf6; color: white; }
#cy { background: #0a0a0f; }
#sidebar { background: #12121a; border-left: 1px solid #2a2a3a; overflow-y: auto; padding: 16px; }
#sidebar h2 { font-size: 14px; color: #8b5cf6; margin-bottom: 12px; }
#sidebar .empty { color: #555; font-size: 13px; text-align: center; padding-top: 40px; }
.detail-card { background: #1a1a2a; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
.detail-card .label { font-size: 11px; color: #666; text-transform: uppercase; margin-bottom: 4px; }
.detail-card .value { font-size: 13px; line-height: 1.5; word-break: break-word; }
.detail-card .value.content { max-height: 200px; overflow-y: auto; white-space: pre-wrap; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-right: 4px; }
.badge.task { background: #1e3a5f; color: #5b9bd5; }
.badge.discovery { background: #1e4d3a; color: #5bd5a0; }
.badge.decision { background: #4d3a1e; color: #d5a05b; }
.badge.state { background: #3a1e4d; color: #a05bd5; }
.badge.issue { background: #4d1e1e; color: #d55b5b; }
.badge.general { background: #2a2a3a; color: #aaa; }
.badge.static { background: #1e3a1e; color: #5bd55b; }
.badge.forgotten { background: #3a1e1e; color: #d55b5b; }
.badge.latest { background: #1e1e3a; color: #5b5bd5; }
.link-item { background: #1a1a2a; border-radius: 6px; padding: 8px; margin-bottom: 6px; cursor: pointer; }
.link-item:hover { background: #222236; }
.link-item .type { font-size: 10px; font-weight: 600; text-transform: uppercase; }
.link-item .type.similarity { color: #888; }
.link-item .type.updates { color: #ef4444; }
.link-item .type.extends { color: #3b82f6; }
.link-item .type.derives { color: #22c55e; }
.link-item .preview { font-size: 12px; color: #aaa; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.version-chain { display: flex; flex-direction: column; gap: 4px; }
.version-item { background: #1a1a2a; border-radius: 6px; padding: 6px 8px; font-size: 12px; cursor: pointer; border-left: 3px solid #2a2a3a; }
.version-item.current { border-left-color: #8b5cf6; background: #1e1e2e; }
.version-item .ver { color: #8b5cf6; font-weight: 600; }
</style>
</head>
<body>
<div id="app">
  <div id="toolbar">
    <h1>MegaMind v3</h1>
    <input id="search" placeholder="Search memories..." />
    <button onclick="doSearch()">Search</button>
    <div class="filter-group">
      <button class="filter-btn active" data-filter="all" onclick="setFilter('all',this)">All</button>
      <button class="filter-btn" data-filter="latest" onclick="setFilter('latest',this)">Latest</button>
      <button class="filter-btn" data-filter="static" onclick="setFilter('static',this)">Static</button>
      <button class="filter-btn" data-filter="forgotten" onclick="setFilter('forgotten',this)">Forgotten</button>
    </div>
    <div class="stats">
      <div>Memories: <span id="stat-mem">0</span></div>
      <div>Links: <span id="stat-links">0</span></div>
      <div>Static: <span id="stat-static">0</span></div>
      <div>Forgotten: <span id="stat-forgotten">0</span></div>
    </div>
  </div>
  <div id="cy"></div>
  <div id="sidebar">
    <h2>Memory Inspector</h2>
    <div class="empty" id="sidebar-empty">Click a node to inspect</div>
    <div id="sidebar-content" style="display:none"></div>
  </div>
</div>
<script>
const BASE = location.origin;
let cy, graphData, currentFilter = 'all';

const CATEGORY_COLORS = {
  task: '#5b9bd5', discovery: '#5bd5a0', decision: '#d5a05b',
  state: '#a05bd5', issue: '#d55b5b', general: '#888'
};
const LINK_COLORS = { similarity: '#444', updates: '#ef4444', extends: '#3b82f6', derives: '#22c55e' };

async function loadGraph() {
  const [gRes, sRes] = await Promise.all([
    fetch(BASE + '/graph').then(r => r.json()),
    fetch(BASE + '/stats').then(r => r.json())
  ]);
  graphData = gRes;
  document.getElementById('stat-mem').textContent = sRes.memories?.total ?? 0;
  document.getElementById('stat-links').textContent = sRes.links?.total ?? 0;
  document.getElementById('stat-static').textContent = sRes.memories?.static ?? 0;
  document.getElementById('stat-forgotten').textContent = sRes.memories?.forgotten ?? 0;
  renderGraph();
}

function renderGraph() {
  const nodes = [], edges = [];
  const mems = graphData.memories || [];
  const links = graphData.links || [];

  for (const m of mems) {
    if (currentFilter === 'latest' && !m.is_latest) continue;
    if (currentFilter === 'static' && !m.is_static) continue;
    if (currentFilter === 'forgotten' && !m.is_forgotten) continue;
    if (currentFilter === 'all' && m.is_forgotten) continue;
    const color = CATEGORY_COLORS[m.category] || '#888';
    const size = 12 + Math.min(m.importance * 3, 24) + Math.min((m.source_count - 1) * 2, 10);
    const opacity = m.is_forgotten ? 0.3 : m.is_latest ? 1 : 0.6;
    nodes.push({ data: {
      id: 'n' + m.id, memId: m.id, label: m.content.slice(0, 40),
      color, size, opacity, category: m.category, is_static: m.is_static,
      is_forgotten: m.is_forgotten, is_latest: m.is_latest, version: m.version
    }});
  }

  const nodeIds = new Set(nodes.map(n => n.data.memId));
  for (const l of links) {
    if (!nodeIds.has(l.source_id) || !nodeIds.has(l.target_id)) continue;
    const color = LINK_COLORS[l.type] || '#444';
    const width = l.type === 'similarity' ? 1 : 2;
    edges.push({ data: {
      id: 'e' + l.source_id + '_' + l.target_id + '_' + l.type,
      source: 'n' + l.source_id, target: 'n' + l.target_id,
      color, width, type: l.type, similarity: l.similarity
    }});
  }

  if (cy) cy.destroy();
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: { nodes, edges },
    style: [
      { selector: 'node', style: {
        'background-color': 'data(color)', 'width': 'data(size)', 'height': 'data(size)',
        'label': 'data(label)', 'font-size': '8px', 'color': '#aaa',
        'text-valign': 'bottom', 'text-margin-y': 4, 'opacity': 'data(opacity)',
        'border-width': 1, 'border-color': '#333',
        'text-max-width': '80px', 'text-wrap': 'ellipsis'
      }},
      { selector: 'edge', style: {
        'line-color': 'data(color)', 'width': 'data(width)',
        'curve-style': 'bezier', 'opacity': 0.6,
        'target-arrow-shape': 'triangle', 'target-arrow-color': 'data(color)',
        'arrow-scale': 0.6
      }},
      { selector: 'node:selected', style: {
        'border-width': 3, 'border-color': '#8b5cf6', 'z-index': 999
      }}
    ],
    layout: { name: 'cose', nodeRepulsion: 8000, idealEdgeLength: 100, animate: false },
    minZoom: 0.1, maxZoom: 5
  });

  cy.on('tap', 'node', async (e) => {
    const memId = e.target.data('memId');
    const res = await fetch(BASE + '/memory/' + memId).then(r => r.json());
    showDetail(res);
  });

  cy.on('tap', (e) => { if (e.target === cy) hideDetail(); });
}

function showDetail(mem) {
  document.getElementById('sidebar-empty').style.display = 'none';
  const el = document.getElementById('sidebar-content');
  el.style.display = 'block';

  let badges = '<span class="badge ' + mem.category + '">' + mem.category + '</span>';
  if (mem.is_static) badges += '<span class="badge static">static</span>';
  if (mem.is_forgotten) badges += '<span class="badge forgotten">forgotten</span>';
  if (mem.is_latest) badges += '<span class="badge latest">latest</span>';

  let html = '<div class="detail-card"><div class="label">ID #' + mem.id + ' v' + (mem.version||1) + ' | source_count: ' + (mem.source_count||1) + '</div>' + badges + '</div>';
  html += '<div class="detail-card"><div class="label">Content</div><div class="value content">' + escHtml(mem.content) + '</div></div>';
  html += '<div class="detail-card"><div class="label">Source</div><div class="value">' + (mem.source||'unknown') + '</div></div>';
  html += '<div class="detail-card"><div class="label">Importance</div><div class="value">' + mem.importance + '/10</div></div>';
  html += '<div class="detail-card"><div class="label">Created</div><div class="value">' + mem.created_at + '</div></div>';

  if (mem.forget_after) {
    html += '<div class="detail-card"><div class="label">Forget After</div><div class="value">' + mem.forget_after + (mem.forget_reason ? ' (' + escHtml(mem.forget_reason) + ')' : '') + '</div></div>';
  }

  if (mem.version_chain && mem.version_chain.length > 0) {
    html += '<div class="detail-card"><div class="label">Version Chain</div><div class="version-chain">';
    for (const v of mem.version_chain) {
      const cls = v.id === mem.id ? 'current' : '';
      html += '<div class="version-item ' + cls + '" onclick="loadMem(' + v.id + ')"><span class="ver">v' + v.version + '</span> #' + v.id + (v.is_latest ? ' (latest)' : '') + '<br><span style="color:#888;font-size:11px">' + escHtml(v.content.slice(0, 60)) + '</span></div>';
    }
    html += '</div></div>';
  }

  if (mem.links && mem.links.length > 0) {
    html += '<div class="detail-card"><div class="label">Links (' + mem.links.length + ')</div>';
    for (const l of mem.links) {
      html += '<div class="link-item" onclick="loadMem(' + l.id + ')"><span class="type ' + l.type + '">' + l.type + '</span> <span style="color:#666;font-size:10px">' + (l.similarity||0).toFixed(3) + '</span><div class="preview">#' + l.id + ' ' + escHtml(l.content.slice(0, 80)) + '</div></div>';
    }
    html += '</div>';
  }

  el.innerHTML = html;
}

async function loadMem(id) {
  const res = await fetch(BASE + '/memory/' + id).then(r => r.json());
  showDetail(res);
  cy.$('#n' + id).select();
}

function hideDetail() {
  document.getElementById('sidebar-empty').style.display = 'block';
  document.getElementById('sidebar-content').style.display = 'none';
}

function setFilter(f, btn) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderGraph();
}

async function doSearch() {
  const q = document.getElementById('search').value.trim();
  if (!q) { loadGraph(); return; }
  const res = await fetch(BASE + '/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: q, limit: 30, include_links: true, expand_relationships: true })
  }).then(r => r.json());

  if (res.results && res.results.length > 0) {
    cy.nodes().style('opacity', 0.15);
    cy.edges().style('opacity', 0.05);
    for (const r of res.results) {
      const node = cy.$('#n' + r.id);
      if (node.length) {
        node.style('opacity', 1);
        node.connectedEdges().style('opacity', 0.6);
        node.connectedEdges().connectedNodes().style('opacity', 0.5);
      }
    }
    showDetail(res.results[0]);
    const first = cy.$('#n' + res.results[0].id);
    if (first.length) { cy.animate({ center: { eles: first }, zoom: 2 }, { duration: 300 }); }
  }
}

document.getElementById('search').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
loadGraph();
<\/script>
</body>
</html>`;

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
    // WEB GUI
    // ========================================================================
    if ((url.pathname === "/" || url.pathname === "/gui") && method === "GET") {
      return new Response(GUI_HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
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
      const dbSize = Bun.file(DB_PATH).size;
      return json({
        status: "ok",
        version: 3,
        memories: memCount.count,
        embedded: embCount.count,
        unembedded: noEmbCount,
        links: linkCount.count,
        forgotten: forgottenCount.count,
        static: staticCount.count,
        versioned: versionedCount.count,
        conversations: convCount.count,
        messages: msgCount.count,
        embedding_model: EMBEDDING_MODEL,
        llm_model: LLM_MODEL,
        llm_configured: !!LLM_API_KEY,
        db_size_mb: Math.round(dbSize / 1048576 * 100) / 100,
      });
    }

    // ========================================================================
    // STORE — v3 with async fact extraction
    // ========================================================================

    if (url.pathname === "/store" && method === "POST") {
      try {
        const body = await req.json();
        const { content, category, source, session_id, importance } = body;
        if (!content || typeof content !== "string" || content.trim().length === 0) {
          return errorResponse("content is required and must be a non-empty string");
        }

        const imp = Math.max(1, Math.min(10, Number(importance) || DEFAULT_IMPORTANCE));

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

        let linked = 0;
        if (embArray) {
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

        return json({
          stored: true,
          id: result.id,
          created_at: result.created_at,
          importance: imp,
          linked,
          embedded: !!embBuffer,
          fact_extraction: LLM_API_KEY ? "queued" : "disabled",
        });
      } catch (e: any) {
        return errorResponse(`Failed to store: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // SEARCH — v3
    // ========================================================================

    if (url.pathname === "/search" && method === "POST") {
      try {
        const body = await req.json();
        const { query, limit, include_links, expand_relationships, latest_only } = body;
        if (!query || typeof query !== "string") return errorResponse("query is required");
        const results = await hybridSearch(
          query,
          Math.min(limit || 10, 50),
          include_links || false,
          expand_relationships ?? true,
          latest_only ?? true
        );
        return json({ results });
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
      const results = category ? listByCategory.all(category, limit) : listRecent.all(limit);
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

    if (url.pathname.startsWith("/memory/") && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      const memory = getMemoryWithoutEmbedding.get(id) as any;
      if (!memory) return errorResponse("Not found", 404);

      const links = getLinksFor.all(id, id) as Array<{
        id: number; similarity: number; type: string; content: string; category: string;
      }>;

      const rootId = memory.root_memory_id || memory.id;
      const chain = getVersionChain.all(rootId, rootId) as Array<{
        id: number; content: string; version: number; is_latest: boolean;
        created_at: string; source_count: number;
      }>;

      return json({
        ...memory,
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
        const profile = await generateProfile(summary);
        return json(profile);
      } catch (e: any) {
        return errorResponse(`Profile generation failed: ${e.message}`, 500);
      }
    }

    // ========================================================================
    // GRAPH DATA
    // ========================================================================

    if (url.pathname === "/graph" && method === "GET") {
      const memories = getAllMemoriesForGraph.all();
      const links = getAllLinksForGraph.all();
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
    // STATS — v3
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

sweepExpiredMemories();

console.log(`MegaMind v3 listening on ${HOST}:${PORT}`);
console.log(`Database: ${DB_PATH}`);
console.log(`Embedding model: ${EMBEDDING_MODEL} (${EMBEDDING_DIM}d)`);
console.log(`LLM: ${LLM_API_KEY ? `${LLM_MODEL} via ${LLM_URL}` : "not configured"}`);
console.log(`Auto-link: threshold=${AUTO_LINK_THRESHOLD}, max=${AUTO_LINK_MAX}`);
console.log(`Auto-forget sweep: every ${FORGET_SWEEP_INTERVAL / 1000}s`);
