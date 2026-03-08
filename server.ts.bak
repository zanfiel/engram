import { Database } from "bun:sqlite";
import { resolve } from "path";

const DATA_DIR = resolve(import.meta.dir, "data");
const DB_PATH = resolve(DATA_DIR, "memory.db");
const PORT = Number(process.env.ZANMEMORY_PORT || 4200);
const HOST = process.env.ZANMEMORY_HOST || "0.0.0.0";

// Ensure data directory exists
await Bun.$`mkdir -p ${DATA_DIR}`;

// Initialize database
const db = new Database(DB_PATH, { create: true });
db.exec("PRAGMA journal_mode=WAL");
db.exec("PRAGMA foreign_keys=ON");
db.exec("PRAGMA busy_timeout=5000");

// ============================================================================
// MEMORIES TABLE (existing — key-value notes, discoveries, decisions)
// ============================================================================

db.exec(`
  CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    source TEXT NOT NULL DEFAULT 'unknown',
    session_id TEXT,
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

// ============================================================================
// CONVERSATIONS TABLE (new — full conversation logs from every agent)
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
// PREPARED STATEMENTS — memories
// ============================================================================

const insertMemory = db.prepare(
  `INSERT INTO memories (content, category, source, session_id) VALUES (?, ?, ?, ?) RETURNING id, created_at`
);
const searchMemories = db.prepare(
  `SELECT m.id, m.content, m.category, m.source, m.session_id, m.created_at
   FROM memories_fts f
   JOIN memories m ON f.rowid = m.id
   WHERE memories_fts MATCH ?
   ORDER BY rank
   LIMIT ?`
);
const listRecent = db.prepare(
  `SELECT id, content, category, source, session_id, created_at
   FROM memories ORDER BY created_at DESC LIMIT ?`
);
const listByCategory = db.prepare(
  `SELECT id, content, category, source, session_id, created_at
   FROM memories WHERE category = ? ORDER BY created_at DESC LIMIT ?`
);
const deleteMemory = db.prepare(`DELETE FROM memories WHERE id = ?`);
const getMemory = db.prepare(`SELECT * FROM memories WHERE id = ?`);

// ============================================================================
// PREPARED STATEMENTS — conversations
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

// Touch conversation updated_at when a message is added
const touchConversation = db.prepare(
  `UPDATE conversations SET updated_at = datetime('now') WHERE id = ?`
);

// ============================================================================
// BULK INSERT for storing entire conversations at once
// ============================================================================

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
// HELPERS
// ============================================================================

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
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
// SERVER
// ============================================================================

const server = Bun.serve({
  port: PORT,
  hostname: HOST,
  async fetch(req) {
    const url = new URL(req.url);
    const method = req.method;

    // ========================================================================
    // HEALTH
    // ========================================================================
    if (url.pathname === "/health" && method === "GET") {
      const memCount = db.prepare("SELECT COUNT(*) as count FROM memories").get() as { count: number };
      const convCount = db.prepare("SELECT COUNT(*) as count FROM conversations").get() as { count: number };
      const msgCount = db.prepare("SELECT COUNT(*) as count FROM messages").get() as { count: number };
      const dbSize = Bun.file(DB_PATH).size;
      return json({
        status: "ok",
        memories: memCount.count,
        conversations: convCount.count,
        messages: msgCount.count,
        db_size_mb: Math.round(dbSize / 1048576 * 100) / 100,
      });
    }

    // ========================================================================
    // MEMORIES (existing endpoints — unchanged)
    // ========================================================================

    if (url.pathname === "/store" && method === "POST") {
      try {
        const body = await req.json();
        const { content, category, source, session_id } = body;
        if (!content || typeof content !== "string" || content.trim().length === 0) {
          return errorResponse("content is required and must be a non-empty string");
        }
        const result = insertMemory.get(
          content.trim(),
          (category || "general").trim(),
          (source || "unknown").trim(),
          session_id || null
        ) as { id: number; created_at: string };
        return json({ stored: true, id: result.id, created_at: result.created_at });
      } catch (e: any) {
        return errorResponse(`Failed to store: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/search" && method === "POST") {
      try {
        const body = await req.json();
        const { query, limit } = body;
        if (!query || typeof query !== "string") return errorResponse("query is required");
        const sanitized = sanitizeFTS(query);
        if (!sanitized) return json({ results: [] });
        const results = searchMemories.all(sanitized, Math.min(limit || 20, 100));
        return json({ results });
      } catch (e: any) {
        return errorResponse(`Search failed: ${e.message}`, 500);
      }
    }

    if (url.pathname === "/list" && method === "GET") {
      const limit = Math.min(Number(url.searchParams.get("limit") || 20), 100);
      const category = url.searchParams.get("category");
      const results = category ? listByCategory.all(category, limit) : listRecent.all(limit);
      return json({ results });
    }

    if (url.pathname.startsWith("/memory/") && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      const memory = getMemory.get(id);
      if (!memory) return errorResponse("Not found", 404);
      return json(memory);
    }

    if (url.pathname.startsWith("/memory/") && method === "DELETE") {
      const id = Number(url.pathname.split("/")[2]);
      if (isNaN(id)) return errorResponse("Invalid id");
      deleteMemory.run(id);
      return json({ deleted: true, id });
    }

    // ========================================================================
    // CONVERSATIONS
    // ========================================================================

    // POST /conversations — create a new conversation
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

    // GET /conversations — list conversations (optional ?agent=&limit=)
    if (url.pathname === "/conversations" && method === "GET") {
      const limit = Math.min(Number(url.searchParams.get("limit") || 50), 500);
      const agent = url.searchParams.get("agent");
      const results = agent
        ? listConversationsByAgent.all(agent, limit)
        : listConversations.all(limit);
      return json({ results });
    }

    // GET /conversations/:id — get conversation with messages
    if (/^\/conversations\/\d+$/.test(url.pathname) && method === "GET") {
      const id = Number(url.pathname.split("/")[2]);
      const conv = getConversation.get(id);
      if (!conv) return errorResponse("Not found", 404);
      const limit = Math.min(Number(url.searchParams.get("limit") || 10000), 100000);
      const offset = Number(url.searchParams.get("offset") || 0);
      const msgs = getMessages.all(id, limit, offset);
      return json({ conversation: conv, messages: msgs });
    }

    // PATCH /conversations/:id — update title/metadata
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

    // DELETE /conversations/:id — delete conversation and all messages
    if (/^\/conversations\/\d+$/.test(url.pathname) && method === "DELETE") {
      const id = Number(url.pathname.split("/")[2]);
      deleteConversation.run(id);
      return json({ deleted: true, id });
    }

    // ========================================================================
    // MESSAGES
    // ========================================================================

    // POST /conversations/:id/messages — add message(s) to conversation
    if (/^\/conversations\/\d+\/messages$/.test(url.pathname) && method === "POST") {
      try {
        const convId = Number(url.pathname.split("/")[2]);
        const conv = getConversation.get(convId);
        if (!conv) return errorResponse("Conversation not found", 404);

        const body = await req.json();

        // Accept single message or array of messages
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
    // BULK STORE — create conversation + all messages in one shot
    // ========================================================================

    // POST /conversations/bulk — store an entire conversation at once
    if (url.pathname === "/conversations/bulk" && method === "POST") {
      try {
        const body = await req.json();
        const { agent, session_id, title, metadata, messages: msgs } = body;
        if (!agent) return errorResponse("agent is required");
        if (!msgs || !Array.isArray(msgs) || msgs.length === 0) {
          return errorResponse("messages array is required and must not be empty");
        }

        const conv = bulkInsertConvo(
          agent.trim(),
          session_id || null,
          title || null,
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

    // ========================================================================
    // UPSERT — find or create conversation by agent+session_id, add messages
    // ========================================================================

    // POST /conversations/upsert — find existing or create, then append messages
    if (url.pathname === "/conversations/upsert" && method === "POST") {
      try {
        const body = await req.json();
        const { agent, session_id, title, metadata, messages: msgs } = body;
        if (!agent) return errorResponse("agent is required");
        if (!session_id) return errorResponse("session_id is required for upsert");

        // Find existing conversation for this agent + session
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
    // SEARCH MESSAGES across all conversations
    // ========================================================================

    // POST /messages/search — full-text search across all message content
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
    // STATS
    // ========================================================================

    if (url.pathname === "/stats" && method === "GET") {
      const memCount = db.prepare("SELECT COUNT(*) as count FROM memories").get() as { count: number };
      const convCount = db.prepare("SELECT COUNT(*) as count FROM conversations").get() as { count: number };
      const msgCount = db.prepare("SELECT COUNT(*) as count FROM messages").get() as { count: number };
      const agents = db.prepare(
        `SELECT agent, COUNT(*) as conversations, 
          (SELECT COUNT(*) FROM messages m JOIN conversations c2 ON m.conversation_id = c2.id WHERE c2.agent = c.agent) as total_messages
         FROM conversations c GROUP BY agent ORDER BY total_messages DESC`
      ).all();
      const dbSize = Bun.file(DB_PATH).size;
      return json({
        memories: memCount.count,
        conversations: convCount.count,
        messages: msgCount.count,
        agents,
        db_size_mb: Math.round(dbSize / 1048576 * 100) / 100,
        db_path: DB_PATH,
      });
    }

    return errorResponse("Not found", 404);
  },
});

console.log(`MegaMind memory server listening on ${HOST}:${PORT}`);
console.log(`Database: ${DB_PATH}`);
