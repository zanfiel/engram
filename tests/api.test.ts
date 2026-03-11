/**
 * Engram API Test Suite — v5.4
 *
 * Node 22 built-in test runner, zero dependencies.
 * Server must be running with ENGRAM_OPEN_ACCESS=1.
 *
 * Run:  ENGRAM_TEST_URL=http://localhost:4200 node --experimental-strip-types --test tests/api.test.ts
 */

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";

const BASE = process.env.ENGRAM_TEST_URL || "http://127.0.0.1:4200";
const API_KEY = process.env.ENGRAM_TEST_KEY || "";
const h: Record<string, string> = { "Content-Type": "application/json" };
if (API_KEY) h["Authorization"] = `Bearer ${API_KEY}`;

const memIds: number[] = [];

async function api(method: string, path: string, body?: unknown) {
  const resp = await fetch(`${BASE}${path}`, { method, headers: h, body: body ? JSON.stringify(body) : undefined });
  const text = await resp.text();
  let data: any;
  try { data = JSON.parse(text); } catch { data = text; }
  return { status: resp.status, data };
}

// ── Health & Stats ──────────────────────────────────────────────────────────

describe("Health & Stats", () => {
  it("GET /health", async () => {
    const { status, data } = await api("GET", "/health");
    assert.equal(status, 200);
    assert.equal(data.status, "ok");
    assert.equal(typeof data.memories, "number");
    assert.ok(data.features.fsrs6 === true);
    assert.ok(data.version >= 5);
    assert.ok(data.db_size_mb > 0);
  });

  it("GET /stats", async () => {
    const { status, data } = await api("GET", "/stats");
    assert.equal(status, 200);
    assert.equal(typeof data.memories, "object"); // nested: {total, embedded, ...}
    assert.equal(typeof data.memories.total, "number");
    assert.equal(typeof data.conversations, "number");
    assert.equal(typeof data.db_size_mb, "number");
  });
});

// ── Store & Retrieve ────────────────────────────────────────────────────────

describe("Store & Retrieve", () => {
  it("POST /store — basic", async () => {
    const { status, data } = await api("POST", "/store", {
      content: "Test memory: the sky is blue on clear days",
      category: "discovery", source: "test-suite",
    });
    assert.equal(status, 200);
    assert.equal(data.stored, true);
    assert.ok(data.id > 0);
    memIds.push(data.id);
  });

  it("POST /store — rejects empty", async () => {
    const { status } = await api("POST", "/store", { content: "", category: "task" });
    assert.ok(status >= 400);
  });

  it("POST /store — importance + session_id", async () => {
    const { status, data } = await api("POST", "/store", {
      content: "Test: importance override check",
      category: "task", source: "test-suite", importance: 9.5, session_id: "test-001",
    });
    assert.equal(status, 200);
    assert.ok(data.id > 0);
    memIds.push(data.id);
  });

  it("POST /store — tags", async () => {
    const { status, data } = await api("POST", "/store", {
      content: "Test: tagged item for filtering",
      category: "state", source: "test-suite", tags: ["test-alpha", "test-beta"],
    });
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.tags));
    memIds.push(data.id);
  });

  it("POST /store — is_static", async () => {
    const { status, data } = await api("POST", "/store", {
      content: "Test: static fact never decays",
      category: "discovery", source: "test-suite", is_static: true,
    });
    assert.equal(status, 200);
    memIds.push(data.id);
  });

  it("POST /store — forget_after TTL", async () => {
    const { status, data } = await api("POST", "/store", {
      content: "Test: ephemeral with TTL",
      category: "task", source: "test-suite",
      forget_after: new Date(Date.now() + 86400000).toISOString(),
    });
    assert.equal(status, 200);
    memIds.push(data.id);
  });

  it("GET /memory/:id — full retrieval", async () => {
    const { status, data } = await api("GET", `/memory/${memIds[0]}`);
    assert.equal(status, 200);
    assert.equal(data.id, memIds[0]);
    assert.ok(data.content.includes("the sky is blue"));
    assert.equal(data.category, "discovery");
    assert.equal(data.source, "test-suite");
    assert.equal(typeof data.created_at, "string");
    assert.equal(typeof data.importance, "number");
    assert.equal(typeof data.version, "number");
    // SQLite returns 0/1, not boolean
    assert.ok(data.is_static !== undefined);
    assert.ok(data.is_forgotten !== undefined);
  });

  it("GET /memory/999999 — 404", async () => {
    const { status } = await api("GET", "/memory/999999");
    assert.equal(status, 404);
  });
});

// ── Search ──────────────────────────────────────────────────────────────────

describe("Search", () => {
  it("POST /search — finds memories, {results} wrapper", async () => {
    const { status, data } = await api("POST", "/search", { query: "sky is blue clear days" });
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
    assert.ok(data.results.length > 0);
    assert.ok(data.results.some((m: any) => m.content.includes("the sky is blue")));
  });

  it("POST /search — limit", async () => {
    const { status, data } = await api("POST", "/search", { query: "test", limit: 2 });
    assert.equal(status, 200);
    assert.ok(data.results.length <= 2);
  });

  it("POST /search — tag filter", async () => {
    const { status, data } = await api("POST", "/search", { query: "tagged item", tag: "test-alpha" });
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
  });

  it("POST /search — rejects empty query", async () => {
    const { status } = await api("POST", "/search", { query: "" });
    assert.ok(status >= 400);
  });
});

// ── List ────────────────────────────────────────────────────────────────────

describe("List", () => {
  it("GET /list — {results} wrapper", async () => {
    const { status, data } = await api("GET", "/list?limit=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
    assert.ok(data.results.length <= 5);
    if (data.results.length > 0) {
      assert.ok(data.results[0].id > 0);
      assert.ok(typeof data.results[0].content === "string");
    }
  });

  it("GET /list — category filter", async () => {
    const { status, data } = await api("GET", "/list?category=discovery&limit=5");
    assert.equal(status, 200);
    for (const m of data.results) assert.equal(m.category, "discovery");
  });

  it("GET /list — source filter", async () => {
    const { status, data } = await api("GET", "/list?source=test-suite&limit=5");
    assert.equal(status, 200);
    for (const m of data.results) assert.equal(m.source, "test-suite");
  });
});

// ── Update & Versioning ─────────────────────────────────────────────────────

describe("Update & Versioning", () => {
  it("POST /memory/:id/update — new version", async () => {
    const id = memIds[0];
    const { status, data } = await api("POST", `/memory/${id}/update`, {
      content: "Updated: the sky has many colors at sunset",
      category: "discovery",
    });
    assert.equal(status, 200);
    assert.equal(data.updated, true);
    assert.ok(data.new_id > id);
    assert.ok(data.version >= 2);
    assert.equal(data.root_id, id);
    memIds.push(data.new_id);
  });

  it("GET /versions/:root_id — {root_id, chain}", async () => {
    const { status, data } = await api("GET", `/versions/${memIds[0]}`);
    assert.equal(status, 200);
    assert.equal(data.root_id, memIds[0]);
    assert.ok(Array.isArray(data.chain));
    assert.ok(data.chain.length >= 2);
  });
});

// ── Forget / Archive / Delete ───────────────────────────────────────────────

describe("Forget / Archive / Delete", () => {
  let targetId: number;

  before(async () => {
    const { data } = await api("POST", "/store", {
      content: "Test: lifecycle target for forget/archive/delete",
      category: "task", source: "test-suite",
    });
    targetId = data.id;
  });

  it("POST /memory/:id/forget", async () => {
    const { status } = await api("POST", `/memory/${targetId}/forget`);
    assert.equal(status, 200);
    const { data } = await api("GET", `/memory/${targetId}`);
    assert.ok(data.is_forgotten == true);
  });

  it("POST /memory/:id/archive", async () => {
    const { status } = await api("POST", `/memory/${targetId}/archive`);
    assert.equal(status, 200);
    const { data } = await api("GET", `/memory/${targetId}`);
    assert.ok(data.is_archived == true);
  });

  it("POST /memory/:id/unarchive", async () => {
    const { status } = await api("POST", `/memory/${targetId}/unarchive`);
    assert.equal(status, 200);
    const { data } = await api("GET", `/memory/${targetId}`);
    assert.ok(data.is_archived == false);
  });

  it("DELETE /memory/:id", async () => {
    const { status } = await api("DELETE", `/memory/${targetId}`);
    assert.equal(status, 200);
    assert.equal((await api("GET", `/memory/${targetId}`)).status, 404);
  });
});

// ── Tags ────────────────────────────────────────────────────────────────────

describe("Tags", () => {
  it("PUT /memory/:id/tags", async () => {
    const { status } = await api("PUT", `/memory/${memIds[2]}/tags`, { tags: ["updated-tag", "another-tag"] });
    assert.equal(status, 200);
  });

  it("GET /tags — {tags} wrapper", async () => {
    const { status, data } = await api("GET", "/tags");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.tags));
  });

  it("POST /tags/search — {results, tag}", async () => {
    const { status, data } = await api("POST", "/tags/search", { tag: "updated-tag" });
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
    assert.equal(data.tag, "updated-tag");
  });
});

// ── Recall & Profile ────────────────────────────────────────────────────────

describe("Recall & Profile", () => {
  it("POST /recall", async () => {
    const { status, data } = await api("POST", "/recall", { query: "sky colors" });
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.memories) || Array.isArray(data.results));
  });

  it("GET /profile", async () => {
    const { status, data } = await api("GET", "/profile");
    assert.equal(status, 200);
    assert.ok(typeof data === "object");
  });
});

// ── Graph ───────────────────────────────────────────────────────────────────

describe("Graph", () => {
  it("GET /graph — {nodes, edges}", async () => {
    const { status, data } = await api("GET", "/graph");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.nodes));
    assert.ok(Array.isArray(data.edges));
  });

  it("GET /graph/raw — {memories, links}", async () => {
    const { status, data } = await api("GET", "/graph/raw");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.memories));
    assert.ok(Array.isArray(data.links));
  });

  it("GET /links/:id — {memory_id, links}", async () => {
    const { status, data } = await api("GET", `/links/${memIds[0]}`);
    assert.equal(status, 200);
    assert.equal(data.memory_id, memIds[0]);
    assert.ok(Array.isArray(data.links));
  });
});

// ── Duplicates ──────────────────────────────────────────────────────────────

describe("Duplicates", () => {
  it("GET /duplicates — {threshold, clusters}", async () => {
    const { status, data } = await api("GET", "/duplicates?threshold=0.9&limit=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.clusters));
    assert.equal(typeof data.threshold, "number");
  });

  it("POST /deduplicate — dry run", async () => {
    const { status, data } = await api("POST", "/deduplicate", { threshold: 0.98, dry_run: true });
    assert.equal(status, 200);
    assert.equal(data.dry_run, true);
    assert.equal(typeof data.clusters_found, "number");
  });
});

// ── Conversations ───────────────────────────────────────────────────────────

describe("Conversations", () => {
  it("POST /conversations — create", async () => {
    const { status, data } = await api("POST", "/conversations", {
      session_id: "test-conv-" + Date.now(),
      agent: "test-agent", model: "test-model",
      started_at: new Date().toISOString(),
    });
    assert.equal(status, 200);
    assert.ok(data.id > 0);
  });

  it("GET /conversations — {results}", async () => {
    const { status, data } = await api("GET", "/conversations?limit=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
  });

  it("POST /conversations/bulk — single conv with messages", async () => {
    const { status, data } = await api("POST", "/conversations/bulk", {
      agent: "test-agent",
      session_id: "test-bulk-" + Date.now(),
      model: "test-model",
      messages: [
        { role: "user", content: "Bulk test hello " + Date.now() },
        { role: "assistant", content: "Bulk test reply" },
      ],
    });
    assert.equal(status, 200);
    assert.ok(data.id > 0);
    assert.ok(data.messages >= 2);
  });

  it("POST /messages/search — {results}", async () => {
    const { status, data } = await api("POST", "/messages/search", { query: "Bulk test" });
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
  });
});

// ── Episodes ────────────────────────────────────────────────────────────────

describe("Episodes", () => {
  it("POST /episodes — create", async () => {
    const ids = memIds.slice(0, 2).filter(id => id > 0);
    if (ids.length < 2) return;
    const { status, data } = await api("POST", "/episodes", { name: "Test episode", memory_ids: ids });
    assert.equal(status, 200);
    assert.ok(data.id > 0);
  });

  it("GET /episodes — {episodes}", async () => {
    const { status, data } = await api("GET", "/episodes?limit=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.episodes));
  });
});

// ── Decay & FSRS ────────────────────────────────────────────────────────────

describe("Decay & FSRS", () => {
  it("GET /decay/scores — {memories}", async () => {
    const { status, data } = await api("GET", "/decay/scores?limit=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.memories));
  });

  it("POST /decay/refresh", async () => {
    const { status } = await api("POST", "/decay/refresh");
    assert.equal(status, 200);
  });

  it("GET /fsrs/state", async () => {
    const { status, data } = await api("GET", `/fsrs/state?id=${memIds[0]}`);
    assert.equal(status, 200);
    assert.ok(typeof data === "object");
  });

  it("POST /fsrs/review — id + grade", async () => {
    const { status, data } = await api("POST", "/fsrs/review", { id: memIds[0], grade: 3 });
    assert.equal(status, 200);
    assert.ok(data.fsrs !== undefined);
  });
});

// ── Pack & Prompt ───────────────────────────────────────────────────────────

describe("Pack & Prompt", () => {
  it("POST /pack — {packed, tokens_estimated}", async () => {
    const { status, data } = await api("POST", "/pack", { query: "test", max_tokens: 2000 });
    assert.equal(status, 200);
    assert.ok(typeof data.packed === "string");
    assert.ok(typeof data.tokens_estimated === "number");
  });

  it("GET /prompt — {prompt}", async () => {
    const { status, data } = await api("GET", "/prompt?template=recall&query=test");
    assert.equal(status, 200);
    assert.ok(typeof data.prompt === "string");
  });
});

// ── Smart Context ───────────────────────────────────────────────────────────

describe("Smart Context", () => {
  it("POST /context — {context, token_estimate}", async () => {
    const { status, data } = await api("POST", "/context", { query: "test", max_tokens: 2000 });
    assert.equal(status, 200);
    assert.ok(typeof data.context === "string");
    assert.ok(typeof data.token_estimate === "number");
  });
});

// ── Time Travel ─────────────────────────────────────────────────────────────

describe("Time Travel", () => {
  it("POST /timetravel — rejects missing as_of", async () => {
    const { status } = await api("POST", "/timetravel", {});
    assert.ok(status >= 400);
  });

  it("POST /timetravel — valid as_of", async () => {
    const { status, data } = await api("POST", "/timetravel", { as_of: new Date().toISOString() });
    assert.equal(status, 200);
    assert.ok(typeof data === "object");
  });
});

// ── Contradictions ──────────────────────────────────────────────────────────

describe("Contradictions", () => {
  it("GET /contradictions — {contradictions, total}", async () => {
    const { status, data } = await api("GET", "/contradictions?limit=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.contradictions));
    assert.equal(typeof data.total, "number");
  });
});

// ── Export & Import ─────────────────────────────────────────────────────────

describe("Export & Import", () => {
  it("GET /export — {version, exported_at, memories}", async () => {
    const { status, data } = await api("GET", "/export?format=json");
    assert.equal(status, 200);
    assert.ok(typeof data.version === "string"); // "engram-v4"
    assert.ok(Array.isArray(data.memories));
    assert.ok(typeof data.exported_at === "string");
  });

  it("POST /import", async () => {
    const { status, data } = await api("POST", "/import", {
      memories: [{ content: "Imported test memory " + Date.now(), category: "task", source: "import-test" }],
    });
    assert.equal(status, 200);
    assert.equal(data.imported, 1);
    assert.equal(data.failed, 0);
  });
});

// ── Sweep & Backfill ────────────────────────────────────────────────────────

describe("Sweep & Backfill", () => {
  it("POST /sweep", async () => {
    assert.equal((await api("POST", "/sweep")).status, 200);
  });

  it("POST /backfill", async () => {
    assert.equal((await api("POST", "/backfill")).status, 200);
  });
});

// ── Webhooks ────────────────────────────────────────────────────────────────

describe("Webhooks", () => {
  it("POST /webhooks — register", async () => {
    const { status, data } = await api("POST", "/webhooks", {
      url: "https://httpbin.org/post", events: ["memory.created"],
    });
    assert.equal(status, 200);
    assert.equal(data.created, true);
    assert.ok(data.id > 0);
  });

  it("GET /webhooks — {webhooks}", async () => {
    const { status, data } = await api("GET", "/webhooks");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.webhooks));
  });
});

// ── Sync ────────────────────────────────────────────────────────────────────

describe("Sync", () => {
  it("GET /sync/changes — {changes, count}", async () => {
    const since = new Date(Date.now() - 86400000).toISOString();
    const { status, data } = await api("GET", `/sync/changes?since=${encodeURIComponent(since)}`);
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.changes));
    assert.equal(typeof data.count, "number");
  });
});

// ── Inbox ───────────────────────────────────────────────────────────────────

describe("Inbox", () => {
  it("GET /inbox — {pending, count}", async () => {
    const { status, data } = await api("GET", "/inbox?limit=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.pending));
    assert.equal(typeof data.count, "number");
  });
});

// ── Audit ───────────────────────────────────────────────────────────────────

describe("Audit", () => {
  it("GET /audit — {entries, total}", async () => {
    const { status, data } = await api("GET", "/audit?limit=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.entries));
    assert.equal(typeof data.total, "number");
  });
});

// ── Spaces ──────────────────────────────────────────────────────────────────

describe("Spaces", () => {
  let spaceId: number;

  it("POST /spaces — create", async () => {
    const { status, data } = await api("POST", "/spaces", {
      name: "test-space-" + Date.now(), description: "test",
    });
    assert.equal(status, 200);
    assert.ok(data.id > 0);
    spaceId = data.id;
  });

  it("GET /spaces — {spaces}", async () => {
    const { status, data } = await api("GET", "/spaces");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.spaces));
    assert.ok(data.spaces.length >= 1);
  });

  it("DELETE /spaces/:id", async () => {
    if (!spaceId) return;
    assert.equal((await api("DELETE", `/spaces/${spaceId}`)).status, 200);
  });
});

// ── API Keys ────────────────────────────────────────────────────────────────

describe("API Keys", () => {
  let keyId: number;

  it("POST /keys — eg_ prefix", async () => {
    const { status, data } = await api("POST", "/keys", { name: "test-key-" + Date.now(), scopes: "read,write" });
    assert.equal(status, 200);
    assert.ok(data.key.startsWith("eg_"));
  });

  it("GET /keys — {keys}", async () => {
    const { status, data } = await api("GET", "/keys");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.keys));
    if (data.keys.length > 0) keyId = data.keys[data.keys.length - 1].id;
  });

  it("DELETE /keys/:id", async () => {
    if (!keyId) return;
    assert.equal((await api("DELETE", `/keys/${keyId}`)).status, 200);
  });
});

// ── Users ───────────────────────────────────────────────────────────────────

describe("Users", () => {
  it("POST /users — {username}", async () => {
    const { status, data } = await api("POST", "/users", { username: "test-" + Date.now() });
    assert.equal(status, 200);
    assert.ok(data.id > 0);
  });

  it("GET /users — {users}", async () => {
    const { status, data } = await api("GET", "/users");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.users));
  });
});

// ── Digests ─────────────────────────────────────────────────────────────────

describe("Digests", () => {
  it("POST /digests — create schedule", async () => {
    const { status, data } = await api("POST", "/digests", {
      name: "test-" + Date.now(), cron: "0 9 * * *",
      webhook_url: "https://httpbin.org/post",
    });
    assert.equal(status, 200);
    assert.ok(data.id > 0);
  });

  it("GET /digests — {digests}", async () => {
    const { status, data } = await api("GET", "/digests");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.digests));
  });
});

// ── Entities & Projects ─────────────────────────────────────────────────────

describe("Entities & Projects", () => {
  it("POST /entities", async () => {
    const { status, data } = await api("POST", "/entities", {
      name: "Test Entity " + Date.now(), type: "person", description: "test",
    });
    assert.equal(status, 200);
    assert.equal(data.created, true);
  });

  it("GET /entities — {entities, count}", async () => {
    const { status, data } = await api("GET", "/entities");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.entities));
    assert.equal(typeof data.count, "number");
  });

  it("POST /projects", async () => {
    const { status, data } = await api("POST", "/projects", {
      name: "Test Project " + Date.now(), description: "test",
    });
    assert.equal(status, 200);
    assert.equal(data.created, true);
  });

  it("GET /projects — {projects, count}", async () => {
    const { status, data } = await api("GET", "/projects");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.projects));
    assert.equal(typeof data.count, "number");
  });
});

// ── URL Ingest ──────────────────────────────────────────────────────────────

describe("URL Ingest", () => {
  it("POST /ingest — doesn't crash", async () => {
    const { status } = await api("POST", "/ingest", { url: "https://example.com" });
    assert.ok(status >= 200);
  });
});

// ── Backup ──────────────────────────────────────────────────────────────────

describe("Backup", () => {
  it("GET /backup — SQLite binary", async () => {
    const resp = await fetch(`${BASE}/backup`, { headers: h });
    assert.equal(resp.status, 200);
    const buf = await resp.arrayBuffer();
    assert.ok(buf.byteLength > 0);
    assert.equal(String.fromCharCode(...new Uint8Array(buf.slice(0, 6))), "SQLite");
  });
});

// ── GUI ─────────────────────────────────────────────────────────────────────

describe("GUI", () => {
  it("GET /gui — HTML", async () => {
    const resp = await fetch(`${BASE}/gui`);
    assert.equal(resp.status, 200);
    const text = await resp.text();
    assert.ok(text.includes("<html") || text.includes("<!DOCTYPE"));
  });
});

// ── Cleanup ─────────────────────────────────────────────────────────────────

describe("Cleanup", () => {
  it("delete test memories", async () => {
    for (const id of memIds) await api("DELETE", `/memory/${id}`).catch(() => {});
    assert.ok(true);
  });
});
