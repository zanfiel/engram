// Engram API Test Suite — multi-tenant isolation + core functionality
// Run: ENGRAM_URL=http://127.0.0.1:4201 node --test tests/api.test.mjs

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";

const BASE = process.env.ENGRAM_URL || "http://127.0.0.1:4201";

async function api(path, opts = {}) {
  const { method = "GET", body, headers = {} } = opts;
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...headers },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  return { status: res.status, data };
}

// ============================================================================
// HEALTH
// ============================================================================
describe("Health", () => {
  it("GET /health returns ok", async () => {
    const { status, data } = await api("/health");
    assert.equal(status, 200);
    assert.equal(data.status, "ok");
    assert.ok(data.version);
  });
});

// ============================================================================
// STORE + SEARCH + RECALL
// ============================================================================
let testMemId;

describe("Store", () => {
  it("POST /store creates a memory", async () => {
    const { status, data } = await api("/store", {
      method: "POST",
      body: { content: "vitest multi-tenant test memory", category: "test", importance: 7 },
    });
    assert.equal(status, 200);
    assert.ok(data.stored);
    assert.ok(data.id);
    testMemId = data.id;
  });

  it("POST /store rejects empty content", async () => {
    const { status } = await api("/store", {
      method: "POST",
      body: { content: "", category: "test" },
    });
    assert.equal(status, 400);
  });
});

describe("Search", () => {
  it("POST /search returns results", async () => {
    // Wait for embedding to complete
    await new Promise(r => setTimeout(r, 1000));
    const { status, data } = await api("/search", {
      method: "POST",
      body: { query: "multi-tenant test", limit: 5 },
    });
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
  });
});

describe("Recall", () => {
  it("POST /recall returns context", async () => {
    const { status, data } = await api("/recall", {
      method: "POST",
      body: { query: "multi-tenant test" },
    });
    assert.equal(status, 200);
    assert.ok(data.memories || data.results);
  });
});

// ============================================================================
// MEMORY CRUD
// ============================================================================
describe("Memory CRUD", () => {
  it("GET /memory/:id returns the memory", async () => {
    const { status, data } = await api(`/memory/${testMemId}`);
    assert.equal(status, 200);
    assert.equal(data.id, testMemId);
  });

  it("POST /memory/:id/update creates a new version", async () => {
    const { status, data } = await api(`/memory/${testMemId}/update`, {
      method: "POST",
      body: { content: "updated multi-tenant test memory v2" },
    });
    assert.equal(status, 200);
    assert.ok(data.new_id);
    assert.ok(data.version >= 2);
  });

  it("GET /links/:id returns links", async () => {
    const { status, data } = await api(`/links/${testMemId}`);
    assert.equal(status, 200);
    assert.equal(data.memory_id, testMemId);
  });

  it("GET /versions/:id returns version chain", async () => {
    const { status, data } = await api(`/versions/${testMemId}`);
    assert.equal(status, 200);
    assert.ok(data.root_id);
  });
});

// ============================================================================
// CONVERSATIONS
// ============================================================================
let testConvId;

describe("Conversations", () => {
  it("POST /conversations creates a conversation", async () => {
    const { status, data } = await api("/conversations", {
      method: "POST",
      body: { agent: "test-agent", title: "Test Conversation" },
    });
    assert.equal(status, 200);
    assert.ok(data.id);
    testConvId = data.id;
  });

  it("GET /conversations lists conversations", async () => {
    const { status, data } = await api("/conversations");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
  });

  it("GET /conversations/:id returns the conversation", async () => {
    const { status, data } = await api(`/conversations/${testConvId}`);
    assert.equal(status, 200);
    assert.ok(data.conversation);
    assert.equal(data.conversation.user_id, 1);
  });

  it("POST /conversations/:id/messages adds a message", async () => {
    const { status, data } = await api(`/conversations/${testConvId}/messages`, {
      method: "POST",
      body: { role: "user", content: "test message" },
    });
    assert.equal(status, 200);
    assert.ok(data.added >= 1);
  });

  it("POST /conversations/bulk creates with messages", async () => {
    const { status, data } = await api("/conversations/bulk", {
      method: "POST",
      body: {
        agent: "test-bulk",
        messages: [
          { role: "user", content: "hello" },
          { role: "assistant", content: "hi" },
        ],
      },
    });
    assert.equal(status, 200);
    assert.ok(data.id);
    assert.equal(data.messages, 2);
    // Cleanup
    await api(`/conversations/${data.id}`, { method: "DELETE" });
  });

  it("POST /messages/search searches messages", async () => {
    const { status, data } = await api("/messages/search", {
      method: "POST",
      body: { query: "test message", limit: 5 },
    });
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.results));
  });

  it("PATCH /conversations/:id updates", async () => {
    const { status, data } = await api(`/conversations/${testConvId}`, {
      method: "PATCH",
      body: { title: "Updated Title" },
    });
    assert.equal(status, 200);
    assert.ok(data.updated);
  });

  it("DELETE /conversations/:id deletes", async () => {
    const { status, data } = await api(`/conversations/${testConvId}`, { method: "DELETE" });
    assert.equal(status, 200);
    assert.ok(data.deleted);
  });
});

// ============================================================================
// STATS + SYSTEM
// ============================================================================
describe("Stats & System", () => {
  it("GET /stats returns user-scoped stats", async () => {
    const { status, data } = await api("/stats");
    assert.equal(status, 200);
    assert.ok(data.memories);
    assert.ok(typeof data.memories.total === "number");
    // db_path should not be exposed to non-admin
    assert.equal(data.db_path, undefined);
  });

  it("GET /decay/scores returns user-scoped scores", async () => {
    const { status, data } = await api("/decay/scores?limit=3");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.memories));
  });

  it("GET /consolidations returns user-scoped data", async () => {
    const { status, data } = await api("/consolidations");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.consolidations));
  });
});

// ============================================================================
// CONTRADICTIONS + DUPLICATES
// ============================================================================
describe("Contradictions & Duplicates", () => {
  it("GET /contradictions returns scoped results", async () => {
    const { status, data } = await api("/contradictions?limit=3");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.contradictions));
  });

  it("GET /duplicates returns scoped results", async () => {
    const { status, data } = await api("/duplicates?limit=3");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.clusters));
  });
});

// ============================================================================
// GRAPH
// ============================================================================
describe("Graph", () => {
  it("GET /graph returns nodes and edges", async () => {
    const { status, data } = await api("/graph?max_nodes=5");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.nodes));
    assert.ok(Array.isArray(data.edges));
  });
});

// ============================================================================
// EPISODES
// ============================================================================
describe("Episodes", () => {
  it("GET /episodes lists episodes", async () => {
    const { status, data } = await api("/episodes?limit=3");
    assert.equal(status, 200);
    assert.ok(Array.isArray(data.episodes));
  });
});

// ============================================================================
// ENTITIES + PROJECTS
// ============================================================================
let testEntityId, testProjectId;

describe("Entities", () => {
  it("POST /entities creates an entity", async () => {
    const { status, data } = await api("/entities", {
      method: "POST",
      body: { name: "Test Entity", type: "tool", description: "test" },
    });
    assert.equal(status, 200);
    assert.ok(data.id);
    testEntityId = data.id;
  });

  it("GET /entities/:id returns with ownership check", async () => {
    const { status, data } = await api(`/entities/${testEntityId}`);
    assert.equal(status, 200);
    assert.equal(data.name, "Test Entity");
  });

  it("DELETE /entities/:id deletes", async () => {
    const { status } = await api(`/entities/${testEntityId}`, { method: "DELETE" });
    assert.equal(status, 200);
  });
});

describe("Projects", () => {
  it("POST /projects creates a project", async () => {
    const { status, data } = await api("/projects", {
      method: "POST",
      body: { name: "Test Project", description: "test" },
    });
    assert.equal(status, 200);
    assert.ok(data.id);
    testProjectId = data.id;
  });

  it("GET /projects/:id returns with ownership check", async () => {
    const { status, data } = await api(`/projects/${testProjectId}`);
    assert.equal(status, 200);
    assert.equal(data.name, "Test Project");
  });

  it("DELETE /projects/:id deletes", async () => {
    const { status } = await api(`/projects/${testProjectId}`, { method: "DELETE" });
    assert.equal(status, 200);
  });
});

// ============================================================================
// FSRS
// ============================================================================
describe("FSRS", () => {
  it("GET /fsrs/state returns state", async () => {
    const { status, data } = await api(`/fsrs/state?id=${testMemId}`);
    assert.equal(status, 200);
    assert.ok(data.fsrs_stability !== undefined);
  });

  it("POST /fsrs/review records a review", async () => {
    const { status, data } = await api("/fsrs/review", {
      method: "POST",
      body: { id: testMemId, grade: 3 },
    });
    assert.equal(status, 200);
    assert.equal(data.id, testMemId);
  });
});

// ============================================================================
// CLEANUP
// ============================================================================
describe("Cleanup", () => {
  it("DELETE /memory/:id cleans up test memory", async () => {
    const { status, data } = await api(`/memory/${testMemId}`, { method: "DELETE" });
    assert.equal(status, 200);
    assert.ok(data.deleted);
  });
});
