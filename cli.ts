#!/usr/bin/env bun
/**
 * Engram CLI — Command-line interface for Engram memory server
 * Usage: engram <command> [options]
 */

const ENGRAM_URL = process.env.ENGRAM_URL || "http://localhost:4200";
const ENGRAM_KEY = process.env.ENGRAM_API_KEY || "";

const headers: Record<string, string> = { "Content-Type": "application/json" };
if (ENGRAM_KEY) headers["Authorization"] = `Bearer ${ENGRAM_KEY}`;

async function api(path: string, opts?: RequestInit): Promise<any> {
  const resp = await fetch(`${ENGRAM_URL}${path}`, { ...opts, headers: { ...headers, ...opts?.headers } });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

const [, , cmd, ...args] = process.argv;

const commands: Record<string, () => Promise<void>> = {
  // Store a memory
  async store() {
    const content = args.join(" ") || await readStdin();
    if (!content?.trim()) { console.error("Usage: engram store <content>"); process.exit(1); }

    const opts: any = { content: content.trim() };
    // Parse flags
    for (let i = 0; i < args.length; i++) {
      if (args[i] === "-c" || args[i] === "--category") { opts.category = args[++i]; }
      else if (args[i] === "-i" || args[i] === "--importance") { opts.importance = Number(args[++i]); }
      else if (args[i] === "-s" || args[i] === "--source") { opts.source = args[++i]; }
      else if (args[i] === "-t" || args[i] === "--tags") { opts.tags = args[++i].split(","); }
      else if (args[i] === "--static") { opts.is_static = true; }
    }
    // Re-join non-flag args as content
    const nonFlags = [];
    for (let i = 0; i < args.length; i++) {
      if (["-c", "--category", "-i", "--importance", "-s", "--source", "-t", "--tags"].includes(args[i])) { i++; continue; }
      if (args[i] === "--static") continue;
      nonFlags.push(args[i]);
    }
    if (nonFlags.length > 0) opts.content = nonFlags.join(" ");

    const result = await api("/store", { method: "POST", body: JSON.stringify(opts) });
    console.log(`✓ Stored #${result.id} (importance: ${result.importance}, linked: ${result.linked}${result.tags?.length ? `, tags: [${result.tags.join(", ")}]` : ""})`);
  },

  // Search memories
  async search() {
    const query = args.filter(a => !a.startsWith("-")).join(" ");
    if (!query) { console.error("Usage: engram search <query>"); process.exit(1); }
    const limit = getFlag("-n", "--limit") || "10";

    const result = await api("/search", {
      method: "POST",
      body: JSON.stringify({ query, limit: Number(limit), expand_relationships: true }),
    });

    for (const r of result.results) {
      const tags = r.tags?.length ? ` [${r.tags.join(", ")}]` : "";
      console.log(`  #${r.id} (${r.score.toFixed(3)}) [${r.category}]${tags}`);
      console.log(`    ${r.content}`);
      console.log();
    }
    console.log(`${result.results.length} results`);
  },

  // Smart recall
  async recall() {
    const context = args.filter(a => !a.startsWith("-")).join(" ");
    const limit = getFlag("-n", "--limit") || "20";
    const tokens = getFlag("--tokens");

    if (tokens) {
      // Use pack endpoint
      const result = await api("/pack", {
        method: "POST",
        body: JSON.stringify({ context, tokens: Number(tokens), format: "text" }),
      });
      console.log(result.packed);
      console.error(`\n--- ${result.memories_included} memories, ~${result.tokens_estimated} tokens (${result.utilization}) ---`);
    } else {
      const result = await api("/recall", {
        method: "POST",
        body: JSON.stringify({ context, limit: Number(limit) }),
      });
      for (const m of result.memories) {
        console.log(`  #${m.id} [${m.recall_source}] (${m.recall_score}) ${m.content.substring(0, 120)}`);
      }
      const b = result.breakdown;
      console.log(`\n${result.memories.length} memories: ${b.static} static, ${b.semantic} semantic, ${b.important} important, ${b.recent} recent`);
    }
  },

  // List memories
  async list() {
    const limit = getFlag("-n", "--limit") || "20";
    const category = getFlag("-c", "--category");
    const params = new URLSearchParams({ limit });
    if (category) params.set("category", category);
    const result = await api(`/list?${params}`);
    for (const m of result.results) {
      const age = timeAgo(m.created_at);
      console.log(`  #${m.id} [${m.category}] (imp=${m.importance}) ${age}`);
      console.log(`    ${m.content.substring(0, 120)}`);
      console.log();
    }
  },

  // Get a specific memory
  async get() {
    const id = args[0];
    if (!id) { console.error("Usage: engram get <id>"); process.exit(1); }
    const mem = await api(`/memory/${id}`);
    console.log(`Memory #${mem.id}`);
    console.log(`  Category:   ${mem.category}`);
    console.log(`  Importance: ${mem.importance}`);
    console.log(`  Confidence: ${(mem.confidence || 1).toFixed(2)}`);
    console.log(`  Decay:      ${(mem.decay_score || mem.importance).toFixed(3)}`);
    console.log(`  Accessed:   ${mem.access_count || 0} times`);
    if (mem.tags?.length) console.log(`  Tags:       [${mem.tags.join(", ")}]`);
    if (mem.episode) console.log(`  Episode:    #${mem.episode.id} (${mem.episode.title || mem.episode.session_id})`);
    console.log(`  Created:    ${mem.created_at}`);
    console.log(`  Content:\n    ${mem.content}`);
    if (mem.links?.length) {
      console.log(`  Links:`);
      for (const l of mem.links) {
        console.log(`    → #${l.id} [${l.type}] (${l.similarity.toFixed(3)}) ${l.content.substring(0, 80)}`);
      }
    }
  },

  // Tags
  async tags() {
    if (args[0] === "search" && args[1]) {
      const result = await api("/tags/search", { method: "POST", body: JSON.stringify({ tag: args[1] }) });
      for (const r of result.results) {
        console.log(`  #${r.id} [${r.tags?.join(", ")}] ${r.content.substring(0, 100)}`);
      }
    } else {
      const result = await api("/tags");
      console.log(result.tags.join(", "));
    }
  },

  // Episodes
  async episodes() {
    if (args[0] && !isNaN(Number(args[0]))) {
      const result = await api(`/episodes/${args[0]}`);
      console.log(`Episode #${result.id}: ${result.title || result.session_id || "untitled"}`);
      console.log(`  Agent: ${result.agent}, Memories: ${result.memory_count}`);
      if (result.memories) {
        for (const m of result.memories) {
          console.log(`    #${m.id} [${m.category}] ${m.content.substring(0, 100)}`);
        }
      }
    } else {
      const result = await api("/episodes?limit=20");
      for (const e of result.episodes) {
        console.log(`  #${e.id} [${e.agent}] ${e.title || e.session_id || "untitled"} (${e.memory_count} memories)`);
      }
    }
  },

  // Health check
  async health() {
    const h = await api("/health");
    console.log(`Engram v${h.version} — ${h.status}`);
    console.log(`  Memories: ${h.memories} (${h.embedded} embedded, ${h.tagged} tagged)`);
    console.log(`  Links: ${h.links} | Episodes: ${h.episodes} | Consolidations: ${h.consolidations}`);
    console.log(`  Features: ${Object.entries(h.features).filter(([, v]) => v).map(([k]) => k).join(", ")}`);
    console.log(`  DB: ${h.db_size_mb}MB | Model: ${h.embedding_model}`);
  },

  // Prompt template
  async prompt() {
    const format = getFlag("-f", "--format") || "raw";
    const tokens = getFlag("--tokens") || "4000";
    const context = args.filter(a => !a.startsWith("-") && !a.startsWith("--")).join(" ");
    const params = new URLSearchParams({ format, tokens });
    if (context) params.set("context", context);
    const result = await api(`/prompt?${params}`);
    console.log(result.prompt);
    console.error(`\n--- ${result.memories_included} memories, ~${result.tokens_estimated} tokens ---`);
  },

  // Sync
  async sync() {
    const remote = args[0];
    if (!remote) { console.error("Usage: engram sync <remote_url>"); process.exit(1); }

    // Pull changes from remote
    console.log(`Pulling from ${remote}...`);
    const remoteHeaders: Record<string, string> = { "Content-Type": "application/json" };
    if (ENGRAM_KEY) remoteHeaders["Authorization"] = `Bearer ${ENGRAM_KEY}`;

    const changes = await fetch(`${remote}/sync/changes?since=1970-01-01T00:00:00&limit=1000`, { headers: remoteHeaders }).then(r => r.json());
    if (changes.changes?.length > 0) {
      const result = await api("/sync/receive", {
        method: "POST",
        body: JSON.stringify({ memories: changes.changes }),
      });
      console.log(`  ← Received: ${result.created} created, ${result.updated} updated, ${result.skipped} skipped`);
    } else {
      console.log("  ← No changes");
    }

    // Push changes to remote
    console.log(`Pushing to ${remote}...`);
    const local = await api("/sync/changes?since=1970-01-01T00:00:00&limit=1000");
    if (local.changes?.length > 0) {
      const pushResult = await fetch(`${remote}/sync/receive`, {
        method: "POST", headers: remoteHeaders,
        body: JSON.stringify({ memories: local.changes }),
      }).then(r => r.json());
      console.log(`  → Sent: ${pushResult.created} created, ${pushResult.updated} updated, ${pushResult.skipped} skipped`);
    } else {
      console.log("  → No changes");
    }
  },

  // Forget
  async forget() {
    const id = args[0];
    if (!id) { console.error("Usage: engram forget <id>"); process.exit(1); }
    await api(`/memory/${id}/forget`, { method: "POST" });
    console.log(`✓ Forgotten #${id}`);
  },

  // Help
  async help() {
    console.log(`
engram — CLI for Engram memory server

Commands:
  store <content> [-c cat] [-i imp] [-t tags] [-s src] [--static]
  search <query> [-n limit]
  recall [context] [-n limit] [--tokens N]
  list [-n limit] [-c category]
  get <id>
  forget <id>
  tags [search <tag>]
  episodes [<id>]
  prompt [-f format] [--tokens N] [context]
  sync <remote_url>
  health
  help

Environment:
  ENGRAM_URL     Server URL (default: http://localhost:4200)
  ENGRAM_API_KEY API key for authentication

Examples:
  engram store "PostgreSQL upgraded to v16" -c task -i 8 -t database,migration
  engram search "database configuration"
  engram recall "deploy the new API" --tokens 4000
  engram prompt -f anthropic --tokens 2000 "user wants help with auth"
  engram sync https://other-instance.example.com
  echo "piped content" | engram store -c discovery
`);
  },
};

// Helpers
function getFlag(...flags: string[]): string | undefined {
  for (const f of flags) {
    const idx = args.indexOf(f);
    if (idx !== -1 && args[idx + 1]) return args[idx + 1];
  }
  return undefined;
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr + "Z").getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

async function readStdin(): Promise<string> {
  if (process.stdin.isTTY) return "";
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf-8").trim();
}

// Run
const fn = commands[cmd || "help"];
if (!fn) {
  console.error(`Unknown command: ${cmd}. Run 'engram help' for usage.`);
  process.exit(1);
}
fn().catch(e => {
  console.error(`Error: ${e.message}`);
  process.exit(1);
});
