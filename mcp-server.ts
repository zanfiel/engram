#!/usr/bin/env bun
/**
 * Engram MCP Server — Model Context Protocol integration
 *
 * Exposes Engram as an MCP tool server for Claude Desktop, Cursor, Windsurf, etc.
 *
 * Setup (claude_desktop_config.json):
 * {
 *   "mcpServers": {
 *     "engram": {
 *       "command": "bun",
 *       "args": ["run", "/path/to/mcp-server.ts"],
 *       "env": {
 *         "ENGRAM_URL": "http://localhost:4200",
 *         "ENGRAM_API_KEY": "eg_yourkey"
 *       }
 *     }
 *   }
 * }
 */

const ENGRAM_URL = process.env.ENGRAM_URL || "http://localhost:4200";
const ENGRAM_KEY = process.env.ENGRAM_API_KEY || "";

const headers: Record<string, string> = { "Content-Type": "application/json" };
if (ENGRAM_KEY) headers["Authorization"] = `Bearer ${ENGRAM_KEY}`;

async function api(path: string, opts?: RequestInit): Promise<any> {
  const resp = await fetch(`${ENGRAM_URL}${path}`, { ...opts, headers: { ...headers, ...opts?.headers } });
  return resp.json();
}

// MCP Protocol types
interface MCPRequest {
  jsonrpc: "2.0";
  id: number | string;
  method: string;
  params?: any;
}

interface MCPResponse {
  jsonrpc: "2.0";
  id: number | string;
  result?: any;
  error?: { code: number; message: string };
}

// Tool definitions
const tools = [
  {
    name: "engram_store",
    description: "Store a memory in Engram. Use this to persist important information, decisions, code context, or anything the user might need in future sessions.",
    inputSchema: {
      type: "object",
      properties: {
        content: { type: "string", description: "The memory content to store" },
        category: { type: "string", enum: ["task", "discovery", "decision", "state", "issue", "general"], description: "Memory category" },
        importance: { type: "number", minimum: 1, maximum: 10, description: "Importance (1-10, default 5)" },
        tags: { type: "array", items: { type: "string" }, description: "Tags for categorization" },
        source: { type: "string", description: "Source identifier (e.g. agent name)" },
      },
      required: ["content"],
    },
  },
  {
    name: "engram_search",
    description: "Search memories by semantic similarity. Returns the most relevant memories matching the query.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default 10)" },
        tag: { type: "string", description: "Filter by tag" },
      },
      required: ["query"],
    },
  },
  {
    name: "engram_recall",
    description: "Smart recall — get contextually relevant memories using four-layer retrieval (static facts, semantic matches, high importance, recent activity).",
    inputSchema: {
      type: "object",
      properties: {
        context: { type: "string", description: "Current conversation context or topic" },
        limit: { type: "number", description: "Max memories (default 20)" },
        tags: { type: "array", items: { type: "string" }, description: "Boost memories with these tags" },
      },
      required: [],
    },
  },
  {
    name: "engram_pack",
    description: "Get optimally packed memories for a token budget. Perfect for system prompts.",
    inputSchema: {
      type: "object",
      properties: {
        context: { type: "string", description: "Context to search for" },
        tokens: { type: "number", description: "Token budget (default 4000)" },
        format: { type: "string", enum: ["text", "xml", "json"], description: "Output format" },
      },
      required: [],
    },
  },
  {
    name: "engram_forget",
    description: "Mark a memory as forgotten (soft delete).",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "number", description: "Memory ID to forget" },
      },
      required: ["id"],
    },
  },
  {
    name: "engram_update",
    description: "Update an existing memory (creates a new version).",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "number", description: "Memory ID to update" },
        content: { type: "string", description: "New content" },
        importance: { type: "number", description: "New importance (1-10)" },
      },
      required: ["id", "content"],
    },
  },
  {
    name: "engram_tags",
    description: "List all tags or search memories by tag.",
    inputSchema: {
      type: "object",
      properties: {
        tag: { type: "string", description: "Search for memories with this tag (omit to list all tags)" },
      },
      required: [],
    },
  },
  {
    name: "engram_entities",
    description: "Manage entities (people, organizations, devices, products, services). Create, list, search, or get details.",
    inputSchema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["create", "list", "get", "search", "link", "unlink", "relate"], description: "Action to perform" },
        id: { type: "number", description: "Entity ID (for get/link/unlink/relate)" },
        name: { type: "string", description: "Entity name (for create)" },
        type: { type: "string", enum: ["person", "organization", "team", "device", "product", "service", "generic"], description: "Entity type" },
        description: { type: "string", description: "Entity description" },
        aka: { type: "string", description: "Also known as / aliases" },
        query: { type: "string", description: "Search query (for search)" },
        memory_id: { type: "number", description: "Memory ID to link/unlink" },
        target_id: { type: "number", description: "Target entity ID (for relate)" },
        relationship: { type: "string", description: "Relationship type (for relate, e.g. 'works_at', 'owns')" },
      },
      required: ["action"],
    },
  },
  {
    name: "engram_projects",
    description: "Manage projects for scoping memories. Create, list, search within a project, or link memories.",
    inputSchema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["create", "list", "get", "search", "link", "unlink"], description: "Action to perform" },
        id: { type: "number", description: "Project ID (for get/search/link/unlink)" },
        name: { type: "string", description: "Project name (for create)" },
        description: { type: "string", description: "Project description" },
        status: { type: "string", enum: ["active", "paused", "completed", "archived"], description: "Project status" },
        memory_id: { type: "number", description: "Memory ID to link/unlink" },
        query: { type: "string", description: "Search query (for scoped search)" },
      },
      required: ["action"],
    },
  },
];

// Resource definitions
const resources = [
  {
    uri: "engram://health",
    name: "Engram Health",
    description: "Current Engram instance status",
    mimeType: "application/json",
  },
  {
    uri: "engram://profile",
    name: "Engram Profile",
    description: "User profile built from stored memories",
    mimeType: "application/json",
  },
];

// Tool execution
async function executeTool(name: string, args: any): Promise<string> {
  switch (name) {
    case "engram_store": {
      const result = await api("/store", {
        method: "POST",
        body: JSON.stringify({
          content: args.content,
          category: args.category || "general",
          importance: args.importance || 5,
          tags: args.tags,
          source: args.source || "mcp",
        }),
      });
      return `Stored memory #${result.id} (importance: ${result.importance}, linked: ${result.linked}, tags: [${(result.tags || []).join(", ")}])`;
    }

    case "engram_search": {
      const result = await api("/search", {
        method: "POST",
        body: JSON.stringify({ query: args.query, limit: args.limit || 10, tag: args.tag }),
      });
      if (!result.results?.length) return "No results found.";
      return result.results.map((r: any) =>
        `[#${r.id}] (score: ${r.score.toFixed(3)}, ${r.category}) ${r.content}`
      ).join("\n\n");
    }

    case "engram_recall": {
      const result = await api("/recall", {
        method: "POST",
        body: JSON.stringify({ context: args.context || "", limit: args.limit || 20, tags: args.tags }),
      });
      if (!result.memories?.length) return "No memories to recall.";
      const b = result.breakdown;
      let text = result.memories.map((m: any) =>
        `[#${m.id}] (${m.recall_source}) ${m.content}`
      ).join("\n\n");
      text += `\n\n---\n${result.memories.length} memories: ${b.static} static, ${b.semantic} semantic, ${b.important} important, ${b.recent} recent`;
      return text;
    }

    case "engram_pack": {
      const result = await api("/pack", {
        method: "POST",
        body: JSON.stringify({
          context: args.context || "",
          tokens: args.tokens || 4000,
          format: args.format || "text",
        }),
      });
      return `${result.packed}\n\n---\n${result.memories_included} memories, ~${result.tokens_estimated} tokens (${result.utilization})`;
    }

    case "engram_forget": {
      await api(`/memory/${args.id}/forget`, { method: "POST" });
      return `Memory #${args.id} forgotten.`;
    }

    case "engram_update": {
      const result = await api(`/memory/${args.id}/update`, {
        method: "POST",
        body: JSON.stringify({ content: args.content, importance: args.importance }),
      });
      return `Memory #${args.id} updated → new version #${result.id || args.id}`;
    }

    case "engram_tags": {
      if (args.tag) {
        const result = await api("/tags/search", { method: "POST", body: JSON.stringify({ tag: args.tag }) });
        if (!result.results?.length) return `No memories with tag "${args.tag}"`;
        return result.results.map((r: any) =>
          `[#${r.id}] [${(r.tags || []).join(", ")}] ${r.content.substring(0, 120)}`
        ).join("\n");
      } else {
        const result = await api("/tags");
        return `Tags: ${result.tags.join(", ")}`;
      }
    }

    case "engram_entities": {
      switch (args.action) {
        case "create": {
          if (!args.name) throw new Error("name required for create");
          const result = await api("/entities", {
            method: "POST",
            body: JSON.stringify({ name: args.name, type: args.type || "generic", description: args.description, aka: args.aka }),
          });
          return `Created entity #${result.id}: ${args.name} (${args.type || "generic"})`;
        }
        case "list": {
          const params = args.type ? `?type=${args.type}` : "";
          const result = await api(`/entities${params}`);
          if (!result.entities?.length) return "No entities found.";
          return result.entities.map((e: any) =>
            `#${e.id} [${e.type}] ${e.name}${e.aka ? ` (aka: ${e.aka})` : ""} — ${e.memory_count} memories`
          ).join("\n");
        }
        case "get": {
          if (!args.id) throw new Error("id required for get");
          const e = await api(`/entities/${args.id}`);
          let text = `Entity #${e.id}: ${e.name} (${e.type})`;
          if (e.description) text += `\n  ${e.description}`;
          if (e.aka) text += `\n  AKA: ${e.aka}`;
          if (e.relationships?.length) {
            text += `\n  Relationships:`;
            for (const r of e.relationships) text += `\n    ${r.direction === "outgoing" ? "→" : "←"} ${r.relationship} ${r.related_entity_name} (#${r.related_entity_id})`;
          }
          if (e.memories?.length) {
            text += `\n  Memories (${e.memories.length}):`;
            for (const m of e.memories) text += `\n    #${m.id} [${m.category}] ${m.content.substring(0, 100)}`;
          }
          return text;
        }
        case "search": {
          if (!args.query) throw new Error("query required for search");
          const result = await api(`/entities?q=${encodeURIComponent(args.query)}`);
          if (!result.entities?.length) return `No entities matching "${args.query}"`;
          return result.entities.map((e: any) => `#${e.id} [${e.type}] ${e.name} — ${e.memory_count} memories`).join("\n");
        }
        case "link": {
          if (!args.id || !args.memory_id) throw new Error("id and memory_id required for link");
          await api(`/entities/${args.id}/memories/${args.memory_id}`, { method: "PUT" });
          return `Linked memory #${args.memory_id} to entity #${args.id}`;
        }
        case "unlink": {
          if (!args.id || !args.memory_id) throw new Error("id and memory_id required for unlink");
          await api(`/entities/${args.id}/memories/${args.memory_id}`, { method: "DELETE" });
          return `Unlinked memory #${args.memory_id} from entity #${args.id}`;
        }
        case "relate": {
          if (!args.id || !args.target_id || !args.relationship) throw new Error("id, target_id, and relationship required");
          await api(`/entities/${args.id}/relationships`, {
            method: "POST",
            body: JSON.stringify({ target_id: args.target_id, relationship: args.relationship }),
          });
          return `Entity #${args.id} → ${args.relationship} → Entity #${args.target_id}`;
        }
        default: throw new Error(`Unknown entity action: ${args.action}`);
      }
    }

    case "engram_projects": {
      switch (args.action) {
        case "create": {
          if (!args.name) throw new Error("name required for create");
          const result = await api("/projects", {
            method: "POST",
            body: JSON.stringify({ name: args.name, description: args.description, status: args.status || "active" }),
          });
          return `Created project #${result.id}: ${args.name} (${result.status})`;
        }
        case "list": {
          const params = args.status ? `?status=${args.status}` : "";
          const result = await api(`/projects${params}`);
          if (!result.projects?.length) return "No projects found.";
          return result.projects.map((p: any) =>
            `#${p.id} [${p.status}] ${p.name} — ${p.memory_count} memories`
          ).join("\n");
        }
        case "get": {
          if (!args.id) throw new Error("id required for get");
          const p = await api(`/projects/${args.id}`);
          let text = `Project #${p.id}: ${p.name} (${p.status})`;
          if (p.description) text += `\n  ${p.description}`;
          if (p.memories?.length) {
            text += `\n  Memories (${p.memories.length}):`;
            for (const m of p.memories) text += `\n    #${m.id} [${m.category}] ${m.content.substring(0, 100)}`;
          }
          return text;
        }
        case "search": {
          if (!args.id || !args.query) throw new Error("id and query required for scoped search");
          const result = await api(`/projects/${args.id}/search`, {
            method: "POST",
            body: JSON.stringify({ query: args.query }),
          });
          if (!result.results?.length) return `No memories matching "${args.query}" in project #${args.id}`;
          return result.results.map((r: any) => `#${r.id} (${r.score.toFixed(3)}) ${r.content.substring(0, 120)}`).join("\n");
        }
        case "link": {
          if (!args.id || !args.memory_id) throw new Error("id and memory_id required for link");
          await api(`/projects/${args.id}/memories/${args.memory_id}`, { method: "PUT" });
          return `Linked memory #${args.memory_id} to project #${args.id}`;
        }
        case "unlink": {
          if (!args.id || !args.memory_id) throw new Error("id and memory_id required for unlink");
          await api(`/projects/${args.id}/memories/${args.memory_id}`, { method: "DELETE" });
          return `Unlinked memory #${args.memory_id} from project #${args.id}`;
        }
        default: throw new Error(`Unknown project action: ${args.action}`);
      }
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// Resource reading
async function readResource(uri: string): Promise<string> {
  if (uri === "engram://health") {
    return JSON.stringify(await api("/health"), null, 2);
  }
  if (uri === "engram://profile") {
    return JSON.stringify(await api("/profile"), null, 2);
  }
  throw new Error(`Unknown resource: ${uri}`);
}

// MCP JSON-RPC handler
async function handleRequest(req: MCPRequest): Promise<MCPResponse> {
  try {
    switch (req.method) {
      case "initialize":
        return {
          jsonrpc: "2.0",
          id: req.id,
          result: {
            protocolVersion: "2024-11-05",
            capabilities: {
              tools: {},
              resources: {},
            },
            serverInfo: {
              name: "engram",
              version: "4.3.0",
            },
          },
        };

      case "notifications/initialized":
        return { jsonrpc: "2.0", id: req.id, result: {} };

      case "tools/list":
        return { jsonrpc: "2.0", id: req.id, result: { tools } };

      case "tools/call": {
        const { name, arguments: toolArgs } = req.params;
        const text = await executeTool(name, toolArgs || {});
        return {
          jsonrpc: "2.0",
          id: req.id,
          result: { content: [{ type: "text", text }] },
        };
      }

      case "resources/list":
        return { jsonrpc: "2.0", id: req.id, result: { resources } };

      case "resources/read": {
        const text = await readResource(req.params.uri);
        return {
          jsonrpc: "2.0",
          id: req.id,
          result: { contents: [{ uri: req.params.uri, mimeType: "application/json", text }] },
        };
      }

      case "ping":
        return { jsonrpc: "2.0", id: req.id, result: {} };

      default:
        return {
          jsonrpc: "2.0",
          id: req.id,
          error: { code: -32601, message: `Method not found: ${req.method}` },
        };
    }
  } catch (e: any) {
    return {
      jsonrpc: "2.0",
      id: req.id,
      error: { code: -32000, message: e.message },
    };
  }
}

// stdio transport — reads JSON-RPC from stdin, writes to stdout
async function main() {
  const decoder = new TextDecoder();
  let buffer = "";

  process.stderr.write("Engram MCP server started\n");

  for await (const chunk of process.stdin) {
    buffer += decoder.decode(chunk, { stream: true });

    // MCP uses Content-Length headers (LSP-style framing)
    while (true) {
      const headerEnd = buffer.indexOf("\r\n\r\n");
      if (headerEnd === -1) break;

      const headerSection = buffer.substring(0, headerEnd);
      const contentLengthMatch = headerSection.match(/Content-Length: (\d+)/i);
      if (!contentLengthMatch) {
        // Try parsing as raw JSON (some clients don't use LSP framing)
        const nlIdx = buffer.indexOf("\n");
        if (nlIdx === -1) break;
        const line = buffer.substring(0, nlIdx).trim();
        buffer = buffer.substring(nlIdx + 1);
        if (!line) continue;
        try {
          const req = JSON.parse(line) as MCPRequest;
          const resp = await handleRequest(req);
          if (req.id !== undefined) {
            const respStr = JSON.stringify(resp);
            process.stdout.write(`Content-Length: ${Buffer.byteLength(respStr)}\r\n\r\n${respStr}`);
          }
        } catch {}
        continue;
      }

      const contentLength = Number(contentLengthMatch[1]);
      const bodyStart = headerEnd + 4;

      if (buffer.length < bodyStart + contentLength) break;

      const body = buffer.substring(bodyStart, bodyStart + contentLength);
      buffer = buffer.substring(bodyStart + contentLength);

      try {
        const req = JSON.parse(body) as MCPRequest;
        const resp = await handleRequest(req);
        if (req.id !== undefined) {
          const respStr = JSON.stringify(resp);
          process.stdout.write(`Content-Length: ${Buffer.byteLength(respStr)}\r\n\r\n${respStr}`);
        }
      } catch (e: any) {
        process.stderr.write(`Parse error: ${e.message}\n`);
      }
    }
  }
}

main();
