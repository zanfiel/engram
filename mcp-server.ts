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
              version: "4.2.0",
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
