#!/usr/bin/env node
/**
 * Engram MCP Server — exposes Engram memory tools to OpenCode
 *
 * Tools: memory_store, memory_recall, memory_list, memory_delete, memory_context
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { signToolManifest, hashTool, type SignedToolManifest, type ToolDefinition } from "./sign/index.ts";

const ENGRAM_URL = process.env.ENGRAM_URL ?? "http://127.0.0.1:4200";
const ENGRAM_API_KEY = process.env.ENGRAM_API_KEY ?? "";
const ENGRAM_SIGNING_SECRET = process.env.ENGRAM_SIGNING_SECRET ?? "";
const SOURCE = "opencode";

async function engram(path: string, method = "GET", body?: unknown) {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (ENGRAM_API_KEY) headers["Authorization"] = `Bearer ${ENGRAM_API_KEY}`;
  const res = await fetch(`${ENGRAM_URL}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error(`Engram ${method} ${path} → ${res.status} ${await res.text()}`);
  return res.json() as Promise<any>;
}

const server = new Server(
  { name: "engram", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

// ── Tool Definitions (signed for integrity binding) ─────────────────

const TOOLS: ToolDefinition[] = [
  {
    name: "memory_store",
    description: "Store a persistent memory in Engram. Use for important decisions, discoveries, state, and task progress.",
    inputSchema: {
      type: "object",
      properties: {
        content: { type: "string", description: "Memory content" },
        category: {
          type: "string",
          enum: ["task", "discovery", "decision", "state", "issue", "general", "reference"],
          description: "Category (default: task)",
        },
        importance: { type: "number", description: "Importance 1-10 (default: 5)" },
      },
      required: ["content"],
    },
  },
  {
    name: "memory_recall",
    description: "Search Engram memories by semantic similarity. Use at session start or when you need context about past work.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "What to search for" },
        limit: { type: "number", description: "Max results (default: 10)" },
      },
      required: ["query"],
    },
  },
  {
    name: "memory_context",
    description: "Get a budget-aware context blob from Engram — relevance-ranked memories ready to inject into the conversation.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Topic / session description" },
        token_budget: { type: "number", description: "Max tokens to return (default: 6000)" },
      },
      required: ["query"],
    },
  },
  {
    name: "memory_list",
    description: "List recent Engram memories, optionally filtered by category.",
    inputSchema: {
      type: "object",
      properties: {
        category: { type: "string", description: "Filter by category (optional)" },
        limit: { type: "number", description: "Max results (default: 20)" },
      },
    },
  },
  {
    name: "memory_delete",
    description: "Delete a memory from Engram by ID.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Memory ID" },
      },
      required: ["id"],
    },
  },
];

// Sign the tool manifest at startup — clients can verify tools haven't been poisoned
const toolManifest: SignedToolManifest | null = ENGRAM_SIGNING_SECRET
  ? signToolManifest(ENGRAM_SIGNING_SECRET, TOOLS)
  : null;

// Compute per-tool hashes for inclusion in tool metadata
const toolHashes = new Map(TOOLS.map(t => [t.name, hashTool(t)]));

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS.map(t => ({
    ...t,
    // Include integrity hash so clients can verify tool definitions weren't tampered with
    ...(toolManifest ? { _integrity: { hash: toolHashes.get(t.name), manifest_hash: toolManifest.manifest_hash } } : {}),
  })),
  // Include signed manifest in _meta for clients that support verification
  ...(toolManifest ? { _meta: { tool_manifest: toolManifest } } : {}),
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args } = req.params;
  try {
    switch (name) {
      case "memory_store": {
        const result = await engram("/store", "POST", {
          content: args!.content,
          category: args!.category ?? "task",
          importance: args!.importance ?? 5,
          source: SOURCE,
        });
        return { content: [{ type: "text", text: `Stored memory (id: ${result.id ?? "ok"})` }] };
      }

      case "memory_recall": {
        const result = await engram("/recall", "POST", {
          query: args!.query,
          limit: args!.limit ?? 10,
        });
        const memories: any[] = result.memories ?? [];
        if (memories.length === 0) return { content: [{ type: "text", text: "No memories found." }] };
        const text = memories
          .map((m) => `[${m.category}] (id:${m.id}) ${m.content}`)
          .join("\n\n");
        return { content: [{ type: "text", text }] };
      }

      case "memory_context": {
        const result = await engram("/context", "POST", {
          query: args!.query,
          max_tokens: args!.token_budget ?? 8000,
        });
        const ctx = typeof result === "string" ? result : result.context ?? JSON.stringify(result);
        return { content: [{ type: "text", text: ctx || "No context available." }] };
      }

      case "memory_list": {
        const params = new URLSearchParams();
        if (args!.category) params.set("category", String(args!.category));
        params.set("limit", String(args!.limit ?? 20));
        const result = await engram(`/list?${params}`);
        const memories: any[] = result.memories ?? result ?? [];
        if (memories.length === 0) return { content: [{ type: "text", text: "No memories." }] };
        const text = memories
          .map((m) => `[${m.category}] (id:${m.id}) ${String(m.content).slice(0, 200)}`)
          .join("\n");
        return { content: [{ type: "text", text }] };
      }

      case "memory_delete": {
        await engram(`/memory/${args!.id}`, "DELETE");
        return { content: [{ type: "text", text: `Deleted memory ${args!.id}` }] };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (err: any) {
    return { content: [{ type: "text", text: `Error: ${err.message}` }], isError: true };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
