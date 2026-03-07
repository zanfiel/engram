import { type Plugin, tool } from "@opencode-ai/plugin";
import { hostname } from "os";

/**
 * MegaMind — Zan's self-hosted persistent memory system for all agents.
 *
 * Talks to the MegaMind API server (Bun + SQLite FTS5) running on Rocky
 * at port 4200 (reachable via Tailscale at 100.120.223.125:4200).
 * NO external APIs, NO Copilot calls, NO OpenCode official servers.
 *
 * Features:
 *  - Auto-recall: Injects recent memories into compaction context so every
 *    session (new or compacted) starts with full continuity.
 *  - Tools: memory_store, memory_search, memory_list, memory_delete
 *  - Source tagging: Every memory is tagged with hostname for traceability.
 *  - Conversations DB: Full conversation storage with FTS search.
 */

const MEMORY_URL = process.env.MEGAMIND_URL || process.env.OPUS_MEMORY_URL || "http://127.0.0.1:4200";
const SOURCE_TAG = `opencode@${hostname()}`;

interface Memory {
  id: number;
  content: string;
  category: string;
  source: string;
  session_id: string | null;
  created_at: string;
}

interface ListResponse {
  results: Memory[];
}

interface StoreResponse {
  stored: boolean;
  id: number;
  created_at: string;
}

interface SearchResponse {
  results: Memory[];
}

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${MEMORY_URL}${path}`, {
    ...opts,
    headers: { "Content-Type": "application/json", ...opts?.headers },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`MegaMind API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

function formatMemoryBlock(memories: Memory[], heading: string): string {
  if (!memories.length) return "";
  const lines = memories.map(
    (m) => `- [${m.created_at}] (${m.category}) ${m.content}`
  );
  return `## ${heading}\nThese are persistent memories from previous sessions. Use them to maintain continuity and avoid re-asking the user for context they already provided.\n\n${lines.join("\n")}`;
}

export const MegaMind: Plugin = async (ctx) => {
  // Cache of memories fetched at session start, used for compaction injection
  let cachedMemories: Memory[] = [];
  let initialRecallDone = false;

  // Fetch memories from the server (best-effort, never throws to caller)
  async function recallMemories(limit = 25): Promise<Memory[]> {
    try {
      const { results } = await apiFetch<ListResponse>(
        `/list?limit=${limit}`
      );
      return results || [];
    } catch (e: any) {
      await ctx.client.app.log({
        body: {
          service: "megamind",
          level: "warn",
          message: `Recall failed: ${e.message}`,
        },
      });
      return [];
    }
  }

  // Do initial recall once
  async function ensureRecall() {
    if (!initialRecallDone) {
      initialRecallDone = true;
      cachedMemories = await recallMemories(25);
      if (cachedMemories.length > 0) {
        await ctx.client.app.log({
          body: {
            service: "megamind",
            level: "info",
            message: `Recalled ${cachedMemories.length} memories for session`,
          },
        });
      }
    }
  }

  return {
    // Listen for session events to trigger recall
    event: async ({ event }) => {
      if (event.type === "session.created") {
        await ensureRecall();
      }
    },

    // Inject memories into compaction context so they survive context resets.
    // This fires before the LLM generates a continuation summary.
    "experimental.session.compacting": async (_input, output) => {
      // Refresh memories (they may have been updated during the session)
      const memories = await recallMemories(30);
      if (memories.length > 0) {
        output.context.push(
          formatMemoryBlock(memories, "MegaMind — Persistent Context")
        );
      }
    },

    tool: {
      memory_store: tool({
        description:
          "Store a memory in MegaMind (Zan's persistent memory system). Use this to record completed work, decisions, discovered information, infrastructure state, or known issues. Memories persist across ALL OpenCode sessions. Be specific and concise — include what, where, and relevant details. You MUST use this tool at the end of every task to ensure continuity.",
        args: {
          content: tool.schema
            .string()
            .describe(
              "The memory content. Be specific: what was done, on which server/file, any relevant IDs or values."
            ),
          category: tool.schema
            .enum(["task", "discovery", "decision", "state", "issue"])
            .describe(
              "Category: task=completed work, discovery=found info, decision=choice made, state=current status, issue=known problem"
            ),
        },
        async execute(args, context) {
          try {
            const result = await apiFetch<StoreResponse>("/store", {
              method: "POST",
              body: JSON.stringify({
                content: args.content,
                category: args.category,
                source: SOURCE_TAG,
                session_id: context.sessionID,
              }),
            });
            return `Stored memory #${result.id} (${args.category}): ${args.content.substring(0, 80)}...`;
          } catch (e: any) {
            return `Failed to store memory: ${e.message}`;
          }
        },
      }),

      memory_search: tool({
        description:
          "Search MegaMind for relevant memories using full-text search. Use this to find what was previously done, decisions made, current state of infrastructure, or known issues.",
        args: {
          query: tool.schema
            .string()
            .describe("Search keywords — what you want to find"),
          limit: tool.schema
            .number()
            .optional()
            .describe("Max results (default 10, max 50)"),
        },
        async execute(args) {
          try {
            const result = await apiFetch<SearchResponse>("/search", {
              method: "POST",
              body: JSON.stringify({
                query: args.query,
                limit: Math.min(args.limit || 10, 50),
              }),
            });
            if (!result.results?.length) {
              return "No memories found matching that query.";
            }
            return result.results
              .map(
                (m) =>
                  `[#${m.id}] ${m.created_at} (${m.category}) ${m.content}`
              )
              .join("\n\n");
          } catch (e: any) {
            return `Search failed: ${e.message}`;
          }
        },
      }),

      memory_list: tool({
        description:
          "List recent memories from MegaMind, optionally filtered by category. Use this at the START of every new session to understand what was done previously.",
        args: {
          category: tool.schema
            .enum(["task", "discovery", "decision", "state", "issue"])
            .optional()
            .describe("Filter by category (omit for all)"),
          limit: tool.schema
            .number()
            .optional()
            .describe("Number of results (default 15, max 50)"),
        },
        async execute(args) {
          try {
            const limit = Math.min(args.limit || 15, 50);
            const params = new URLSearchParams({ limit: String(limit) });
            if (args.category) params.set("category", args.category);
            const result = await apiFetch<ListResponse>(
              `/list?${params}`
            );
            if (!result.results?.length) {
              return "No memories found.";
            }
            return result.results
              .map(
                (m) =>
                  `[#${m.id}] ${m.created_at} (${m.category}) ${m.content}`
              )
              .join("\n\n");
          } catch (e: any) {
            return `List failed: ${e.message}`;
          }
        },
      }),

      memory_delete: tool({
        description:
          "Delete a memory by ID. Use this to clean up outdated or incorrect information.",
        args: {
          id: tool.schema
            .number()
            .describe("The memory ID to delete (shown as #N in list/search)"),
        },
        async execute(args) {
          try {
            await apiFetch(`/memory/${args.id}`, { method: "DELETE" });
            return `Memory #${args.id} deleted.`;
          } catch (e: any) {
            return `Delete failed: ${e.message}`;
          }
        },
      }),
    },
  };
};
