import { type Plugin, tool } from "@opencode-ai/plugin";
import { hostname } from "os";

/**
 * MegaMind v3.1 — Zan's self-hosted persistent memory system plugin.
 *
 * Server: Bun + SQLite FTS5 + embeddings on Rocky:4200
 * Reachable: 127.0.0.1:4200, 192.168.8.133:4200, memory.zanverse.dev
 *
 * v3.1 Features:
 *  - Smart recall: context-aware memory injection using /recall endpoint
 *  - Profile injection: static facts + recent activity on session start
 *  - Smart compaction: static + high-importance + semantically relevant memories
 *  - Tools: store, search, list, delete, update (versioned), archive
 *  - Conversation search: full-text search across all past conversations
 *  - Source tagging: every memory tagged with hostname
 */

const MEMORY_URL = process.env.MEGAMIND_URL || "http://127.0.0.1:4200";
const SOURCE_TAG = `opencode@${hostname()}`;

// ============================================================================
// TYPES
// ============================================================================

interface Memory {
  id: number;
  content: string;
  category: string;
  source: string;
  session_id: string | null;
  importance: number;
  created_at: string;
  version?: number;
  is_latest?: boolean;
  is_static?: boolean;
  is_archived?: boolean;
  source_count?: number;
  recall_source?: string;
  recall_score?: number;
}

interface RecallResponse {
  memories: Memory[];
  breakdown: {
    static: number;
    semantic: number;
    important: number;
    recent: number;
  };
}

interface ProfileResponse {
  static_facts: Array<{ id: number; content: string; category: string; source_count: number }>;
  recent_activity: Array<{ id: number; content: string; category: string; created_at: string }>;
  summary?: string;
}

interface StoreResponse {
  stored: boolean;
  id: number;
  created_at: string;
  importance: number;
  linked: number;
  embedded: boolean;
  fact_extraction: string;
}

interface UpdateResponse {
  updated: boolean;
  old_id: number;
  new_id: number;
  version: number;
  root_id: number;
  linked: number;
  embedded: boolean;
}

interface SearchResponse {
  results: Memory[];
}

interface ListResponse {
  results: Memory[];
}

interface MessageSearchResult {
  id: number;
  conversation_id: number;
  role: string;
  content: string;
  created_at: string;
  agent: string;
  conv_title: string;
}

interface MessageSearchResponse {
  results: MessageSearchResult[];
}

// ============================================================================
// API CLIENT
// ============================================================================

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

// ============================================================================
// FORMATTING
// ============================================================================

function formatBadges(m: Memory): string {
  const badges: string[] = [];
  if (m.is_static) badges.push("STATIC");
  if (m.is_archived) badges.push("ARCHIVED");
  if (m.version && m.version > 1) badges.push(`v${m.version}`);
  if (m.source_count && m.source_count > 1) badges.push(`x${m.source_count}`);
  return badges.length ? ` [${badges.join(", ")}]` : "";
}

function formatMemoryLine(m: Memory): string {
  return `[#${m.id}] ${m.created_at} (${m.category})${formatBadges(m)} ${m.content}`;
}

function formatMemoryBlock(memories: Memory[], heading: string): string {
  if (!memories.length) return "";
  const lines = memories.map(formatMemoryLine);
  return `## ${heading}\nThese are persistent memories from previous sessions. Use them to maintain continuity and avoid re-asking the user for context they already provided.\n\n${lines.join("\n")}`;
}

function formatProfileBlock(profile: ProfileResponse): string {
  const parts: string[] = ["## MegaMind — Profile"];
  if (profile.static_facts.length > 0) {
    parts.push("### Permanent Facts");
    parts.push(...profile.static_facts.map(f => `- [${f.category}] ${f.content}`));
  }
  if (profile.recent_activity.length > 0) {
    parts.push("### Recent Activity");
    parts.push(...profile.recent_activity.slice(0, 10).map(f => `- [${f.created_at}] (${f.category}) ${f.content}`));
  }
  return parts.join("\n");
}

// ============================================================================
// PLUGIN
// ============================================================================

export const MegaMind: Plugin = async (ctx) => {
  let initialRecallDone = false;
  let cachedProfile: ProfileResponse | null = null;
  let firstUserMessage: string | null = null;

  // Smart recall: uses /recall endpoint with context
  async function smartRecall(context: string = "", limit = 20): Promise<Memory[]> {
    try {
      const resp = await apiFetch<RecallResponse>("/recall", {
        method: "POST",
        body: JSON.stringify({ context, limit }),
      });
      if (resp.memories?.length > 0) {
        await ctx.client.app.log({
          body: {
            service: "megamind",
            level: "info",
            message: `Smart recall: ${resp.memories.length} memories (${resp.breakdown.static}s/${resp.breakdown.semantic}sem/${resp.breakdown.important}imp/${resp.breakdown.recent}rec)`,
          },
        });
      }
      return resp.memories || [];
    } catch (e: any) {
      await ctx.client.app.log({
        body: {
          service: "megamind",
          level: "warn",
          message: `Smart recall failed: ${e.message}`,
        },
      });
      // Fallback to basic list
      try {
        const { results } = await apiFetch<ListResponse>(`/list?limit=${limit}`);
        return results || [];
      } catch {
        return [];
      }
    }
  }

  // Fetch profile
  async function fetchProfile(): Promise<ProfileResponse | null> {
    try {
      return await apiFetch<ProfileResponse>("/profile");
    } catch (e: any) {
      await ctx.client.app.log({
        body: {
          service: "megamind",
          level: "warn",
          message: `Profile fetch failed: ${e.message}`,
        },
      });
      return null;
    }
  }

  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        // Pre-fetch profile on session start
        cachedProfile = await fetchProfile();
      }
    },

    // Intercept the first user message to do context-aware recall
    "experimental.session.prompt": async (_input, output) => {
      if (!initialRecallDone) {
        initialRecallDone = true;

        // Extract the user's first message as context for smart recall
        const userMessage = output.parts
          .filter((p: any) => p.type === "text")
          .map((p: any) => p.text || "")
          .join(" ")
          .trim();

        firstUserMessage = userMessage;

        // Parallel fetch: profile (if not cached) + smart recall
        const [profile, memories] = await Promise.all([
          cachedProfile ? Promise.resolve(cachedProfile) : fetchProfile(),
          smartRecall(userMessage, 20),
        ]);

        // Inject profile first, then memories
        const contextBlocks: string[] = [];

        if (profile && (profile.static_facts.length > 0 || profile.recent_activity.length > 0)) {
          contextBlocks.push(formatProfileBlock(profile));
          cachedProfile = profile;
        }

        if (memories.length > 0) {
          contextBlocks.push(formatMemoryBlock(memories, "MegaMind — Persistent Context"));
        }

        if (contextBlocks.length > 0) {
          output.parts.unshift({
            type: "text",
            text: contextBlocks.join("\n\n"),
          });
        }
      }
    },

    // Smart compaction: injects static + important + semantically relevant memories
    "experimental.session.compacting": async (_input, output) => {
      // Get the latest context from the conversation for semantic relevance
      const recentContext = output.context.slice(-3).join(" ").substring(0, 500);

      const [profile, memories] = await Promise.all([
        fetchProfile(),
        smartRecall(recentContext || firstUserMessage || "", 25),
      ]);

      const contextBlocks: string[] = [];

      if (profile && (profile.static_facts.length > 0 || profile.recent_activity.length > 0)) {
        contextBlocks.push(formatProfileBlock(profile));
      }

      if (memories.length > 0) {
        contextBlocks.push(formatMemoryBlock(memories, "MegaMind — Persistent Context"));
      }

      if (contextBlocks.length > 0) {
        output.context.push(contextBlocks.join("\n\n"));
      }
    },

    tool: {
      // ====================================================================
      // STORE
      // ====================================================================
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
            const extras: string[] = [];
            if (result.linked > 0) extras.push(`${result.linked} links`);
            if (result.fact_extraction === "queued") extras.push("fact extraction queued");
            const suffix = extras.length ? ` (${extras.join(", ")})` : "";
            return `Stored memory #${result.id} (${args.category})${suffix}: ${args.content.substring(0, 80)}...`;
          } catch (e: any) {
            return `Failed to store memory: ${e.message}`;
          }
        },
      }),

      // ====================================================================
      // SEARCH
      // ====================================================================
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
                expand_relationships: true,
              }),
            });
            if (!result.results?.length) {
              return "No memories found matching that query.";
            }
            return result.results.map(formatMemoryLine).join("\n\n");
          } catch (e: any) {
            return `Search failed: ${e.message}`;
          }
        },
      }),

      // ====================================================================
      // LIST
      // ====================================================================
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
            const result = await apiFetch<ListResponse>(`/list?${params}`);
            if (!result.results?.length) {
              return "No memories found.";
            }
            return result.results.map(formatMemoryLine).join("\n\n");
          } catch (e: any) {
            return `List failed: ${e.message}`;
          }
        },
      }),

      // ====================================================================
      // DELETE
      // ====================================================================
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

      // ====================================================================
      // UPDATE (versioned — creates new version, preserves history)
      // ====================================================================
      memory_update: tool({
        description:
          "Update a memory by creating a new version. The old version is preserved in the version chain. Use this instead of delete+store when correcting or updating existing information.",
        args: {
          id: tool.schema
            .number()
            .describe("The memory ID to update (shown as #N in list/search)"),
          content: tool.schema
            .string()
            .describe("The new/updated content for this memory"),
          category: tool.schema
            .enum(["task", "discovery", "decision", "state", "issue"])
            .optional()
            .describe("Optionally change the category"),
        },
        async execute(args) {
          try {
            const body: any = { content: args.content };
            if (args.category) body.category = args.category;
            const result = await apiFetch<UpdateResponse>(`/memory/${args.id}/update`, {
              method: "POST",
              body: JSON.stringify(body),
            });
            return `Updated memory #${args.id} -> #${result.new_id} (v${result.version}, root=#${result.root_id}): ${args.content.substring(0, 80)}...`;
          } catch (e: any) {
            return `Update failed: ${e.message}`;
          }
        },
      }),

      // ====================================================================
      // ARCHIVE (soft-remove from active recall, still searchable)
      // ====================================================================
      memory_archive: tool({
        description:
          "Archive a memory — removes it from active recall and lists, but it remains searchable. Use this for stale/outdated memories you want to keep but not see in daily context.",
        args: {
          id: tool.schema
            .number()
            .describe("The memory ID to archive"),
        },
        async execute(args) {
          try {
            await apiFetch(`/memory/${args.id}/archive`, { method: "POST" });
            return `Memory #${args.id} archived. It will no longer appear in recall/lists but can still be found via search.`;
          } catch (e: any) {
            return `Archive failed: ${e.message}`;
          }
        },
      }),

      // ====================================================================
      // CONVERSATION SEARCH
      // ====================================================================
      conversation_search: tool({
        description:
          "Search through all past OpenCode conversation messages. Use this to find what was discussed, when something was done, or to locate specific conversations by content.",
        args: {
          query: tool.schema
            .string()
            .describe("Search keywords to find in past conversations"),
          limit: tool.schema
            .number()
            .optional()
            .describe("Max results (default 20, max 100)"),
        },
        async execute(args) {
          try {
            const result = await apiFetch<MessageSearchResponse>("/messages/search", {
              method: "POST",
              body: JSON.stringify({
                query: args.query,
                limit: Math.min(args.limit || 20, 100),
              }),
            });
            if (!result.results?.length) {
              return "No conversation messages found matching that query.";
            }
            return result.results
              .map((m) => {
                const preview = m.content.length > 200 ? m.content.substring(0, 200) + "..." : m.content;
                return `[Conv #${m.conversation_id}${m.conv_title ? ` "${m.conv_title}"` : ""}] ${m.created_at} (${m.role}/${m.agent})\n${preview}`;
              })
              .join("\n\n---\n\n");
          } catch (e: any) {
            return `Conversation search failed: ${e.message}`;
          }
        },
      }),
    },
  };
};
