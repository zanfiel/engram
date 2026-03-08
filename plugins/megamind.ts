import { type Plugin, tool } from "@opencode-ai/plugin";
import type { Part } from "@opencode-ai/sdk";
import { hostname } from "os";
import { readFileSync } from "fs";
import { join } from "path";

/**
 * MegaMind — Zan's self-hosted persistent memory system for all agents.
 *
 * Talks to the MegaMind API server (Bun + SQLite + Vector Embeddings)
 * running on Rocky at port 4200.
 *
 * The URL is resolved dynamically from megamind-url.txt in the OpenCode
 * config directory, so it can be changed without restarting OpenCode.
 *
 * v3 upgrades:
 *  - Memory versioning: contradiction resolution, version chains
 *  - LLM-powered fact extraction: async pipeline on store
 *  - Auto-forgetting: temporal bounds, sweep timer
 *  - Static/dynamic classification
 *  - Typed relationships: updates, extends, derives, similarity
 *  - Source count reinforcement: duplicates boost existing memories
 *  - Relationship expansion in search
 *  - Dynamic profile generation
 *  - Keyword detection: auto-nudge on "remember", "save this", etc.
 *  - <private> tag support: strip private content before storing
 *
 * Features:
 *  - Auto-recall: On the FIRST message of every session, fetches recent
 *    memories + profile and injects them as context.
 *  - Compaction injection: Re-injects memories during compaction.
 *  - Keyword detection: Nudges agent to store when user says "remember", etc.
 *  - Privacy tags: Strips <private>...</private> content before storing.
 *  - Tools: memory_store, memory_search, memory_list, memory_delete
 *  - Source tagging: Every memory is tagged with hostname for traceability.
 *  - Conversations DB: Full conversation storage with FTS search.
 */

const DEFAULT_URL = "http://100.64.0.1:4200";
const URL_FILE = join(
  process.env.HOME || process.env.USERPROFILE || ".",
  ".config",
  "opencode",
  "megamind-url.txt"
);

let _cachedUrl: string | null = null;
let _cacheTime = 0;
const CACHE_TTL = 30_000; // re-read file every 30s

function getMemoryUrl(): string {
  const now = Date.now();
  if (_cachedUrl && now - _cacheTime < CACHE_TTL) return _cachedUrl;
  try {
    const fileUrl = readFileSync(URL_FILE, "utf-8").trim();
    if (fileUrl) {
      _cachedUrl = fileUrl;
      _cacheTime = now;
      return fileUrl;
    }
  } catch {}
  _cachedUrl =
    process.env.MEGAMIND_URL ||
    process.env.OPUS_MEMORY_URL ||
    DEFAULT_URL;
  _cacheTime = now;
  return _cachedUrl;
}

const SOURCE_TAG = `opencode@${hostname()}`;
const RECALL_LIMIT = 25;
const SEARCH_ON_FIRST_MSG = true;

// ================================================================
// Keyword detection — trigger memory storage nudge
// ================================================================
const CODE_BLOCK_PATTERN = /```[\s\S]*?```/g;
const INLINE_CODE_PATTERN = /`[^`]+`/g;

const MEMORY_KEYWORDS = [
  "remember",
  "remember this",
  "save this",
  "don't forget",
  "do not forget",
  "dont forget",
  "note that",
  "note this",
  "keep in mind",
  "store this",
  "memorize",
  "write this down",
  "for future reference",
  "for next time",
  "always use",
  "never use",
  "from now on",
  "going forward",
  "important to know",
];

const MEMORY_KEYWORD_PATTERN = new RegExp(
  `\\b(${MEMORY_KEYWORDS.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`,
  "i"
);

const MEMORY_NUDGE_MESSAGE = `[MEMORY TRIGGER DETECTED]
The user wants you to remember something. You MUST use the \`memory_store\` tool to save this information NOW.

Extract the key information the user wants remembered and save it as a concise, searchable memory.
- Category "decision" for preferences, rules, choices
- Category "state" for current status of things
- Category "discovery" for learned info
- Category "issue" for known problems

DO NOT skip this step. The user explicitly asked you to remember.`;

function removeCodeBlocks(text: string): string {
  return text.replace(CODE_BLOCK_PATTERN, "").replace(INLINE_CODE_PATTERN, "");
}

function detectMemoryKeyword(text: string): boolean {
  const cleaned = removeCodeBlocks(text);
  return MEMORY_KEYWORD_PATTERN.test(cleaned);
}

// ================================================================
// Privacy tags — strip <private>...</private> content
// ================================================================
const PRIVATE_TAG_PATTERN = /<private>[\s\S]*?<\/private>/gi;

function stripPrivateContent(text: string): string {
  return text.replace(PRIVATE_TAG_PATTERN, "[PRIVATE]").trim();
}

function isFullyPrivate(text: string): boolean {
  const stripped = text.replace(PRIVATE_TAG_PATTERN, "").trim();
  return stripped.length === 0;
}

// ================================================================
// Interfaces — v3 response types
// ================================================================
interface MemoryLink {
  id: number;
  content: string;
  category: string;
  similarity?: number;
  type?: "similarity" | "updates" | "extends" | "derives";
}

interface Memory {
  id: number;
  content: string;
  category: string;
  source: string;
  session_id: string | null;
  importance: number;
  created_at: string;
  score?: number;
  // v3 fields
  version?: number;
  is_latest?: boolean | number;
  parent_memory_id?: number | null;
  root_memory_id?: number | null;
  source_count?: number;
  is_static?: boolean | number;
  is_forgotten?: boolean | number;
  is_inference?: boolean | number;
  forget_after?: string | null;
  // linked memories (from search with include_links)
  linked?: MemoryLink[];
}

interface ListResponse {
  results: Memory[];
}

interface StoreResponse {
  stored: boolean;
  id: number;
  created_at: string;
  importance: number;
  linked: number;
  embedded: boolean;
  // v3 fields
  fact_extraction?: string; // "pending" when async extraction triggered
}

interface SearchResponse {
  results: Memory[];
}

interface ProfileResponse {
  static_facts: Array<{
    id: number;
    content: string;
    category: string;
    source_count: number;
    created_at: string;
  }>;
  recent_activity: Array<{
    id: number;
    content: string;
    category: string;
    created_at: string;
  }>;
  summary?: string;
}

interface HealthResponse {
  status: string;
  version?: number;
  memories: number;
  conversations: number;
  messages: number;
  db_size_mb: number;
  embedded_count?: number;
  unembedded_count?: number;
  embedded?: number;
  unembedded?: number;
  links_count?: number;
  links?: number;
  embedding_model?: string;
  // v3 fields
  forgotten?: number;
  static?: number;
  versioned?: number;
  llm_model?: string;
  llm_configured?: boolean;
}

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(`${getMemoryUrl()}${path}`, {
      ...opts,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...opts?.headers },
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`MegaMind API error ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  } finally {
    clearTimeout(timeout);
  }
}

// ================================================================
// Formatting — context blocks with v3 metadata
// ================================================================

function formatBadges(m: Memory): string {
  const badges: string[] = [];
  if (m.is_static) badges.push("STATIC");
  if (m.is_forgotten) badges.push("FORGOTTEN");
  if (m.is_inference) badges.push("INFERRED");
  if (m.version && m.version > 1) badges.push(`v${m.version}`);
  if (m.source_count && m.source_count > 1) badges.push(`x${m.source_count}`);
  if (m.is_latest === false || m.is_latest === 0) badges.push("SUPERSEDED");
  return badges.length > 0 ? ` {${badges.join(", ")}}` : "";
}

function formatLinkedSummary(m: Memory): string {
  if (!m.linked || m.linked.length === 0) return "";
  const linkStrs = m.linked.map((l) => {
    const typeLabel = l.type && l.type !== "similarity" ? `[${l.type}]` : "";
    const simPct = l.similarity ? `${(l.similarity * 100).toFixed(0)}%` : "";
    return `#${l.id}${typeLabel}${simPct ? ` ${simPct}` : ""}`;
  });
  return ` -> ${linkStrs.join(", ")}`;
}

function formatMemoryBlock(memories: Memory[], heading: string): string {
  if (!memories.length) return "";
  const lines = memories.map((m) => {
    const badges = formatBadges(m);
    const links = formatLinkedSummary(m);
    return `- [#${m.id} | ${m.created_at} | ${m.category}${badges}] ${m.content}${links}`;
  });
  return `<megamind-context>
## ${heading}
These are persistent memories from MegaMind (Zan's self-hosted memory DB).
They contain prior session work, infrastructure state, decisions, and discoveries.
Use them to maintain continuity — do NOT re-ask for context already here.

${lines.join("\n")}
</megamind-context>`;
}

function formatProfileBlock(profile: ProfileResponse): string {
  const parts: string[] = [];

  if (profile.static_facts.length > 0) {
    parts.push("### Static Facts (permanent knowledge)");
    for (const f of profile.static_facts) {
      const countBadge = f.source_count > 1 ? ` (confirmed x${f.source_count})` : "";
      parts.push(`- [#${f.id} | ${f.category}${countBadge}] ${f.content}`);
    }
  }

  if (profile.recent_activity.length > 0) {
    parts.push("### Recent Activity");
    for (const a of profile.recent_activity.slice(0, 10)) {
      parts.push(`- [#${a.id} | ${a.created_at} | ${a.category}] ${a.content}`);
    }
  }

  if (parts.length === 0) return "";

  return `<megamind-profile>
## MegaMind Profile
${parts.join("\n")}
</megamind-profile>`;
}

// ================================================================
// Plugin export
// ================================================================

export const MegaMind: Plugin = async (ctx) => {
  const injectedSessions = new Set<string>();

  // Fetch recent memories (best-effort, never throws to caller)
  async function recallMemories(limit = RECALL_LIMIT): Promise<Memory[]> {
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

  // Search memories by query (best-effort)
  async function searchMemories(
    query: string,
    limit = 10
  ): Promise<Memory[]> {
    try {
      const { results } = await apiFetch<SearchResponse>("/search", {
        method: "POST",
        body: JSON.stringify({
          query,
          limit,
          include_links: true,
          expand_relationships: true,
          latest_only: false,
        }),
      });
      return results || [];
    } catch {
      return [];
    }
  }

  // Fetch profile (best-effort)
  async function fetchProfile(): Promise<ProfileResponse | null> {
    try {
      return await apiFetch<ProfileResponse>("/profile");
    } catch {
      return null;
    }
  }

  // Health check on startup
  (async () => {
    try {
      const health = await apiFetch<HealthResponse>("/health");
      const url = getMemoryUrl();
      const v3Info = health.version
        ? ` v${health.version}, LLM: ${health.llm_configured ? health.llm_model || "configured" : "none"}`
        : "";
      await ctx.client.app.log({
        body: {
          service: "megamind",
          level: "info",
          message: `Connected to MegaMind at ${url}${v3Info} — ${health.memories} memories, ${health.conversations} conversations`,
        },
      });
    } catch (e: any) {
      const url = getMemoryUrl();
      await ctx.client.app.log({
        body: {
          service: "megamind",
          level: "error",
          message: `Cannot reach MegaMind at ${url}: ${e.message}`,
        },
      });
    }
  })();

  return {
    // ================================================================
    // AUTO-RECALL + KEYWORD DETECTION on every message
    // ================================================================
    "chat.message": async (input, output) => {
      const sessionID = input.sessionID;

      // Extract user message text
      const textParts = output.parts.filter(
        (p): p is Part & { type: "text"; text: string } => p.type === "text"
      );
      const userMessage = textParts.map((p) => p.text).join("\n").trim();

      // --- Keyword detection (every message, not just first) ---
      if (userMessage && detectMemoryKeyword(userMessage)) {
        const nudgePart: Part = {
          id: `megamind-nudge-${Date.now()}`,
          sessionID,
          messageID: output.message.id,
          type: "text",
          text: MEMORY_NUDGE_MESSAGE,
          synthetic: true,
        };
        output.parts.push(nudgePart);

        await ctx.client.app.log({
          body: {
            service: "megamind",
            level: "info",
            message: `Memory keyword detected in session ${sessionID}`,
          },
        });
      }

      // --- First message: inject context ---
      if (injectedSessions.has(sessionID)) return;
      injectedSessions.add(sessionID);

      try {
        // Fetch recent memories, profile, and search-relevant ones in parallel
        const [recentMemories, relevantMemories, profile] = await Promise.all([
          recallMemories(RECALL_LIMIT),
          userMessage.length > 5 && SEARCH_ON_FIRST_MSG
            ? searchMemories(userMessage, 10)
            : Promise.resolve([] as Memory[]),
          fetchProfile(),
        ]);

        // Deduplicate: search results may overlap with recent
        const seenIds = new Set(recentMemories.map((m) => m.id));
        const extraRelevant = relevantMemories.filter(
          (m) => !seenIds.has(m.id)
        );

        const allMemories = [...recentMemories, ...extraRelevant];
        const contextParts: Part[] = [];

        // Profile block (if has static facts or recent activity)
        if (profile && (profile.static_facts.length > 0 || profile.recent_activity.length > 0)) {
          const profileBlock = formatProfileBlock(profile);
          if (profileBlock) {
            contextParts.push({
              id: `megamind-profile-${Date.now()}`,
              sessionID,
              messageID: output.message.id,
              type: "text",
              text: profileBlock,
              synthetic: true,
            });
          }
        }

        // Memory context block
        if (allMemories.length > 0) {
          const contextBlock = formatMemoryBlock(
            allMemories,
            "MegaMind — Persistent Context (Auto-Recalled)"
          );
          contextParts.push({
            id: `megamind-recall-${Date.now()}`,
            sessionID,
            messageID: output.message.id,
            type: "text",
            text: contextBlock,
            synthetic: true,
          });
        }

        // Prepend so agent sees context before the user's message
        if (contextParts.length > 0) {
          output.parts.unshift(...contextParts);
        }

        const totalInjected = allMemories.length + (profile?.static_facts.length || 0);
        await ctx.client.app.log({
          body: {
            service: "megamind",
            level: "info",
            message: `Injected ${allMemories.length} memories${profile?.static_facts.length ? ` + ${profile.static_facts.length} static facts` : ""} into session ${sessionID}`,
          },
        });
      } catch (e: any) {
        await ctx.client.app.log({
          body: {
            service: "megamind",
            level: "error",
            message: `Auto-recall failed: ${e.message}`,
          },
        });
      }
    },

    // ================================================================
    // COMPACTION: Re-inject memories so they survive context resets
    // ================================================================
    "experimental.session.compacting": async (_input, output) => {
      const [memories, profile] = await Promise.all([
        recallMemories(30),
        fetchProfile(),
      ]);

      if (profile && (profile.static_facts.length > 0 || profile.recent_activity.length > 0)) {
        const profileBlock = formatProfileBlock(profile);
        if (profileBlock) {
          output.context.push(profileBlock);
        }
      }

      if (memories.length > 0) {
        output.context.push(
          formatMemoryBlock(memories, "MegaMind — Persistent Context")
        );
      }
    },

    // ================================================================
    // TOOLS: memory_store, memory_search, memory_list, memory_delete
    // ================================================================
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
            // Check for fully private content
            if (isFullyPrivate(args.content)) {
              return "Cannot store memory: content is entirely within <private> tags.";
            }

            // Strip private sections before storing
            const sanitizedContent = stripPrivateContent(args.content);

            const result = await apiFetch<StoreResponse>("/store", {
              method: "POST",
              body: JSON.stringify({
                content: sanitizedContent,
                category: args.category,
                source: SOURCE_TAG,
                session_id: context.sessionID,
              }),
            });
            const linkMsg = result.linked > 0 ? `, auto-linked to ${result.linked} related memories` : "";
            const extractMsg = result.fact_extraction === "pending" ? " (fact extraction pending)" : "";
            return `Stored memory #${result.id} (${args.category}, importance ${result.importance})${linkMsg}${extractMsg}: ${sanitizedContent.substring(0, 80)}...`;
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
                include_links: true,
                expand_relationships: true,
              }),
            });
            if (!result.results?.length) {
              return "No memories found matching that query.";
            }
            return result.results
              .map((m) => {
                const badges = formatBadges(m);
                let line = `[#${m.id}] ${m.created_at} (${m.category}${badges}) ${m.content}`;
                if (m.linked && m.linked.length > 0) {
                  const linkLines = m.linked.map((l) => {
                    const typeLabel = l.type || "similarity";
                    const simPct = l.similarity ? ` ${(l.similarity * 100).toFixed(0)}%` : "";
                    return `    #${l.id} [${typeLabel}${simPct}] ${l.content.substring(0, 60)}...`;
                  });
                  line += `\n  Links:\n${linkLines.join("\n")}`;
                }
                return line;
              })
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
              .map((m) => {
                const badges = formatBadges(m);
                return `[#${m.id}] ${m.created_at} (${m.category}${badges}) ${m.content}`;
              })
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
