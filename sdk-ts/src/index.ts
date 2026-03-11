/**
 * Engram SDK — Persistent memory for AI agents
 *
 * @example
 * ```ts
 * import { Engram } from "@engram/sdk";
 *
 * const engram = new Engram({
 *   url: "http://localhost:4200",
 *   apiKey: "eg_your_key_here",
 * });
 *
 * // Store a memory
 * await engram.store("User prefers dark mode", { category: "decision" });
 *
 * // Search memories
 * const results = await engram.search("user preferences");
 *
 * // Recall context for an agent
 * const context = await engram.recall("What does the user like?");
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export type Category = "task" | "discovery" | "decision" | "state" | "issue" | "general";

export interface EngramConfig {
  /** Engram server URL (e.g., "http://localhost:4200") */
  url: string;
  /** API key with eg_ prefix */
  apiKey?: string;
  /** Memory space name (uses default if not set) */
  space?: string;
  /** Source identifier for stored memories */
  source?: string;
  /** Request timeout in ms (default: 30000) */
  timeout?: number;
}

export interface Memory {
  id: number;
  content: string;
  category: Category;
  source: string;
  session_id: string | null;
  importance: number;
  created_at: string;
  version: number;
  is_latest: boolean;
  parent_memory_id: number | null;
  root_memory_id: number | null;
  source_count: number;
  is_static: boolean;
  is_forgotten: boolean;
  is_inference: boolean;
  is_archived: boolean;
  forget_after: string | null;
}

export interface SearchResult extends Memory {
  score: number;
}

export interface StoreOptions {
  category?: Category;
  importance?: number;
  source?: string;
  session_id?: string;
}

export interface StoreResponse {
  stored: boolean;
  id: number;
  created_at: string;
  importance: number;
  linked: number;
  embedded: boolean;
  fact_extraction: string;
}

export interface SearchOptions {
  limit?: number;
  threshold?: number;
}

export interface RecallOptions {
  limit?: number;
}

export interface RecallResponse {
  memories: Memory[];
  breakdown: {
    static_facts: number;
    semantic_matches: number;
    important: number;
    recent: number;
  };
}

export interface ListOptions {
  limit?: number;
  category?: Category;
}

export interface Profile {
  static_facts: Array<{
    id: number;
    content: string;
    category: string;
    source_count: number;
  }>;
  recent_activity: Array<{
    id: number;
    content: string;
    category: string;
  }>;
  summary?: string;
}

export interface UpdateResponse {
  updated: boolean;
  new_id: number;
  old_id: number;
  version: number;
}

export interface HealthResponse {
  status: string;
  version: number;
  memories: number;
  embedded: number;
  links: number;
  forgotten: number;
  archived: number;
  conversations: number;
  messages: number;
}

export interface ExportData {
  version: string;
  exported_at: string;
  memories: Memory[];
  links: Array<{
    source_id: number;
    target_id: number;
    similarity: number;
    type: string;
  }>;
  stats: { memory_count: number; link_count: number };
}

export interface ImportResult {
  imported: number;
  failed: number;
  total: number;
}

export interface Space {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  memory_count: number;
}

export interface ApiKeyInfo {
  id: number;
  key_prefix: string;
  name: string;
  scopes: string;
  rate_limit: number;
  is_active: boolean;
  last_used_at: string | null;
  created_at: string;
}

// ============================================================================
// SDK Client
// ============================================================================

export class Engram {
  private url: string;
  private apiKey?: string;
  private space?: string;
  private source: string;
  private timeout: number;

  constructor(config: EngramConfig) {
    this.url = config.url.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.space = config.space;
    this.source = config.source || "sdk";
    this.timeout = config.timeout || 30_000;
  }

  // --------------------------------------------------------------------------
  // Internal
  // --------------------------------------------------------------------------

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    if (this.space) h["X-Space"] = this.space;
    return h;
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const res = await fetch(`${this.url}${path}`, {
        method,
        headers: this.headers(),
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new EngramError((err as any).error || res.statusText, res.status);
      }

      return await res.json() as T;
    } finally {
      clearTimeout(timer);
    }
  }

  // --------------------------------------------------------------------------
  // Core Memory Operations
  // --------------------------------------------------------------------------

  /** Store a new memory */
  async store(content: string, options: StoreOptions = {}): Promise<StoreResponse> {
    return this.request("POST", "/store", {
      content,
      category: options.category || "general",
      importance: options.importance,
      source: options.source || this.source,
      session_id: options.session_id,
    });
  }

  /** Semantic search across memories */
  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    const res = await this.request<{ results: SearchResult[] }>("POST", "/search", {
      query,
      limit: options.limit || 10,
      threshold: options.threshold,
    });
    return res.results;
  }

  /** Contextual recall — optimized for agent context injection */
  async recall(query: string, options: RecallOptions = {}): Promise<RecallResponse> {
    return this.request("POST", "/recall", {
      query,
      limit: options.limit || 25,
    });
  }

  /** List recent memories */
  async list(options: ListOptions = {}): Promise<Memory[]> {
    const params = new URLSearchParams();
    if (options.limit) params.set("limit", String(options.limit));
    if (options.category) params.set("category", options.category);
    const qs = params.toString();
    const res = await this.request<{ results: Memory[] }>("GET", `/list${qs ? `?${qs}` : ""}`);
    return res.results;
  }

  /** Get user profile (static facts + recent activity) */
  async profile(summary = false): Promise<Profile> {
    return this.request("GET", `/profile${summary ? "?summary=true" : ""}`);
  }

  /** Get full memory graph */
  async graph(): Promise<{ memories: Memory[]; links: Array<{ source_id: number; target_id: number; similarity: number; type: string }> }> {
    return this.request("GET", "/graph");
  }

  // --------------------------------------------------------------------------
  // Memory Management
  // --------------------------------------------------------------------------

  /** Update a memory (creates a new version) */
  async update(id: number, content: string, category?: Category): Promise<UpdateResponse> {
    return this.request("POST", `/memory/${id}/update`, { content, category });
  }

  /** Forget a memory (soft delete) */
  async forget(id: number): Promise<void> {
    await this.request("POST", `/memory/${id}/forget`);
  }

  /** Archive a memory (hidden from recall, still searchable) */
  async archive(id: number): Promise<void> {
    await this.request("POST", `/memory/${id}/archive`);
  }

  /** Unarchive a memory */
  async unarchive(id: number): Promise<void> {
    await this.request("POST", `/memory/${id}/unarchive`);
  }

  /** Delete a memory permanently */
  async delete(id: number): Promise<void> {
    await this.request("DELETE", `/memory/${id}`);
  }

  // --------------------------------------------------------------------------
  // Data Operations
  // --------------------------------------------------------------------------

  /** Export all memories and links */
  async export(format: "json" | "jsonl" = "json"): Promise<ExportData> {
    return this.request("GET", `/export?format=${format}`);
  }

  /** Bulk import memories */
  async import(memories: Array<{ content: string; category?: Category; source?: string; importance?: number }>): Promise<ImportResult> {
    return this.request("POST", "/import", { memories });
  }

  // --------------------------------------------------------------------------
  // Spaces
  // --------------------------------------------------------------------------

  /** List memory spaces */
  async listSpaces(): Promise<Space[]> {
    const res = await this.request<{ spaces: Space[] }>("GET", "/spaces");
    return res.spaces;
  }

  /** Create a memory space */
  async createSpace(name: string, description?: string): Promise<Space> {
    return this.request("POST", "/spaces", { name, description });
  }

  /** Delete a memory space */
  async deleteSpace(id: number): Promise<void> {
    await this.request("DELETE", `/spaces/${id}`);
  }

  /** Switch to a different space */
  useSpace(name: string): this {
    this.space = name;
    return this;
  }

  // --------------------------------------------------------------------------
  // API Keys
  // --------------------------------------------------------------------------

  /** List API keys (values not returned) */
  async listKeys(): Promise<ApiKeyInfo[]> {
    const res = await this.request<{ keys: ApiKeyInfo[] }>("GET", "/keys");
    return res.keys;
  }

  /** Create a new API key (key value returned only once) */
  async createKey(options: { name?: string; scopes?: string; rate_limit?: number; user_id?: number } = {}): Promise<{ key: string; name: string; scopes: string; rate_limit: number }> {
    return this.request("POST", "/keys", options);
  }

  /** Revoke an API key */
  async revokeKey(id: number): Promise<void> {
    await this.request("DELETE", `/keys/${id}`);
  }

  // --------------------------------------------------------------------------
  // Maintenance
  // --------------------------------------------------------------------------

  /** Find duplicate memory clusters */
  async duplicates(threshold = 0.92, limit = 20): Promise<any> {
    return this.request("GET", `/duplicates?threshold=${threshold}&limit=${limit}`);
  }

  /** Auto-deduplicate memories */
  async deduplicate(options: { threshold?: number; dry_run?: boolean } = {}): Promise<any> {
    return this.request("POST", "/deduplicate", options);
  }

  /** Trigger forget sweep */
  async sweep(): Promise<any> {
    return this.request("POST", "/sweep");
  }

  /** Backfill missing embeddings */
  async backfill(): Promise<any> {
    return this.request("POST", "/backfill");
  }

  // --------------------------------------------------------------------------
  // Conversations
  // --------------------------------------------------------------------------

  /** Create a conversation */
  async createConversation(options: { agent: string; session_id?: string; model?: string; title?: string; started_at?: string }): Promise<any> {
    return this.request("POST", "/conversations", options);
  }

  /** List conversations */
  async listConversations(options: { limit?: number; offset?: number; agent?: string } = {}): Promise<any> {
    const params = new URLSearchParams();
    if (options.limit) params.set("limit", String(options.limit));
    if (options.offset) params.set("offset", String(options.offset));
    if (options.agent) params.set("agent", options.agent);
    return this.request("GET", `/conversations?${params}`);
  }

  /** Bulk insert a conversation with messages */
  async conversationsBulk(options: { agent: string; session_id?: string; model?: string; messages: Array<{ role: string; content: string }> }): Promise<any> {
    return this.request("POST", "/conversations/bulk", options);
  }

  /** Search conversation messages */
  async searchMessages(query: string, options: { limit?: number; agent?: string } = {}): Promise<any> {
    return this.request("POST", "/messages/search", { query, ...options });
  }

  // --------------------------------------------------------------------------
  // Episodes
  // --------------------------------------------------------------------------

  /** Create an episode from memory IDs */
  async createEpisode(name: string, memoryIds: number[]): Promise<any> {
    return this.request("POST", "/episodes", { name, memory_ids: memoryIds });
  }

  /** List episodes */
  async listEpisodes(limit = 20): Promise<any> {
    return this.request("GET", `/episodes?limit=${limit}`);
  }

  // --------------------------------------------------------------------------
  // Tags
  // --------------------------------------------------------------------------

  /** Set tags on a memory */
  async setTags(id: number, tags: string[]): Promise<any> {
    return this.request("PUT", `/memory/${id}/tags`, { tags });
  }

  /** List all tags */
  async listTags(): Promise<any> {
    return this.request("GET", "/tags");
  }

  /** Search memories by tag */
  async searchByTag(tag: string): Promise<any> {
    return this.request("POST", "/tags/search", { tag });
  }

  // --------------------------------------------------------------------------
  // FSRS & Decay
  // --------------------------------------------------------------------------

  /** Get decay scores */
  async decayScores(limit = 20): Promise<any> {
    return this.request("GET", `/decay/scores?limit=${limit}`);
  }

  /** Refresh decay calculations */
  async decayRefresh(): Promise<any> {
    return this.request("POST", "/decay/refresh");
  }

  /** Record an FSRS review */
  async fsrsReview(id: number, grade: 1 | 2 | 3 | 4): Promise<any> {
    return this.request("POST", "/fsrs/review", { id, grade });
  }

  /** Get FSRS state for a memory */
  async fsrsState(id: number): Promise<any> {
    return this.request("GET", `/fsrs/state?id=${id}`);
  }

  // --------------------------------------------------------------------------
  // Pack & Context
  // --------------------------------------------------------------------------

  /** Pack memories into a token-budgeted context string */
  async pack(query: string, maxTokens = 4000): Promise<any> {
    return this.request("POST", "/pack", { query, max_tokens: maxTokens });
  }

  /** Get a formatted prompt template */
  async prompt(template: string, query: string): Promise<any> {
    return this.request("GET", `/prompt?template=${encodeURIComponent(template)}&query=${encodeURIComponent(query)}`);
  }

  /** Get smart context with blocks and token budget */
  async context(query: string, maxTokens = 4000): Promise<any> {
    return this.request("POST", "/context", { query, max_tokens: maxTokens });
  }

  // --------------------------------------------------------------------------
  // Time Travel & Contradictions
  // --------------------------------------------------------------------------

  /** View memory state at a point in time */
  async timeTravel(asOf: string): Promise<any> {
    return this.request("POST", "/timetravel", { as_of: asOf });
  }

  /** List detected contradictions */
  async contradictions(limit = 20): Promise<any> {
    return this.request("GET", `/contradictions?limit=${limit}`);
  }

  // --------------------------------------------------------------------------
  // Webhooks
  // --------------------------------------------------------------------------

  /** Register a webhook */
  async createWebhook(url: string, events: string[]): Promise<any> {
    return this.request("POST", "/webhooks", { url, events });
  }

  /** List webhooks */
  async listWebhooks(): Promise<any> {
    return this.request("GET", "/webhooks");
  }

  // --------------------------------------------------------------------------
  // Sync
  // --------------------------------------------------------------------------

  /** Get changes since a timestamp for syncing */
  async syncChanges(since: string): Promise<any> {
    return this.request("GET", `/sync/changes?since=${encodeURIComponent(since)}`);
  }

  // --------------------------------------------------------------------------
  // Inbox & Audit
  // --------------------------------------------------------------------------

  /** Get pending inbox items */
  async inbox(limit = 20): Promise<any> {
    return this.request("GET", `/inbox?limit=${limit}`);
  }

  /** Get audit log */
  async audit(limit = 50): Promise<any> {
    return this.request("GET", `/audit?limit=${limit}`);
  }

  // --------------------------------------------------------------------------
  // Entities & Projects
  // --------------------------------------------------------------------------

  /** Create an entity */
  async createEntity(name: string, type: string, description?: string): Promise<any> {
    return this.request("POST", "/entities", { name, type, description });
  }

  /** List entities */
  async listEntities(): Promise<any> {
    return this.request("GET", "/entities");
  }

  /** Create a project */
  async createProject(name: string, description?: string): Promise<any> {
    return this.request("POST", "/projects", { name, description });
  }

  /** List projects */
  async listProjects(): Promise<any> {
    return this.request("GET", "/projects");
  }

  // --------------------------------------------------------------------------
  // Digests
  // --------------------------------------------------------------------------

  /** Create a scheduled digest */
  async createDigest(options: { name: string; cron: string; webhook_url: string }): Promise<any> {
    return this.request("POST", "/digests", options);
  }

  /** List digest schedules */
  async listDigests(): Promise<any> {
    return this.request("GET", "/digests");
  }

  // --------------------------------------------------------------------------
  // Users
  // --------------------------------------------------------------------------

  /** Create a user */
  async createUser(username: string): Promise<any> {
    return this.request("POST", "/users", { username });
  }

  /** List users */
  async listUsers(): Promise<any> {
    return this.request("GET", "/users");
  }

  // --------------------------------------------------------------------------
  // Version History & Links
  // --------------------------------------------------------------------------

  /** Get version chain for a root memory */
  async versions(rootId: number): Promise<any> {
    return this.request("GET", `/versions/${rootId}`);
  }

  /** Get links for a memory */
  async links(id: number): Promise<any> {
    return this.request("GET", `/links/${id}`);
  }

  // --------------------------------------------------------------------------
  // URL Ingest
  // --------------------------------------------------------------------------

  /** Ingest content from a URL */
  async ingest(url: string): Promise<any> {
    return this.request("POST", "/ingest", { url });
  }

  // --------------------------------------------------------------------------
  // Backup
  // --------------------------------------------------------------------------

  /** Download database backup as ArrayBuffer */
  async backup(): Promise<ArrayBuffer> {
    const resp = await fetch(`${this.config.url}/backup`, {
      headers: this.config.apiKey ? { Authorization: `Bearer ${this.config.apiKey}` } : {},
    });
    if (!resp.ok) throw new EngramError(`Backup failed: ${resp.status}`, resp.status);
    return resp.arrayBuffer();
  }

  // --------------------------------------------------------------------------
  // System
  // --------------------------------------------------------------------------

  /** Health check */
  async health(): Promise<HealthResponse> {
    return this.request("GET", "/health");
  }

  /** Detailed stats */
  async stats(): Promise<any> {
    return this.request("GET", "/stats");
  }
}

// ============================================================================
// Error class
// ============================================================================

export class EngramError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "EngramError";
    this.status = status;
  }
}

// ============================================================================
// Default export
// ============================================================================

export default Engram;
