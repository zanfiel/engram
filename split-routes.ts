// Extract routes from server.ts.monolith into src/routes/index.ts
// Run: node --experimental-strip-types split-routes.ts

import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { resolve } from "path";

const SRC = readFileSync(resolve(import.meta.dirname!, "server.ts.monolith"), "utf8");
const lines = SRC.split("\n");

function extract(start: number, end: number): string {
  return lines.slice(start - 1, end).join("\n");
}

mkdirSync(resolve(import.meta.dirname!, "src/routes"), { recursive: true });

// ============================================================================
// src/routes/index.ts — The fetch handler with all routes
// Lines 2881-7283 from monolith (fetchHandler function)
// ============================================================================

const IMPORT_HEADER = `// ============================================================================
// ROUTES — All HTTP request handling
// Auto-extracted from server.ts.monolith lines 2881-7283
// ============================================================================

import { readFileSync, statSync, copyFileSync, existsSync, unlinkSync } from "fs";
import { readFile } from "fs/promises";
import { randomUUID, timingSafeEqual } from "crypto";
import { resolve } from "path";

// Config
import {
  PORT, HOST, OPEN_ACCESS, CORS_ORIGIN, MAX_BODY_SIZE, MAX_CONTENT_SIZE,
  ALLOWED_IPS, LLM_URL, LLM_API_KEY, LLM_MODEL, AUTO_LINK_THRESHOLD, AUTO_LINK_MAX,
  DEFAULT_IMPORTANCE, RERANKER_ENABLED, RERANKER_TOP_K, DATA_DIR, DB_PATH,
  CONSOLIDATION_THRESHOLD, RATE_WINDOW_MS, OPEN_ACCESS_RATE_LIMIT,
  GUI_AUTH_MAX_ATTEMPTS, GUI_AUTH_WINDOW_MS, GUI_AUTH_LOCKOUT_MS,
  ENABLE_CAUSAL_CHAINS, ENABLE_PREDICTIVE_RECALL, ENABLE_EMOTIONAL_VALENCE,
  ENABLE_RECONSOLIDATION,
} from "../config/index.ts";
import { log } from "../config/logger.ts";

// Database + prepared statements
import {
  db, audit,
  insertMemory, getMemory, listRecent, listByCategory, searchParams,
  insertLink, getLinksFor, markForgotten, markArchived, markUnarchived, markSuperseded,
  getVersionChain, rootLatest, updateMemoryEmbedding, updateMemoryVec,
  getAllTags, getByTag, insertEpisode, getEpisode, getEpisodeBySession, getEpisodeMemories,
  listEpisodes, assignToEpisode, updateEpisode,
  insertConversation, getConversation, getConversationBySession, listConversations,
  listConversationsByAgent, touchConversation, updateConversation, deleteConversation,
  insertMessage, getMessages, searchMessages, bulkInsertConvo,
  getStaticMemories, getRecentDynamicMemories,
  getAllMemoriesForGraph, getAllLinksForGraph,
  countNoEmbedding, getMemoryWithoutEmbedding,
  updateFSRS, getFSRS,
  insertWebhook, listWebhooks, deleteWebhook,
  getChangesSince, getMemoryBySyncId,
  insertEntity, listEntities, listEntitiesByType, getEntity, getEntityMemories,
  getEntityRelationships, updateEntity, deleteEntity,
  linkMemoryEntity, unlinkMemoryEntity, searchEntities,
  insertEntityRelationship, deleteEntityRelationship,
  insertProject, listProjects, listProjectsByStatus, getProject, getProjectMemories,
  updateProject, deleteProject, linkMemoryProject, unlinkMemoryProject,
  listPending, countPending, approveMemory, rejectMemory, deleteMemory,
} from "../db/index.ts";

// Embeddings
import {
  embed, cosineSimilarity, getCachedEmbeddings, addToEmbeddingCache,
  invalidateEmbeddingCache, embeddingToBuffer, bufferToEmbedding, embeddingToVectorJSON,
} from "../embeddings/index.ts";

// Search + linking
import { hybridSearch, autoLink } from "../memory/search.ts";
import { generateProfile } from "../memory/profile.ts";

// FSRS
import { fsrsProcessReview, fsrsRetrievability, fsrsNextInterval } from "../fsrs/index.ts";

// LLM + extraction
import { callLLM, processExtractionResult, rerank } from "../llm/index.ts";
import { fastExtractFacts } from "../intelligence/extraction.ts";
import { runConsolidationSweep, consolidateCluster } from "../intelligence/consolidation.ts";

// Platform
import { emitWebhookEvent } from "../platform/webhooks.ts";
import { buildDigestPayload, sendDigestWebhook, calculateNextSend, processScheduledDigests } from "../platform/digest.ts";

// Helpers
import { securityHeaders, json, errorResponse, safeError, sanitizeFTS } from "../helpers/index.ts";

// Auth
import {
  type AuthContext, type AuthError,
  authenticate, getAuthOrDefault as _getAuthOrDefault, isAuthError, hasScope, generateApiKey,
} from "../auth/index.ts";

// GUI
import {
  GUI_PASSWORD, GUI_COOKIE_MAX_AGE,
  guiSignCookie, guiAuthed, getGuiHtml, getLoginHtml, reloadGuiHtml,
} from "../gui/index.ts";

// Bind guiAuthed into getAuthOrDefault so routes can call it with just (req)
function getAuthOrDefault(req: Request): AuthContext | AuthError | null {
  return _getAuthOrDefault(req, guiAuthed);
}

// Tier 4
import { detectCausalLinks, getCausalHistory } from "../tier4/causal.ts";
import { predictiveRecall, trackTemporalAccess } from "../tier4/predictive.ts";
import { analyzeValence, storeValence, queryByEmotion, getEmotionalProfile } from "../tier4/valence.ts";
import { reconsolidateMemory, runReconsolidationSweep, recordRecallOutcome } from "../tier4/reconsolidation.ts";
`;

// Functions that are defined OUTSIDE fetchHandler but used inside it
// These need to be included in the routes file or passed in
const EXTRA_FUNCTIONS = `
// --- Functions used by routes but not in a module yet ---

// Per-IP rate limiting for OPEN_ACCESS mode
const ipRateLimits = new Map<string, { count: number; reset: number }>();
function checkIpRateLimit(ip: string): { allowed: boolean; retryAfter?: number } {
  if (!OPEN_ACCESS) return { allowed: true };
  const now = Date.now();
  let rl = ipRateLimits.get(ip);
  if (!rl || now > rl.reset) {
    rl = { count: 0, reset: now + RATE_WINDOW_MS };
    ipRateLimits.set(ip, rl);
  }
  rl.count++;
  if (rl.count > OPEN_ACCESS_RATE_LIMIT) {
    return { allowed: false, retryAfter: Math.ceil((rl.reset - now) / 1000) };
  }
  return { allowed: true };
}
setInterval(() => {
  const now = Date.now();
  for (const [ip, rl] of ipRateLimits) {
    if (now > rl.reset) ipRateLimits.delete(ip);
  }
}, 5 * 60 * 1000);

// GUI auth rate limiting state
const guiAuthAttempts = new Map<string, { count: number; first: number; locked_until: number }>();

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// Decay score calculation
function calculateDecayScore(mem: any): number {
  if (mem.fsrs_stability != null) {
    const r = fsrsRetrievability(mem);
    return mem.importance * r;
  }
  return mem.importance * 0.5;
}

function updateDecayScores(): number {
  const all = db.prepare(
    "SELECT id, importance, fsrs_difficulty, fsrs_stability, fsrs_storage_strength, fsrs_retrieval_strength, fsrs_last_review, fsrs_review_count, created_at, access_count FROM memories WHERE is_forgotten = 0 AND is_archived = 0"
  ).all() as any[];
  let updated = 0;
  for (const mem of all) {
    const score = calculateDecayScore(mem);
    db.prepare("UPDATE memories SET decay_score = ? WHERE id = ?").run(score, mem.id);
    updated++;
  }
  return updated;
}

// Track access with FSRS review
function trackAccessWithFSRS(memoryId: number): void {
  try {
    const state = getFSRS.get(memoryId) as any;
    if (state) {
      const newState = fsrsProcessReview(state, 3); // "Good" grade
      updateFSRS.run(
        newState.difficulty, newState.stability,
        newState.storage_strength, newState.retrieval_strength,
        new Date().toISOString(), (state.review_count || 0) + 1,
        memoryId
      );
    }
  } catch {}
}

// Sweep expired memories
function sweepExpiredMemories(): number {
  const getExpiredMemories = db.prepare(
    "SELECT id, content, forget_reason FROM memories WHERE forget_after IS NOT NULL AND forget_after <= datetime('now') AND is_forgotten = 0"
  );
  const expired = getExpiredMemories.all() as Array<{ id: number; content: string; forget_reason: string }>;
  for (const mem of expired) {
    markForgotten.run(mem.id);
    log.debug({ msg: "auto_forgot", id: mem.id, reason: mem.forget_reason || "expired" });
  }
  return expired.length;
}

// Backfill embeddings
async function backfillEmbeddings(batchSize: number = 50): Promise<number> {
  const getNoEmbedding = db.prepare("SELECT id, content FROM memories WHERE embedding IS NULL LIMIT ?");
  const missing = getNoEmbedding.all(batchSize) as Array<{ id: number; content: string }>;
  let count = 0;
  for (const mem of missing) {
    try {
      const emb = await embed(mem.content);
      updateMemoryEmbedding.run(embeddingToBuffer(emb), mem.id);
      try { updateMemoryVec.run(embeddingToVectorJSON(emb), mem.id); } catch {}
      count++;
    } catch (e: any) {
      log.error({ msg: "embed_failed", id: mem.id, error: e.message });
    }
  }
  if (count > 0) {
    for (const mem of missing.slice(0, count)) {
      const row = getMemory.get(mem.id) as any;
      if (row?.embedding) {
        const emb = bufferToEmbedding(row.embedding);
        await autoLink(mem.id, emb);
      }
    }
  }
  return count;
}

// Write vec helper
function writeVec(id: number, emb: Float32Array): void {
  try { updateMemoryVec.run(embeddingToVectorJSON(emb), id); } catch {}
}

export { sweepExpiredMemories, backfillEmbeddings, updateDecayScores };
`;

// Extract the fetchHandler function body
const routeCode = extract(2881, 7283);

const fullContent = IMPORT_HEADER + "\n" + EXTRA_FUNCTIONS + "\n" + routeCode + "\n\nexport { fetchHandler };\n";

writeFileSync(resolve(import.meta.dirname!, "src/routes/index.ts"), fullContent);
console.log(`  src/routes/index.ts (${fullContent.split("\n").length} lines)`);

// ============================================================================
// Also create the GUI initialization module needed by routes
// ============================================================================
// The routes reference guiSignCookie, guiAuthed, getGuiHtml, getLoginHtml
// These are now in src/gui/index.ts but routes still use bare names.
// We need to add them to the import header above (already done).
// The GUI module needs to be initialized with the HMAC secret.
// This is done in the new server.ts entry point.

console.log("\nDone! Routes extracted.");
console.log("Next: create the new server.ts entry point.");
