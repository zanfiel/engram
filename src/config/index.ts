// ============================================================================
// CONFIG — All env vars, constants, feature flags
// ============================================================================

import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { mkdirSync } from "fs";

const __dirname = dirname(fileURLToPath(import.meta.url));

export const DATA_DIR = resolve(__dirname, "../../data");
export const DB_PATH = resolve(DATA_DIR, "memory.db");
export const PORT = Number(process.env.ENGRAM_PORT || process.env.ZANMEMORY_PORT || 4200);
export const HOST = process.env.ENGRAM_HOST || process.env.ZANMEMORY_HOST || "0.0.0.0";

// Embedding config
export const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
export const EMBEDDING_DIM = 384;
export const AUTO_LINK_THRESHOLD = 0.7;
export const AUTO_LINK_MAX = 3;
export const DEFAULT_IMPORTANCE = 5;

// LLM config (for fact extraction)
export const LLM_URL = process.env.LLM_URL || "http://127.0.0.1:4100/v1/chat/completions";
export const LLM_API_KEY = process.env.LLM_API_KEY || "";
export const LLM_MODEL = process.env.LLM_MODEL || "claude-sonnet-4-20250514";

// Auto-forget sweep interval (every 5 minutes)
export const FORGET_SWEEP_INTERVAL = 5 * 60 * 1000;

// FSRS-6 configuration
export const FSRS_DEFAULT_RETENTION = 0.9;
export const CONSOLIDATION_THRESHOLD = 8;
export const CONSOLIDATION_INTERVAL = 30 * 60 * 1000;

// Reranker config
export const RERANKER_ENABLED = process.env.RERANKER !== "0";
export const RERANKER_TOP_K = Number(process.env.RERANKER_TOP_K || 20);

// API key config
export const API_KEY_PREFIX = "eg_";
export const DEFAULT_RATE_LIMIT = 120;
export const RATE_WINDOW_MS = 60_000;

// Security config
export const OPEN_ACCESS = process.env.ENGRAM_OPEN_ACCESS === "1";
export const CORS_ORIGIN = process.env.ENGRAM_CORS_ORIGIN || "*";
export const MAX_BODY_SIZE = Number(process.env.ENGRAM_MAX_BODY_SIZE || 1_048_576);
export const MAX_CONTENT_SIZE = Number(process.env.ENGRAM_MAX_CONTENT_SIZE || 102_400);
export const ALLOWED_IPS = (process.env.ENGRAM_ALLOWED_IPS || "").split(",").map(s => s.trim()).filter(Boolean);
export const GUI_AUTH_MAX_ATTEMPTS = 5;
export const GUI_AUTH_WINDOW_MS = 60_000;
export const GUI_AUTH_LOCKOUT_MS = 600_000;
export const OPEN_ACCESS_RATE_LIMIT = Number(process.env.ENGRAM_OPEN_RATE_LIMIT || 120);

// Logging config
export const LOG_LEVEL_MAP: Record<string, number> = { debug: 0, info: 1, warn: 2, error: 3, none: 4 };
export const LOG_LEVEL = LOG_LEVEL_MAP[process.env.ENGRAM_LOG_LEVEL || "info"] ?? 1;

// Tier 4: Novel feature flags
export const ENABLE_CAUSAL_CHAINS = process.env.ENGRAM_CAUSAL_CHAINS !== "0";
export const ENABLE_PREDICTIVE_RECALL = process.env.ENGRAM_PREDICTIVE_RECALL !== "0";
export const ENABLE_EMOTIONAL_VALENCE = process.env.ENGRAM_EMOTIONAL_VALENCE !== "0";
export const ENABLE_RECONSOLIDATION = process.env.ENGRAM_RECONSOLIDATION !== "0";
export const ENABLE_ADAPTIVE_IMPORTANCE = process.env.ENGRAM_ADAPTIVE_IMPORTANCE !== "0";
export const RECONSOLIDATION_INTERVAL = Number(process.env.ENGRAM_RECONSOLIDATION_INTERVAL || 60 * 60 * 1000); // 1 hour

// Ensure data directory exists
mkdirSync(DATA_DIR, { recursive: true });
