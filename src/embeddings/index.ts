// ============================================================================
// EMBEDDINGS — Model init, cache, similarity
// ============================================================================

import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { EMBEDDING_MODEL, EMBEDDING_DIM } from "../config/index.ts";
import { log } from "../config/logger.ts";
import { db } from "../db/index.ts";

let embedder: FeatureExtractionPipeline | null = null;

// Prepared statement for cache refresh (imported here to avoid circular dep)
const getAllEmbeddings = db.prepare(
  `SELECT id, user_id, content, category, importance, embedding, is_static, source_count, is_latest, is_forgotten
   FROM memories WHERE embedding IS NOT NULL AND is_archived = 0`
);

export async function initEmbedder(): Promise<void> {
  const start = Date.now();
  log.info({ msg: "loading_embedding_model", model: EMBEDDING_MODEL });
  embedder = await pipeline("feature-extraction", EMBEDDING_MODEL, {
    dtype: "fp32",
  }) as FeatureExtractionPipeline;
  log.info({ msg: "embedding_model_loaded", model: EMBEDDING_MODEL, ms: Date.now() - start });
}

export async function embed(text: string): Promise<Float32Array> {
  if (!embedder) throw new Error("Embedding model not loaded");
  const result = await embedder(text, { pooling: "mean", normalize: true });
  return new Float32Array(result.data as Float32Array);
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ============================================================================
// IN-MEMORY EMBEDDING CACHE — eliminates cold-start DB reads on every search
// At 800 memories * 1.5KB/embedding = ~1.2MB. Trivial vs the 5s DB cold-read.
// ============================================================================
interface CachedMem {
  id: number; user_id: number; content: string; category: string; importance: number;
  embedding: Float32Array; is_static: boolean; source_count: number;
  is_latest?: boolean; is_forgotten?: boolean;
}
let embeddingCache: CachedMem[] = [];
export let embeddingCacheLatest: CachedMem[] = [];
let embeddingCacheVersion = 0;
let graphCache: { key: string; data: any; ts: number } | null = null;

export function refreshEmbeddingCache(): void {
  const t0 = Date.now();
  const allRows = getAllEmbeddings.all() as Array<any>;
  embeddingCache = [];
  embeddingCacheLatest = [];
  for (const row of allRows) {
    if (!row.embedding) continue;
    const mem: CachedMem = {
      id: row.id, user_id: row.user_id, content: row.content, category: row.category,
      importance: row.importance, embedding: bufferToEmbedding(row.embedding),
      is_static: !!row.is_static, source_count: row.source_count || 1,
      is_latest: !!row.is_latest, is_forgotten: !!row.is_forgotten,
    };
    embeddingCache.push(mem);
    if (row.is_latest && !row.is_forgotten) embeddingCacheLatest.push(mem);
  }
  embeddingCacheVersion++;
  log.info({ msg: "embedding_cache_refreshed", total: embeddingCache.length, latest: embeddingCacheLatest.length, ms: Date.now() - t0 });
}

export function getCachedEmbeddings(latestOnly: boolean): CachedMem[] {
  return latestOnly ? embeddingCacheLatest : embeddingCache;
}

export function addToEmbeddingCache(mem: CachedMem): void {
  embeddingCache.push(mem);
  if (mem.is_latest && !mem.is_forgotten) embeddingCacheLatest.push(mem);
}

export function invalidateEmbeddingCache(): void {
  // Full refresh from DB — called after bulk operations
  refreshEmbeddingCache();
}


export function embeddingToBuffer(emb: Float32Array): Buffer {
  return Buffer.from(emb.buffer, emb.byteOffset, emb.byteLength);
}

export function bufferToEmbedding(buf: Buffer | Uint8Array | ArrayBuffer): Float32Array {
  if (buf instanceof ArrayBuffer) return new Float32Array(buf);
  if (buf instanceof Uint8Array) return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

export function embeddingToVectorJSON(emb: Float32Array): string {
  return "[" + Array.from(emb).join(",") + "]";
}
