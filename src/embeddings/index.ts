// ============================================================================
// EMBEDDINGS — Raw onnxruntime-node + hand-written WordPiece tokenizer
// ============================================================================

import * as ort from "onnxruntime-node";
import { resolve } from "path";
import { readFileSync, existsSync, statSync, mkdirSync, createWriteStream, renameSync } from "fs";
import { EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_MAX_SEQ, MODEL_DIR, ONNX_MODEL_FILE, MODEL_URLS } from "../config/index.ts";
import { log } from "../config/logger.ts";
import { db } from "../db/index.ts";

let ortSession: ort.InferenceSession | null = null;
let tokenizer: BertWordPieceTokenizer | null = null;

// Prepared statements for cache refresh
const getAllEmbeddings = db.prepare(
  `SELECT id, user_id, content, category, importance, embedding, is_static, source_count, is_latest, is_forgotten
   FROM memories WHERE embedding IS NOT NULL AND is_archived = 0`
);
const getAllEpisodeEmbeddings = db.prepare(
  `SELECT id, user_id, summary, embedding FROM episodes WHERE embedding IS NOT NULL`
);

// ============================================================================
// MODEL DOWNLOAD / CACHE
// ============================================================================

async function ensureModelFiles(): Promise<void> {
  mkdirSync(MODEL_DIR, { recursive: true });
  const needed = ["tokenizer.json", ONNX_MODEL_FILE];
  for (const file of needed) {
    const dest = resolve(MODEL_DIR, file);
    if (existsSync(dest)) continue;
    const url = MODEL_URLS[file];
    log.info({ msg: "downloading_model_file", file, url });
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`Failed to download ${file}: ${res.status}`);
    const tmp = dest + ".tmp";
    const ws = createWriteStream(tmp);
    // @ts-ignore — Node 22 ReadableStream
    for await (const chunk of res.body) ws.write(Buffer.from(chunk));
    await new Promise<void>((ok, fail) => { ws.end(() => ok()); ws.on("error", fail); });
    renameSync(tmp, dest);
    log.info({ msg: "downloaded_model_file", file, size_mb: Math.round(statSync(dest).size / 1048576) });
  }
}

// ============================================================================
// MINIMAL BERT WORDPIECE TOKENIZER (zero deps)
// ============================================================================

class BertWordPieceTokenizer {
  private vocab: Map<string, number>;
  private unkId: number;
  private clsId: number;
  private sepId: number;
  private padId: number;
  private maxCharsPerWord: number;
  private prefix: string;

  constructor(tokenizerJsonPath: string) {
    const raw = JSON.parse(readFileSync(tokenizerJsonPath, "utf-8"));
    const model = raw.model;
    this.vocab = new Map(Object.entries(model.vocab) as [string, number][]);
    this.unkId = this.vocab.get("[UNK]") ?? 100;
    this.clsId = this.vocab.get("[CLS]") ?? 101;
    this.sepId = this.vocab.get("[SEP]") ?? 102;
    this.padId = this.vocab.get("[PAD]") ?? 0;
    this.maxCharsPerWord = model.max_input_chars_per_word ?? 100;
    this.prefix = model.continuing_subword_prefix ?? "##";
  }

  private normalize(text: string): string {
    let out = "";
    for (const ch of text) {
      const cp = ch.codePointAt(0)!;
      if (cp === 0 || cp === 0xFFFD || isControl(cp)) continue;
      if (cp === 0x09 || cp === 0x0A || cp === 0x0D) { out += " "; continue; }
      if (isCJK(cp)) { out += ` ${ch} `; continue; }
      out += ch;
    }
    return out.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  }

  private preTokenize(text: string): string[] {
    const tokens: string[] = [];
    let current = "";
    for (const ch of text) {
      if (/\s/.test(ch)) {
        if (current) { tokens.push(current); current = ""; }
      } else if (isPunct(ch)) {
        if (current) { tokens.push(current); current = ""; }
        tokens.push(ch);
      } else {
        current += ch;
      }
    }
    if (current) tokens.push(current);
    return tokens;
  }

  private wordPiece(word: string): number[] {
    if (word.length > this.maxCharsPerWord) return [this.unkId];
    const ids: number[] = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let matched = false;
      while (start < end) {
        const sub = (start > 0 ? this.prefix : "") + word.slice(start, end);
        const id = this.vocab.get(sub);
        if (id !== undefined) {
          ids.push(id);
          start = end;
          matched = true;
          break;
        }
        end--;
      }
      if (!matched) return [this.unkId];
    }
    return ids;
  }

  encode(text: string, maxLen: number = EMBEDDING_MAX_SEQ): {
    input_ids: BigInt64Array; attention_mask: BigInt64Array; token_type_ids: BigInt64Array;
  } {
    const normalized = this.normalize(text);
    const words = this.preTokenize(normalized);
    const tokenIds: number[] = [];
    for (const w of words) {
      if (tokenIds.length >= maxLen - 2) break;
      const wp = this.wordPiece(w);
      for (const id of wp) {
        if (tokenIds.length >= maxLen - 2) break;
        tokenIds.push(id);
      }
    }
    const seqLen = tokenIds.length + 2;
    const input_ids = new BigInt64Array(maxLen);
    const attention_mask = new BigInt64Array(maxLen);
    const token_type_ids = new BigInt64Array(maxLen);
    input_ids[0] = BigInt(this.clsId);
    attention_mask[0] = 1n;
    for (let i = 0; i < tokenIds.length; i++) {
      input_ids[i + 1] = BigInt(tokenIds[i]);
      attention_mask[i + 1] = 1n;
    }
    input_ids[seqLen - 1] = BigInt(this.sepId);
    attention_mask[seqLen - 1] = 1n;
    return { input_ids, attention_mask, token_type_ids };
  }
}

function isControl(cp: number): boolean {
  return (cp >= 0x00 && cp <= 0x1F && cp !== 0x09 && cp !== 0x0A && cp !== 0x0D) || (cp >= 0x7F && cp <= 0x9F);
}

function isCJK(cp: number): boolean {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
    (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
    (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
    (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

const PUNCT_RE = /[\p{P}\p{S}]/u;
function isPunct(ch: string): boolean {
  const cp = ch.codePointAt(0)!;
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return PUNCT_RE.test(ch);
}

// ============================================================================
// ONNX INFERENCE
// ============================================================================

export async function initEmbedder(): Promise<void> {
  const start = Date.now();
  log.info({ msg: "loading_embedding_model", model: EMBEDDING_MODEL, file: ONNX_MODEL_FILE });
  await ensureModelFiles();
  tokenizer = new BertWordPieceTokenizer(resolve(MODEL_DIR, "tokenizer.json"));
  ortSession = await ort.InferenceSession.create(resolve(MODEL_DIR, ONNX_MODEL_FILE), {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all" as any,
    intraOpNumThreads: 0,
  });
  log.info({ msg: "embedding_model_loaded", model: EMBEDDING_MODEL, dim: EMBEDDING_DIM, ms: Date.now() - start });
}

export async function embed(text: string): Promise<Float32Array> {
  if (!ortSession || !tokenizer) throw new Error("Embedding model not loaded");
  const { input_ids, attention_mask, token_type_ids } = tokenizer.encode(text);
  const feeds = {
    input_ids: new ort.Tensor("int64", input_ids, [1, EMBEDDING_MAX_SEQ]),
    attention_mask: new ort.Tensor("int64", attention_mask, [1, EMBEDDING_MAX_SEQ]),
    token_type_ids: new ort.Tensor("int64", token_type_ids, [1, EMBEDDING_MAX_SEQ]),
  };
  const results = await ortSession.run(feeds);
  const outputName = ortSession.outputNames[0];
  const hidden = results[outputName].data as Float32Array;
  // Mean pool over non-padding tokens
  const pooled = new Float32Array(EMBEDDING_DIM);
  let maskSum = 0;
  for (let i = 0; i < EMBEDDING_MAX_SEQ; i++) {
    if (attention_mask[i] === 0n) continue;
    maskSum++;
    const offset = i * EMBEDDING_DIM;
    for (let d = 0; d < EMBEDDING_DIM; d++) pooled[d] += hidden[offset + d];
  }
  for (let d = 0; d < EMBEDDING_DIM; d++) pooled[d] /= maskSum;
  // L2 normalize
  let norm = 0;
  for (let d = 0; d < EMBEDDING_DIM; d++) norm += pooled[d] * pooled[d];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let d = 0; d < EMBEDDING_DIM; d++) pooled[d] /= norm;
  return pooled;
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ============================================================================
// IN-MEMORY EMBEDDING CACHE
// ============================================================================

interface CachedMem {
  id: number; user_id: number; content: string; category: string; importance: number;
  embedding: Float32Array; is_static: boolean; source_count: number;
  is_latest?: boolean; is_forgotten?: boolean;
}
let embeddingCache: CachedMem[] = [];
export let embeddingCacheLatest: CachedMem[] = [];
let embeddingCacheVersion = 0;
export let graphCache: { key: string; data: any; ts: number } | null = null;
export function setGraphCache(val: typeof graphCache): void { graphCache = val; }

// Episode embedding cache
interface CachedEpisode {
  id: number; user_id: number; summary: string; embedding: Float32Array;
}
export let episodeCache: CachedEpisode[] = [];

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
  // Load episode embeddings
  episodeCache = [];
  try {
    const epRows = getAllEpisodeEmbeddings.all() as Array<any>;
    for (const row of epRows) {
      if (!row.embedding) continue;
      episodeCache.push({
        id: row.id, user_id: row.user_id, summary: row.summary || "",
        embedding: bufferToEmbedding(row.embedding),
      });
    }
  } catch {}

  embeddingCacheVersion++;
  log.info({ msg: "embedding_cache_refreshed", total: embeddingCache.length, latest: embeddingCacheLatest.length, episodes: episodeCache.length, ms: Date.now() - t0 });
}

export function getCachedEmbeddings(latestOnly: boolean): CachedMem[] {
  return latestOnly ? embeddingCacheLatest : embeddingCache;
}

export function addToEmbeddingCache(mem: CachedMem): void {
  embeddingCache.push(mem);
  if (mem.is_latest && !mem.is_forgotten) embeddingCacheLatest.push(mem);
}

export function invalidateEmbeddingCache(): void {
  refreshEmbeddingCache();
}

export function embeddingToBuffer(emb: Float32Array): Buffer {
  return Buffer.from(emb.buffer, emb.byteOffset, emb.byteLength);
}

export function bufferToEmbedding(buf: Buffer | Uint8Array | ArrayBuffer): Float32Array {
  if (buf instanceof ArrayBuffer) return new Float32Array(buf);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

export function embeddingToVectorJSON(emb: Float32Array): string {
  return "[" + Array.from(emb).join(",") + "]";
}
