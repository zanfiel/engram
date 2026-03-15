// ============================================================================
// ENGRAM SERVER — Modular entry point
// Run: node --experimental-strip-types server-split.ts
// ============================================================================

import { createServer, type IncomingMessage, type ServerResponse } from "http";

// Config
import { PORT, HOST, OPEN_ACCESS, CORS_ORIGIN, ALLOWED_IPS, CONSOLIDATION_INTERVAL, FORGET_SWEEP_INTERVAL } from "./src/config/index.ts";
import { log } from "./src/config/logger.ts";

// Database (importing triggers schema creation + migrations)
import { db, updateMemoryEmbedding, writeVec } from "./src/db/index.ts";

// Embeddings
import { initEmbedder, embed, refreshEmbeddingCache, embeddingCacheLatest, embeddingToBuffer, embeddingToVectorJSON } from "./src/embeddings/index.ts";

// GUI (importing triggers HMAC secret init)
import { reloadGuiHtml } from "./src/gui/index.ts";

// Routes
import { fetchHandler, setClientIp, sweepExpiredMemories, backfillEmbeddings, updateDecayScores } from "./src/routes/index.ts";

// Intelligence
import { runConsolidationSweep } from "./src/intelligence/consolidation.ts";

// Platform
import { processScheduledDigests } from "./src/platform/digest.ts";

// ============================================================================
// INITIALIZATION
// ============================================================================

await initEmbedder();

// Pre-warm: load embedding cache + JIT-compile ONNX model
{
  const _warmStart = Date.now();
  refreshEmbeddingCache();
  await embed("warmup");
  log.info({ msg: "warmup_complete", cache_size: embeddingCacheLatest.length, ms: Date.now() - _warmStart });
}

// ============================================================================
// HTTP SERVER
// ============================================================================

async function nodeToWebRequest(nodeReq: IncomingMessage): Promise<Request> {
  const proto = nodeReq.headers["x-forwarded-proto"] || "http";
  const host = nodeReq.headers.host || `${HOST}:${PORT}`;
  const url = new URL(nodeReq.url || "/", `${proto}://${host}`);
  const method = nodeReq.method || "GET";
  const headers = new Headers();
  for (const [key, val] of Object.entries(nodeReq.headers)) {
    if (val) headers.set(key, Array.isArray(val) ? val.join(", ") : val);
  }
  let body: BodyInit | undefined;
  if (method !== "GET" && method !== "HEAD") {
    body = await new Promise<Buffer>((resolve) => {
      const chunks: Buffer[] = [];
      nodeReq.on("data", (c: Buffer) => chunks.push(c));
      nodeReq.on("end", () => resolve(Buffer.concat(chunks)));
    });
  }
  return new Request(url.toString(), { method, headers, body, duplex: "half" } as any);
}

async function writeWebResponse(nodeRes: ServerResponse, webRes: Response) {
  nodeRes.writeHead(webRes.status, Object.fromEntries(webRes.headers.entries()));
  const body = webRes.body;
  if (!body) { nodeRes.end(); return; }
  const reader = body.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    nodeRes.write(value);
  }
  nodeRes.end();
}

const server = createServer(async (nodeReq, nodeRes) => {
  try {
    setClientIp(nodeReq.socket.remoteAddress?.replace(/^::ffff:/, "") || "unknown");
    const webReq = await nodeToWebRequest(nodeReq);
    const webRes = await fetchHandler(webReq);
    await writeWebResponse(nodeRes, webRes);
  } catch (err: any) {
    log.error({ msg: "unhandled_request_error", error: err.message });
    if (!nodeRes.headersSent) {
      nodeRes.writeHead(500, { "Content-Type": "application/json" });
    }
    nodeRes.end(JSON.stringify({ error: "Internal server error" }));
  }
});

server.listen(PORT, HOST, () => {
  log.info({ msg: "node_http_server_listening", host: HOST, port: PORT });
});

// ============================================================================
// WAL CHECKPOINT (every 5 minutes)
// ============================================================================
function walCheckpoint() {
  try {
    const result = db.prepare("PRAGMA wal_checkpoint(PASSIVE)").get() as any;
    if (result && result.checkpointed > 0) log.debug({ msg: "wal_checkpoint", ...result });
  } catch (e: any) {
    log.error({ msg: "wal_checkpoint_failed", error: e.message });
  }
}
setInterval(walCheckpoint, 5 * 60 * 1000);

// ============================================================================
// GRACEFUL SHUTDOWN
// ============================================================================
async function gracefulShutdown(signal: string) {
  log.info({ msg: "shutdown_start", signal });
  try {
    db.prepare("PRAGMA wal_checkpoint(TRUNCATE)").get();
    log.info({ msg: "wal_final_checkpoint" });
  } catch (e: any) {
    log.error({ msg: "wal_checkpoint_failed", error: e.message });
  }
  try { db.close(); } catch {}
  log.info({ msg: "shutdown_complete", signal });
  process.exit(0);
}
process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));
process.on("SIGHUP", () => reloadGuiHtml());

// ============================================================================
// STARTUP TASKS
// ============================================================================

// Backfill unembedded memories
const countNoEmbedding = db.prepare("SELECT COUNT(*) as count FROM memories WHERE embedding IS NULL");
const noEmb = (countNoEmbedding.get() as { count: number }).count;
if (noEmb > 0) {
  log.info({ msg: "backfill_start", count: noEmb });
  backfillEmbeddings(200).then((n) => {
    log.info({ msg: "backfill_done", backfilled: n, remaining: noEmb - n });
  }).catch(e => log.error({ msg: "backfill_error", error: String(e) }));
}

// Auto-forget sweep timer
setInterval(() => {
  const swept = sweepExpiredMemories();
  if (swept > 0) log.info({ msg: "auto_forget_sweep", swept });
}, FORGET_SWEEP_INTERVAL);

// Decay score refresh (every 15 minutes)
setInterval(() => {
  const updated = updateDecayScores();
  if (updated > 0) log.info({ msg: "decay_refresh", updated });
}, 15 * 60 * 1000);

// Auto-consolidation sweep (if LLM configured)
import { isLLMAvailable } from "./src/llm/index.ts";
if (isLLMAvailable()) {
  setInterval(async () => {
    try {
      const consolidated = await runConsolidationSweep();
      if (consolidated > 0) log.info({ msg: "auto_consolidation", consolidated });
    } catch (e: any) {
      log.error({ msg: "auto_consolidation_error", error: e.message });
    }
  }, CONSOLIDATION_INTERVAL);
}

// Initial sweeps
sweepExpiredMemories();
updateDecayScores();

// One-time embedding dimension migration: detect 384-dim → re-embed to 1024-dim
{
  const sample = db.prepare(
    "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1"
  ).get() as { id: number; embedding: ArrayBuffer | Buffer } | undefined;
  if (sample) {
    const buf = sample.embedding instanceof ArrayBuffer ? sample.embedding
      : sample.embedding.buffer.slice(sample.embedding.byteOffset, sample.embedding.byteOffset + sample.embedding.byteLength);
    const existingDim = buf.byteLength / 4;
    if (existingDim !== 1024) {
      log.info({ msg: "embedding_dimension_migration_start", from: existingDim, to: 1024 });
      const allMems = db.prepare(
        "SELECT id, content FROM memories WHERE embedding IS NOT NULL"
      ).all() as Array<{ id: number; content: string }>;
      let migrated = 0, failed = 0;
      for (const mem of allMems) {
        try {
          const emb = await embed(mem.content);
          updateMemoryEmbedding.run(embeddingToBuffer(emb), mem.id);
          writeVec(mem.id, emb);
          migrated++;
          if (migrated % 50 === 0) log.info({ msg: "migration_progress", migrated, total: allMems.length });
        } catch (e: any) {
          log.error({ msg: "migration_embed_failed", id: mem.id, error: e.message });
          failed++;
        }
      }
      refreshEmbeddingCache();
      log.info({ msg: "embedding_dimension_migration_complete", migrated, failed, total: allMems.length });
    }
  }
}

// Digest scheduler — check every 5 minutes for due digests
setInterval(async () => {
  try {
    const sent = await processScheduledDigests();
    if (sent > 0) log.info({ msg: "digest_sent", count: sent });
  } catch (e: any) {
    log.error({ msg: "digest_scheduler_error", error: e.message });
  }
}, 5 * 60 * 1000);

log.info({ msg: "server_started", version: "5.7.1", host: HOST, port: PORT, open_access: OPEN_ACCESS, cors: CORS_ORIGIN, log_level: process.env.ENGRAM_LOG_LEVEL || "info", allowed_ips: ALLOWED_IPS.length || "any" });
