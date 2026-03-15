// ============================================================================
// CAUSAL CHAINS — Temporal cause-effect detection
// Nobody else in the AI memory space does this.
//
// When you store "deployed X" → "X crashed" → "fixed X with Y",
// this system auto-detects the temporal causality and builds chains.
// Query: "what happened last time I deployed X?" → full causal story.
//
// Mem0 has entity graphs. Letta has version history. Neither tracks causality.
// ============================================================================

import { db, insertCausalChain, insertCausalLink, getCausalChainForMemory, getCausalChainMemories } from "../db/index.ts";
import { log } from "../config/logger.ts";
import { embed, cosineSimilarity, bufferToEmbedding } from "../embeddings/index.ts";

// Causal signal patterns — verbs/phrases that indicate cause-effect
const CAUSAL_TRIGGERS = [
  /\b(because|caused by|due to|as a result of|triggered by|led to|resulted in)\b/i,
  /\b(after|then|next|consequently|therefore|so|thus)\b/i,
  /\b(deployed|crashed|fixed|broke|resolved|reverted|updated|migrated|restarted)\b/i,
  /\b(failed|succeeded|completed|started|stopped|killed|recovered)\b/i,
];

// Temporal proximity threshold (hours) — events within this window may be causal
const TEMPORAL_WINDOW_HOURS = 24;

interface CausalCandidate {
  memory_id: number;
  content: string;
  category: string;
  created_at: string;
  similarity: number;
}

/**
 * detectCausalLinks — Called after a new memory is stored.
 * Looks for recent memories that form a causal sequence with this one.
 * 
 * Algorithm:
 * 1. Check if new memory has causal language
 * 2. Find semantically similar recent memories (last 24h)
 * 3. If temporal + semantic + causal signal align → create/extend chain
 */
export async function detectCausalLinks(
  memoryId: number,
  content: string,
  category: string,
  embedding: Float32Array,
  userId: number = 1
): Promise<{ chainId: number; position: number } | null> {
  // Check for causal signal
  const hasCausalSignal = CAUSAL_TRIGGERS.some(pat => pat.test(content));
  if (!hasCausalSignal) return null;

  // Find recent memories (last 24h) with semantic similarity
  const cutoff = new Date(Date.now() - TEMPORAL_WINDOW_HOURS * 3600000).toISOString();
  const recentMemories = db.prepare(
    `SELECT id, content, category, embedding, created_at
     FROM memories
     WHERE id != ? AND user_id = ? AND created_at > ?
       AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1
       AND embedding IS NOT NULL
     ORDER BY created_at DESC LIMIT 50`
  ).all(memoryId, userId, cutoff) as any[];

  // Score candidates by semantic similarity
  const candidates: CausalCandidate[] = [];
  for (const mem of recentMemories) {
    const memEmb = bufferToEmbedding(mem.embedding);
    const sim = cosineSimilarity(embedding, memEmb);
    if (sim > 0.4) { // lower threshold than auto-link — causal connections can be less semantically similar
      candidates.push({
        memory_id: mem.id,
        content: mem.content,
        category: mem.category,
        created_at: mem.created_at,
        similarity: sim,
      });
    }
  }

  if (candidates.length === 0) return null;

  // Sort by time (oldest first — we want the cause before the effect)
  candidates.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());

  // Check if any candidate is already in a chain
  for (const cand of candidates) {
    const existingChain = getCausalChainForMemory.get(cand.memory_id) as any;
    if (existingChain) {
      // Extend existing chain
      const links = (existingChain.links || "").split("|");
      const maxPos = links.reduce((max: number, l: string) => {
        const pos = parseInt(l.split(":")[1] || "0");
        return Math.max(max, pos);
      }, 0);

      insertCausalLink.run(existingChain.id, memoryId, maxPos + 1, classifyRole(content));
      log.info({ msg: "causal_chain_extended", chain_id: existingChain.id, memory_id: memoryId, position: maxPos + 1 });
      return { chainId: existingChain.id, position: maxPos + 1 };
    }
  }

  // Create new chain with the best candidate + this memory
  const bestCandidate = candidates[0]; // oldest similar memory = most likely cause
  const { id: chainId } = insertCausalChain.get(inferChainName(bestCandidate.content, content), userId) as { id: number };

  insertCausalLink.run(chainId, bestCandidate.memory_id, 0, classifyRole(bestCandidate.content));
  insertCausalLink.run(chainId, memoryId, 1, classifyRole(content));

  log.info({ msg: "causal_chain_created", chain_id: chainId, memories: [bestCandidate.memory_id, memoryId] });
  return { chainId, position: 1 };
}

/**
 * getCausalHistory — Given a query, find relevant causal chains.
 * Returns full causal stories, not just individual facts.
 */
export async function getCausalHistory(
  query: string,
  userId: number = 1,
  limit: number = 5
): Promise<Array<{ chain_id: number; name: string; events: any[] }>> {
  const queryEmb = await embed(query);

  // Find memories matching query
  const matches = db.prepare(
    `SELECT id, embedding FROM memories
     WHERE user_id = ? AND embedding IS NOT NULL AND is_forgotten = 0
     ORDER BY created_at DESC LIMIT 100`
  ).all(userId) as any[];

  const scored = matches
    .map(m => ({ id: m.id, sim: cosineSimilarity(queryEmb, bufferToEmbedding(m.embedding)) }))
    .filter(m => m.sim > 0.3)
    .sort((a, b) => b.sim - a.sim)
    .slice(0, 20);

  // Find chains containing these memories
  const chainIds = new Set<number>();
  const chains: Array<{ chain_id: number; name: string; events: any[] }> = [];

  for (const match of scored) {
    const chain = getCausalChainForMemory.get(match.id) as any;
    if (chain && !chainIds.has(chain.id)) {
      chainIds.add(chain.id);
      const events = getCausalChainMemories.all(chain.id) as any[];
      chains.push({ chain_id: chain.id, name: chain.name, events });
      if (chains.length >= limit) break;
    }
  }

  return chains;
}

function classifyRole(content: string): string {
  if (/\b(deployed|started|created|launched|began|initiated)\b/i.test(content)) return "trigger";
  if (/\b(crashed|failed|broke|error|bug|issue)\b/i.test(content)) return "failure";
  if (/\b(fixed|resolved|patched|recovered|restored)\b/i.test(content)) return "resolution";
  if (/\b(caused|triggered|led to|resulted in)\b/i.test(content)) return "cause";
  return "event";
}

function inferChainName(cause: string, effect: string): string {
  // Extract key noun/verb from cause
  const words = cause.split(/\s+/).filter(w => w.length > 3).slice(0, 3).join(" ");
  return words || "unnamed chain";
}
