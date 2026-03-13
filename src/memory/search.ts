// ============================================================================
// MEMORY SEARCH — Hybrid semantic + full-text
// ============================================================================

import { log } from "../config/logger.ts";
import { RERANKER_ENABLED, RERANKER_TOP_K, AUTO_LINK_THRESHOLD, AUTO_LINK_MAX } from "../config/index.ts";
import { db, searchMemoriesFTS, getMemoryWithoutEmbedding, getVersionChain, getLinksFor, insertLink } from "../db/index.ts";
import { embed, cosineSimilarity, getCachedEmbeddings, embeddingToBuffer, bufferToEmbedding } from "../embeddings/index.ts";
import { calculateDecayScore } from "../fsrs/index.ts";
import { sanitizeFTS } from "../helpers/index.ts";

interface SearchResult {
  id: number;
  content: string;
  category: string;
  source?: string;
  importance: number;
  created_at: string;
  score: number;
  decay_score?: number;
  version?: number;
  is_latest?: boolean;
  is_static?: boolean;
  source_count?: number;
  root_memory_id?: number;
  tags?: string[];
  access_count?: number;
  episode_id?: number;
  combined_score?: number;
  semantic_score?: number;
  linked?: Array<{ id: number; content: string; category: string; similarity: number; type: string }>;
  version_chain?: Array<{ id: number; content: string; version: number; is_latest: boolean }>;
}

export async function hybridSearch(
  query: string,
  limit: number = 10,
  includeLinks: boolean = false,
  expandRelationships: boolean = false,
  latestOnly: boolean = true,
  userId: number = 1
): Promise<SearchResult[]> {
  const results = new Map<number, SearchResult>();

  // 1. Vector search — in-memory cosine similarity (<1ms for 800 memories)
  try {
    const queryEmb = await embed(query);
    const cached = getCachedEmbeddings(latestOnly);
    for (const mem of cached) {
      if (mem.user_id !== userId) continue; // S7 FIX: user isolation
      const sim = cosineSimilarity(queryEmb, mem.embedding);
      if (sim > 0.35) {
        results.set(mem.id, {
          id: mem.id, content: mem.content, category: mem.category,
          importance: mem.importance, created_at: "",
          score: sim * 0.55, is_static: !!mem.is_static,
          source_count: mem.source_count || 1,
        });
      }
    }
  } catch (e: any) {
    log.error({ msg: "vector_search_failed", error: e.message });
  }

  // 2. FTS5 keyword search
  const sanitized = sanitizeFTS(query);
  if (sanitized) {
    try {
      const ftsResults = searchMemoriesFTS.all(sanitized, userId, limit * 3) as Array<{
        id: number; content: string; category: string; source: string;
        session_id: string; importance: number; created_at: string; fts_rank: number;
        version: number; is_latest: boolean; parent_memory_id: number; root_memory_id: number;
        source_count: number; is_static: boolean; is_forgotten: boolean; is_inference: boolean;
      }>;

      const maxRank = ftsResults.length > 0 ? Math.abs(ftsResults[0].fts_rank) : 1;

      for (const r of ftsResults) {
        const ftsScore = (Math.abs(r.fts_rank) / maxRank) * 0.35;
        const existing = results.get(r.id);
        if (existing) {
          existing.score += ftsScore;
          existing.created_at = r.created_at;
          existing.source = r.source;
          existing.version = r.version;
          existing.is_latest = !!r.is_latest;
          existing.root_memory_id = r.root_memory_id;
        } else {
          results.set(r.id, {
            id: r.id,
            content: r.content,
            category: r.category,
            source: r.source,
            importance: r.importance,
            created_at: r.created_at,
            score: ftsScore,
            version: r.version,
            is_latest: !!r.is_latest,
            is_static: !!r.is_static,
            source_count: r.source_count || 1,
            root_memory_id: r.root_memory_id,
          });
        }
      }
    } catch {}
  }

  // 3. Boost by importance + source_count + static priority + decay
  for (const r of results.values()) {
    // Use decay score instead of raw importance
    const decayScore = calculateDecayScore(
      r.importance, r.created_at, (r as any).access_count || 0, null,
      !!r.is_static, r.source_count || 1
    );
    r.decay_score = Math.round(decayScore * 1000) / 1000;
    r.score += (decayScore / 10) * 0.05;
    r.score += Math.min((r.source_count || 1) / 10, 1) * 0.03;
    if (r.is_static) r.score += 0.02;
  }

  // 4. Relationship expansion
  if (expandRelationships) {
    const topIds = Array.from(results.entries())
      .sort((a, b) => b[1].score - a[1].score)
      .slice(0, 5)
      .map(([id]) => id);

    for (const id of topIds) {
      const links = getLinksFor.all(id, id) as Array<{
        id: number; similarity: number; type: string; content: string;
        category: string; importance: number; created_at: string;
        is_latest: boolean; is_forgotten: boolean; version: number; source_count: number;
      }>;

      for (const link of links) {
        if (link.is_forgotten) continue;
        if (!results.has(link.id)) {
          const relBonus = link.type === "updates" ? 0.15 : link.type === "extends" ? 0.12 : 0.08;
          results.set(link.id, {
            id: link.id,
            content: link.content,
            category: link.category,
            importance: link.importance,
            created_at: link.created_at,
            score: relBonus * link.similarity,
            version: link.version,
            is_latest: !!link.is_latest,
            source_count: link.source_count || 1,
          });
        }
      }
    }
  }

  // 5. Guard against NaN scores, sort, and limit
  for (const r of results.values()) {
    if (isNaN(r.score)) r.score = 0;
    if (r.decay_score != null && isNaN(r.decay_score)) r.decay_score = 0;
  }
  let sorted = Array.from(results.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  // 6. Fill missing fields
  for (const r of sorted) {
    if (!r.created_at || !r.version) {
      const mem = getMemoryWithoutEmbedding.get(r.id) as any;
      if (mem) {
        r.created_at = r.created_at || mem.created_at;
        r.source = r.source || mem.source;
        r.version = mem.version;
        r.is_latest = !!mem.is_latest;
        r.is_static = !!mem.is_static;
        r.source_count = mem.source_count;
        r.root_memory_id = mem.root_memory_id;
      }
    }
  }

  // 7. Include linked memories + version chain
  if (includeLinks) {
    for (const r of sorted) {
      const links = getLinksFor.all(r.id, r.id) as Array<{
        id: number; similarity: number; type: string; content: string; category: string;
        importance: number; created_at: string; is_latest: boolean; is_forgotten: boolean;
        version: number; source_count: number;
      }>;
      if (links.length > 0) {
        r.linked = links
          .filter(l => !l.is_forgotten)
          .map(l => ({
            id: l.id,
            content: l.content,
            category: l.category,
            similarity: Math.round(l.similarity * 1000) / 1000,
            type: l.type,
          }));
      }

      const rootId = r.root_memory_id || r.id;
      const chain = getVersionChain.all(rootId, rootId) as Array<{
        id: number; content: string; category: string; version: number; is_latest: boolean;
        created_at: string; source_count: number;
      }>;
      if (chain.length > 1) {
        r.version_chain = chain.map(c => ({
          id: c.id,
          content: c.content,
          version: c.version,
          is_latest: !!c.is_latest,
        }));
      }
    }
  }

  return sorted;
}

// ============================================================================
// AUTO-LINKING
// ============================================================================

export async function autoLink(memoryId: number, embedding: Float32Array): Promise<number> {
  // In-memory cosine scan — <1ms for 800 memories
  const similarities: Array<{ id: number; similarity: number }> = [];
  const cached = getCachedEmbeddings(true);
  for (const mem of cached) {
    if (mem.id === memoryId) continue;
    const sim = cosineSimilarity(embedding, mem.embedding);
    if (sim >= AUTO_LINK_THRESHOLD) similarities.push({ id: mem.id, similarity: sim });
  }

  similarities.sort((a, b) => b.similarity - a.similarity);
  const toLink = similarities.slice(0, AUTO_LINK_MAX);

  let linked = 0;
  for (const { id, similarity } of toLink) {
    insertLink.run(memoryId, id, similarity, "similarity");
    insertLink.run(id, memoryId, similarity, "similarity");
    linked++;
  }

  return linked;
}

// ============================================================================
