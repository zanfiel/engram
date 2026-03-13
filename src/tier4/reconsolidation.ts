// ============================================================================
// MEMORY RECONSOLIDATION — Periodic re-evaluation of stored memories
// Inspired by neuroscience: the brain doesn't just store memories and forget
// them — it periodically pulls old memories back into active state,
// re-evaluates them against current context, and either strengthens or
// rewrites them. This is called "reconsolidation."
//
// Nobody in the AI memory space does this. Mem0 has update/delete.
// Letta has versioning. Neither proactively re-evaluates old memories
// against what's been learned since they were stored.
// ============================================================================

import { db, insertReconsolidation, getMemoriesForReconsolidation, updateAdaptiveScore } from "../db/index.ts";
import { log } from "../config/logger.ts";
import { embed, cosineSimilarity, bufferToEmbedding } from "../embeddings/index.ts";

interface ReconsolidationResult {
  memory_id: number;
  action: "strengthened" | "weakened" | "corrected" | "unchanged";
  old_importance: number;
  new_importance: number;
  old_confidence: number;
  new_confidence: number;
  reason: string;
}

/**
 * reconsolidateMemory — Re-evaluate a single memory against current knowledge.
 *
 * Checks:
 * 1. Is this memory contradicted by newer, higher-confidence memories?
 * 2. Has this memory been accessed often (useful) or ignored (irrelevant)?
 * 3. Has the user corrected or updated related information?
 * 4. Is the memory's FSRS stability declining (being forgotten)?
 */
export async function reconsolidateMemory(memoryId: number, userId: number = 1): Promise<ReconsolidationResult> {
  const mem = db.prepare("SELECT * FROM memories WHERE id = ?").get(memoryId) as any;
  if (!mem) throw new Error("Memory not found");

  let newImportance = mem.importance;
  let newConfidence = mem.confidence ?? 1.0;
  let reason = "";

  // Check 1: Contradictions — newer memories that contradict this one
  if (mem.embedding) {
    const memEmb = bufferToEmbedding(mem.embedding);
    const newer = db.prepare(
      `SELECT id, content, embedding, confidence, created_at
       FROM memories
       WHERE user_id = ? AND id != ? AND is_forgotten = 0 AND is_latest = 1
         AND created_at > ? AND embedding IS NOT NULL
       ORDER BY created_at DESC LIMIT 50`
    ).all(userId, memoryId, mem.created_at) as any[];

    for (const n of newer) {
      const sim = cosineSimilarity(memEmb, bufferToEmbedding(n.embedding));
      if (sim > 0.6) {
        // High similarity to a newer memory — check if it's a correction
        const isCorrection = db.prepare(
          "SELECT id FROM memory_links WHERE source_id = ? AND target_id = ? AND type IN ('corrects', 'updates', 'contradicts')"
        ).get(n.id, memoryId);

        if (isCorrection) {
          // This memory has been superseded
          newConfidence = Math.max(0.1, newConfidence * 0.5);
          newImportance = Math.max(1, newImportance - 2);
          reason += "Superseded by newer memory #" + n.id + ". ";
        }
      }
    }
  }

  // Check 2: Access patterns — adaptive importance
  const hits = mem.recall_hits || 0;
  const misses = mem.recall_misses || 0;
  const totalRecalls = hits + misses;

  if (totalRecalls > 3) {
    const hitRate = hits / totalRecalls;
    if (hitRate > 0.7) {
      // Memory is frequently useful
      newImportance = Math.min(10, newImportance + 1);
      newConfidence = Math.min(1.0, newConfidence + 0.1);
      reason += "High recall utility (" + Math.round(hitRate * 100) + "% hit rate). ";
    } else if (hitRate < 0.3) {
      // Memory is rarely useful
      newImportance = Math.max(1, newImportance - 1);
      reason += "Low recall utility (" + Math.round(hitRate * 100) + "% hit rate). ";
    }
  }

  // Check 3: FSRS stability — if very low, memory is being forgotten
  if (mem.fsrs_stability != null && mem.fsrs_stability < 0.5) {
    newConfidence = Math.max(0.1, newConfidence * 0.8);
    reason += "Low FSRS stability (" + mem.fsrs_stability.toFixed(2) + "). ";
  }

  // Check 4: Age + static classification — very old dynamic memories decay
  const ageDays = (Date.now() - new Date(mem.created_at + "Z").getTime()) / 86400000;
  if (!mem.is_static && ageDays > 30 && mem.access_count < 3) {
    newImportance = Math.max(1, newImportance - 1);
    reason += "Old dynamic memory with low access. ";
  }

  // Determine action
  let action: ReconsolidationResult["action"] = "unchanged";
  if (newImportance !== mem.importance || Math.abs(newConfidence - (mem.confidence ?? 1.0)) > 0.05) {
    if (newImportance > mem.importance || newConfidence > (mem.confidence ?? 1.0)) {
      action = "strengthened";
    } else if (newConfidence < (mem.confidence ?? 1.0) * 0.6) {
      action = "corrected";
    } else {
      action = "weakened";
    }

    // Apply changes
    db.prepare("UPDATE memories SET importance = ?, confidence = ?, updated_at = datetime('now') WHERE id = ?")
      .run(newImportance, newConfidence, memoryId);

    // Update adaptive score
    const adaptiveScore = totalRecalls > 0 ? hits / totalRecalls : 0.5;
    updateAdaptiveScore.run(adaptiveScore, hits, misses, memoryId);

    // Record reconsolidation
    insertReconsolidation.run(memoryId, mem.importance, newImportance, mem.confidence ?? 1.0, newConfidence, reason.trim());

    log.info({ msg: "reconsolidated", memory_id: memoryId, action, old_imp: mem.importance, new_imp: newImportance });
  }

  return {
    memory_id: memoryId,
    action,
    old_importance: mem.importance,
    new_importance: newImportance,
    old_confidence: mem.confidence ?? 1.0,
    new_confidence: newConfidence,
    reason: reason.trim() || "No changes needed",
  };
}

/**
 * runReconsolidationSweep — Periodically re-evaluate memories that need attention.
 * Called on a timer (e.g., every hour).
 */
export async function runReconsolidationSweep(userId: number = 1, batchSize: number = 20): Promise<ReconsolidationResult[]> {
  const candidates = getMemoriesForReconsolidation.all(userId, batchSize) as any[];
  const results: ReconsolidationResult[] = [];

  for (const mem of candidates) {
    try {
      const result = await reconsolidateMemory(mem.id, userId);
      if (result.action !== "unchanged") {
        results.push(result);
      }
    } catch (e) {
      log.warn({ msg: "reconsolidation_error", memory_id: mem.id, error: String(e) });
    }
  }

  if (results.length > 0) {
    log.info({ msg: "reconsolidation_sweep", processed: candidates.length, changed: results.length });
  }

  return results;
}

/**
 * recordRecallOutcome — Track whether a recalled memory was useful.
 * Called by search/recall endpoints when results are used or discarded.
 */
export function recordRecallOutcome(memoryId: number, useful: boolean): void {
  if (useful) {
    db.prepare("UPDATE memories SET recall_hits = recall_hits + 1 WHERE id = ?").run(memoryId);
  } else {
    db.prepare("UPDATE memories SET recall_misses = recall_misses + 1 WHERE id = ?").run(memoryId);
  }
}
