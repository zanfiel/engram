// ============================================================================
// CONSOLIDATION — Auto-summarize memory clusters
// ============================================================================

import { db, insertMemory, markArchived, insertLink, writeVec, getClusterMembers, getClusterCandidates } from "../db/index.ts";
import { log } from "../config/logger.ts";
import { LLM_API_KEY, CONSOLIDATION_THRESHOLD } from "../config/index.ts";
import { callLLM } from "../llm/index.ts";
import { embed, embeddingToBuffer } from "../embeddings/index.ts";
import { autoLink } from "../memory/search.ts";

// ============================================================================
// MEMORY CONSOLIDATION — Auto-summarize large clusters
// ============================================================================

const CONSOLIDATION_PROMPT = `You are a memory consolidation engine. Given a cluster of related memories, create a single concise summary that captures all key information.

Rules:
- Preserve ALL important facts, decisions, and specific values (versions, IPs, dates, names)
- The summary should be self-contained — someone reading only the summary should understand the full picture
- Keep it under 500 words
- Format as clear, dense paragraphs — not bullet points
- Include the most important specifics, not just generalizations

Respond with ONLY a JSON object:
{
  "summary": "the consolidated summary text",
  "title": "short 3-5 word cluster label",
  "importance": 1-10
}`;

export async function consolidateCluster(
  centerMemoryId: number,
  userId: number = 1
): Promise<{ summaryId: number; archivedCount: number } | null> {
  if (!LLM_API_KEY) return null;

  const members = getClusterMembers.all(centerMemoryId, centerMemoryId) as Array<any>;
  if (members.length < CONSOLIDATION_THRESHOLD) return null;

  // Check if already consolidated
  const existing = db.prepare(
    `SELECT id FROM consolidations WHERE source_memory_ids LIKE ?`
  ).get(`%${centerMemoryId}%`) as any;
  if (existing) return null;

  const memberContents = members.map(m =>
    `[#${m.id}, ${m.category}, imp=${m.importance}]: ${m.content}`
  ).join("\n\n");

  try {
    const response = await callLLM(CONSOLIDATION_PROMPT, memberContents);
    let jsonStr = response.trim();
    if (jsonStr.startsWith("```")) {
      jsonStr = jsonStr.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
    }
    const result = JSON.parse(jsonStr) as { summary: string; title: string; importance: number };

    // Create summary memory
    const embArray = await embed(result.summary);
    const embBuffer = embeddingToBuffer(embArray);
    const imp = Math.max(1, Math.min(10, result.importance || 8));

    const summaryMem = insertMemory.get(
      `[Consolidated: ${result.title}] ${result.summary}`,
      "discovery", "consolidation", null, imp, embBuffer,
      1, 1, null, null, members.length, 1, 0, null, null, 0
    ) as { id: number; created_at: string };
    db.prepare("UPDATE memories SET user_id = ?, tags = ? WHERE id = ?").run(
      userId, JSON.stringify(["consolidated", result.title.toLowerCase().replace(/\s+/g, "-")]), summaryMem.id
    );

    // Archive source memories and link to summary
    let archived = 0;
    for (const m of members) {
      markArchived.run(m.id);
      insertLink.run(summaryMem.id, m.id, 1.0, "consolidates");
      archived++;
    }

    // Track consolidation
    db.prepare(
      `INSERT INTO consolidations (summary_memory_id, source_memory_ids, cluster_label)
       VALUES (?, ?, ?)`
    ).run(summaryMem.id, JSON.stringify(members.map(m => m.id)), result.title);

    writeVec(summaryMem.id, embArray);
    await autoLink(summaryMem.id, embArray);
    log.info({ msg: "consolidated", archived, summary_id: summaryMem.id, title: result.title });
    return { summaryId: summaryMem.id, archivedCount: archived };
  } catch (e: any) {
    log.error({ msg: "consolidation_failed", center_id: centerMemoryId, error: e.message });
    return null;
  }
}

export async function runConsolidationSweep(userId: number = 1): Promise<number> {
  const candidates = getClusterCandidates.all(CONSOLIDATION_THRESHOLD) as Array<{ source_id: number; link_count: number }>;
  let totalConsolidated = 0;
  for (const c of candidates) {
    const result = await consolidateCluster(c.source_id, userId);
    if (result) totalConsolidated += result.archivedCount;
  }
  return totalConsolidated;
}
