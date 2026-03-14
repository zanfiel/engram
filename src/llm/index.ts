// ============================================================================
// LLM — Client, fact extraction, reranker
// Supports: Anthropic API key, OpenAI-compatible (Ollama, LiteLLM, vLLM)
// Set via env: LLM_API_KEY, LLM_URL, LLM_MODEL
// ============================================================================

import { LLM_URL, LLM_API_KEY, LLM_MODEL, RERANKER_ENABLED, RERANKER_TOP_K } from "../config/index.ts";
import { log } from "../config/logger.ts";

interface FactExtractionResult {
  facts: Array<{
    content: string;
    category: string;
    is_static: boolean;
    forget_after?: string | null;
    forget_reason?: string | null;
    importance: number;
  }>;
  relation_to_existing: {
    type: "none" | "updates" | "extends" | "duplicate" | "contradicts" | "caused_by" | "prerequisite_for" | "corrects";
    existing_memory_id?: number | null;
    reason?: string;
  };
}

// --- LLM availability check ---

export function isLLMAvailable(): boolean {
  if (LLM_API_KEY) return true;
  if (LLM_URL.includes("127.0.0.1") || LLM_URL.includes("localhost")) return true;
  return false;
}

// --- Main LLM call ---

export async function callLLM(systemPrompt: string, userPrompt: string, model?: string): Promise<string> {
  const useModel = model || LLM_MODEL;
  const isAnthropic = LLM_URL.includes("anthropic.com");

  if (isAnthropic) {
    if (!LLM_API_KEY) throw new Error("LLM_API_KEY required for Anthropic API");
    const resp = await fetch(LLM_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": LLM_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: useModel,
        max_tokens: 2000,
        system: systemPrompt,
        messages: [{ role: "user", content: userPrompt }],
      }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Anthropic request failed (${resp.status}): ${text}`);
    }
    const data = await resp.json() as any;
    return data.content?.[0]?.text || "";
  }

  // OpenAI-compatible format (Ollama, LiteLLM, vLLM, etc.)
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (LLM_API_KEY) headers["Authorization"] = `Bearer ${LLM_API_KEY}`;

  const resp = await fetch(LLM_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: useModel,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      max_tokens: 2000,
      temperature: 0.1,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`LLM request failed (${resp.status}): ${text}`);
  }

  const data = await resp.json() as any;
  return data.choices?.[0]?.message?.content || "";
}

const FACT_EXTRACTION_PROMPT = `You are a fact extraction engine for a persistent memory system. Your job is to analyze new content being stored and compare it with existing memories.

Given the NEW CONTENT and up to 3 SIMILAR EXISTING MEMORIES, you must:
1. Determine if this new content updates, extends, or duplicates any existing memory
2. Classify whether each fact is STATIC (permanent, unlikely to change — like preferences, identity, infrastructure) or DYNAMIC (temporary, likely to change — like current tasks, recent events, moods)
3. For dynamic facts, estimate when they should be forgotten (if applicable)
4. Rate importance 1-10

Respond with ONLY valid JSON (no markdown, no backticks):
{
  "facts": [
    {
      "content": "extracted fact text",
      "category": "task|discovery|decision|state|issue",
      "is_static": true/false,
      "forget_after": "ISO datetime or null",
      "forget_reason": "reason or null",
      "importance": 1-10
    }
  ],
  "tags": ["lowercase", "keyword", "tags"],
  "relation_to_existing": {
    "type": "none|updates|extends|duplicate|contradicts|caused_by|prerequisite_for|corrects",
    "existing_memory_id": number_or_null,
    "reason": "why this relation was determined"
  }
}

Rules:
- "corrects" = explicit correction of existing memory. HIGHEST priority relation.
- "updates" = supersedes with newer info
- "extends" = adds to without contradicting
- "duplicate" = same thing
- "contradicts" = directly conflicts
- "caused_by" / "prerequisite_for" = causal relationships
- "none" = no meaningful relation
- For forget_after: ISO 8601 datetime. Permanent facts = null.
- 1-3 key facts per content
- Include "tags": 2-5 lowercase keywords
- Include "structured_facts", "preferences", "state_updates" if applicable`;

export async function extractFacts(
  content: string,
  category: string,
  similarMemories: Array<{ id: number; content: string; category: string; score: number }>
): Promise<FactExtractionResult | null> {
  try {
    let userPrompt = `NEW CONTENT (category: ${category}):\n${content}\n\n`;
    if (similarMemories.length > 0) {
      userPrompt += "SIMILAR EXISTING MEMORIES:\n";
      for (const m of similarMemories) {
        userPrompt += `[ID: ${m.id}, category: ${m.category}, similarity: ${m.score.toFixed(3)}]\n${m.content}\n\n`;
      }
    } else {
      userPrompt += "SIMILAR EXISTING MEMORIES: none found\n";
    }
    const response = await callLLM(FACT_EXTRACTION_PROMPT, userPrompt);
    let jsonStr = response.trim();
    if (jsonStr.startsWith("```")) {
      jsonStr = jsonStr.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
    }
    return JSON.parse(jsonStr) as FactExtractionResult;
  } catch (e: any) {
    log.error({ msg: "fact_extraction_failed", error: e.message });
    return null;
  }
}

// ============================================================================
// PROCESS FACT EXTRACTION RESULTS
// ============================================================================

import {
  db, getMemoryWithoutEmbedding, insertLink, markSuperseded, updateConfidence,
} from "../db/index.ts";
import { emitWebhookEvent } from "../platform/webhooks.ts";

function propagateConfidence(memoryId: number, relationType: string, existingMemoryId: number): void {
  if (relationType === "updates") {
    // Old memory's confidence drops — it's been superseded
    updateConfidence.run(0.3, existingMemoryId);
  } else if (relationType === "contradicts") {
    // Both memories get reduced confidence — conflict needs resolution
    const existing = getMemoryWithoutEmbedding.get(existingMemoryId) as any;
    const current = getMemoryWithoutEmbedding.get(memoryId) as any;
    if (existing) {
      const newConf = Math.max(0.2, (existing.confidence || 1.0) * 0.6);
      updateConfidence.run(newConf, existingMemoryId);
    }
    if (current) {
      updateConfidence.run(0.7, memoryId); // newer info gets slight benefit of doubt
    }

    emitWebhookEvent("contradiction.detected", {
      memory_id: memoryId,
      contradicts_memory_id: existingMemoryId,
      memory_content: current?.content,
      existing_content: existing?.content,
    });
  } else if (relationType === "extends") {
    // Extended memory gets a small confidence boost — it's been corroborated
    const existing = getMemoryWithoutEmbedding.get(existingMemoryId) as any;
    if (existing) {
      const newConf = Math.min(1.0, (existing.confidence || 1.0) * 1.05);
      updateConfidence.run(newConf, existingMemoryId);
    }
  }
}
const incrementSourceCount = db.prepare("UPDATE memories SET source_count = source_count + 1 WHERE id = ?");

export function processExtractionResult(
  newMemoryId: number,
  result: FactExtractionResult,
  embArray: Float32Array | null
): void {
  const rel = result.relation_to_existing;

  if (rel.type === "duplicate" && rel.existing_memory_id) {
    const existing = getMemoryWithoutEmbedding.get(rel.existing_memory_id) as any;
    if (existing && !existing.is_forgotten) {
      incrementSourceCount.run(rel.existing_memory_id);
      markSuperseded.run(newMemoryId);
      insertLink.run(newMemoryId, rel.existing_memory_id, 1.0, "derives");
      return;
    }
  }

  if (rel.type === "updates" && rel.existing_memory_id) {
    const existing = getMemoryWithoutEmbedding.get(rel.existing_memory_id) as any;
    if (existing) {
      markSuperseded.run(rel.existing_memory_id);
      const rootId = existing.root_memory_id || existing.id;
      const newVersion = (existing.version || 1) + 1;
      db.prepare(`UPDATE memories SET version = ?, root_memory_id = ?, parent_memory_id = ?, is_latest = 1 WHERE id = ?`)
        .run(newVersion, rootId, existing.id, newMemoryId);
      insertLink.run(newMemoryId, rel.existing_memory_id, 1.0, "updates");
      propagateConfidence(newMemoryId, "updates", rel.existing_memory_id);
    }
  }

  if (rel.type === "extends" && rel.existing_memory_id) {
    insertLink.run(newMemoryId, rel.existing_memory_id, 0.9, "extends");
  }

  if (rel.type === "contradicts" && rel.existing_memory_id) {
    insertLink.run(newMemoryId, rel.existing_memory_id, 0.85, "contradicts");
    insertLink.run(rel.existing_memory_id, newMemoryId, 0.85, "contradicts");
  }

  if (rel.type === "caused_by" && rel.existing_memory_id) {
    insertLink.run(newMemoryId, rel.existing_memory_id, 0.8, "caused_by");
  }

  if (rel.type === "prerequisite_for" && rel.existing_memory_id) {
    insertLink.run(rel.existing_memory_id, newMemoryId, 0.8, "prerequisite_for");
  }

  if (rel.type === "corrects" && rel.existing_memory_id) {
    const existing = getMemoryWithoutEmbedding.get(rel.existing_memory_id) as any;
    if (existing) {
      markSuperseded.run(rel.existing_memory_id);
      const rootId = existing.root_memory_id || existing.id;
      const newVersion = (existing.version || 1) + 1;
      db.prepare(`UPDATE memories SET version = ?, root_memory_id = ?, parent_memory_id = ?, is_latest = 1, is_static = 1,
         importance = CASE WHEN importance < 9 THEN 9 ELSE importance END WHERE id = ?`)
        .run(newVersion, rootId, existing.id, newMemoryId);
      insertLink.run(newMemoryId, rel.existing_memory_id, 1.0, "corrects");
    }
  }

  if (result.facts.length > 0) {
    const f = result.facts[0];
    db.prepare(`UPDATE memories SET is_static = ?, forget_after = ?, forget_reason = ?,
       importance = CASE WHEN importance = 5 THEN ? ELSE importance END, updated_at = datetime('now') WHERE id = ?`)
      .run(f.is_static ? 1 : 0, f.forget_after || null, f.forget_reason || null, f.importance, newMemoryId);
  }

  if ((result as any).tags?.length) {
    const inferred = (result as any).tags.map((t: any) => String(t).trim().toLowerCase()).filter(Boolean);
    const mem = getMemoryWithoutEmbedding.get(newMemoryId) as any;
    let existing: string[] = [];
    if (mem?.tags) try { existing = JSON.parse(mem.tags); } catch {}
    const merged = [...new Set([...existing, ...inferred])];
    db.prepare("UPDATE memories SET tags = ? WHERE id = ?").run(JSON.stringify(merged), newMemoryId);
  }

  if ((result as any).structured_facts?.length) {
    const insertSF = db.prepare(
      `INSERT INTO structured_facts (memory_id, subject, verb, object, quantity, unit, date_ref, date_approx, user_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)`
    );
    for (const sf of (result as any).structured_facts) {
      try { insertSF.run(newMemoryId, sf.subject || "user", sf.verb || "unknown", sf.object || null,
        sf.quantity != null ? Number(sf.quantity) : null, sf.unit || null, sf.date_ref || null, sf.date_approx || null); } catch {}
    }
  }

  if ((result as any).preferences?.length) {
    const upsertPref = db.prepare(
      `INSERT INTO user_preferences (domain, preference, evidence_memory_id, user_id) VALUES (?, ?, ?, 1)
       ON CONFLICT(domain, preference, user_id) DO UPDATE SET strength = strength + 0.5, evidence_memory_id = excluded.evidence_memory_id, updated_at = datetime('now')`
    );
    for (const p of (result as any).preferences) {
      try { upsertPref.run(p.domain || "general", p.preference, newMemoryId); } catch {}
    }
  }

  if ((result as any).state_updates?.length) {
    const upsertState = db.prepare(
      `INSERT INTO current_state (key, value, memory_id, user_id) VALUES (?, ?, ?, 1)
       ON CONFLICT(key, user_id) DO UPDATE SET previous_value = current_state.value, previous_memory_id = current_state.memory_id,
         value = excluded.value, memory_id = excluded.memory_id, updated_count = updated_count + 1, updated_at = datetime('now')`
    );
    for (const s of (result as any).state_updates) {
      try { upsertState.run(s.key, s.value, newMemoryId); } catch {}
    }
  }
}

// ============================================================================
// LLM-BASED RERANKER
// ============================================================================

export async function rerank(
  query: string,
  candidates: Array<{ id: number; content: string; score: number; [k: string]: any }>,
  topK: number = RERANKER_TOP_K
): Promise<typeof candidates> {
  if (!RERANKER_ENABLED || !isLLMAvailable() || candidates.length <= 3) return candidates;

  const toRerank = candidates.slice(0, Math.min(topK, candidates.length));
  const numbered = toRerank.map((c, i) => `[${i}] ${c.content.substring(0, 200)}`).join("\n");

  const prompt = `Given the query, rank the following documents by relevance. Return ONLY a JSON array of indices from most to least relevant.

Query: "${query}"

Documents:
${numbered}

Return format: [most_relevant_index, next_most_relevant, ...]`;

  try {
    const resp = await callLLM("You are a document reranking engine. Return only a JSON array of integer indices.", prompt);
    if (!resp) return candidates;
    const match = resp.match(/\[[\d,\s]+\]/);
    if (!match) return candidates;
    const indices = JSON.parse(match[0]) as number[];
    const reranked: typeof candidates = [];
    for (let rank = 0; rank < indices.length; rank++) {
      const idx = indices[rank];
      if (idx >= 0 && idx < toRerank.length) {
        const item = { ...toRerank[idx] };
        item.score = item.score * (1 + (indices.length - rank) / indices.length * 0.5);
        reranked.push(item);
      }
    }
    const rerankedIds = new Set(reranked.map(r => r.id));
    for (const c of candidates) {
      if (!rerankedIds.has(c.id)) reranked.push(c);
    }
    return reranked;
  } catch {
    return candidates;
  }
}
