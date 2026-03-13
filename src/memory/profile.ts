// ============================================================================
// MEMORY PROFILE — Dynamic user profile generation
// ============================================================================

import { db } from "../db/index.ts";

interface UserProfile {
  static_facts: Array<{ id: number; content: string; category: string; source_count: number }>;
  recent_activity: Array<{ id: number; content: string; category: string; created_at: string }>;
  summary?: string;
}

export async function generateProfile(userId: number = 1, generateSummary: boolean = false): Promise<UserProfile> {
  const staticFacts = getStaticMemories.all(userId) as Array<{
    id: number; content: string; category: string; source_count: number; created_at: string; updated_at: string;
  }>;

  const recentDynamic = getRecentDynamicMemories.all(userId, 20) as Array<{
    id: number; content: string; category: string; source_count: number; created_at: string;
  }>;

  const profile: UserProfile = {
    static_facts: staticFacts.map(f => ({
      id: f.id,
      content: f.content,
      category: f.category,
      source_count: f.source_count,
    })),
    recent_activity: recentDynamic.map(f => ({
      id: f.id,
      content: f.content,
      category: f.category,
      created_at: f.created_at,
    })),
  };

  if (generateSummary && LLM_API_KEY) {
    try {
      const staticText = staticFacts.map(f => `- [${f.category}] ${f.content}`).join("\n");
      const dynamicText = recentDynamic.slice(0, 10).map(f => `- [${f.category}] ${f.content}`).join("\n");

      const summary = await callLLM(
        "You are a profile summarizer. Given a user's permanent facts and recent activity, write a concise 2-4 sentence profile summary. Be factual and direct.",
        `PERMANENT FACTS:\n${staticText || "None"}\n\nRECENT ACTIVITY:\n${dynamicText || "None"}`
      );
      profile.summary = summary.trim();
    } catch (e: any) {
      log.error({ msg: "profile_summary_failed", error: e.message });
    }
  }

  return profile;
}

// ============================================================================
// BACKFILL
// ============================================================================

async function backfillEmbeddings(batchSize: number = 50): Promise<number> {
  const missing = getNoEmbedding.all(batchSize) as Array<{ id: number; content: string }>;
  let count = 0;

  for (const mem of missing) {
    try {
      const emb = await embed(mem.content);
      updateMemoryEmbedding.run(embeddingToBuffer(emb), mem.id); try { updateMemoryVec.run(embeddingToVectorJSON(emb), mem.id); } catch {}
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
