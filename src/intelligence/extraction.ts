// ============================================================================
// FAST FACT EXTRACTION — Regex-based, no LLM needed
// ============================================================================

import { db } from "../db/index.ts";
import { log } from "../config/logger.ts";

export function fastExtractFacts(content: string, memoryId: number, userId: number): void {
  try {
    const insertSF = db.prepare(
      `INSERT INTO structured_facts (memory_id, subject, verb, object, quantity, unit, date_ref, date_approx, user_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
    );

    // Extract date from content if present
    const dateMatch = content.match(/\[Conversation date:\s*([\d/]+)\]/);
    const dateApprox = dateMatch ? dateMatch[1].replace(/\//g, "-") : null;

    // Extract relative time references
    const relTimeMatch = content.match(/\b(yesterday|today|last (?:week|month|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|two (?:days|weeks|months) ago|a (?:week|month) ago|(?:this|last) (?:morning|afternoon|evening))\b/i);
    const dateRef = relTimeMatch ? relTimeMatch[1] : null;

    let factCount = 0;

    // Pattern 1: bought/purchased N items
    const buyPattern = /\b(bought|purchased|got|acquired|received|ordered|picked up)\s+(\d+)\s+(.+?)(?:\.|,|$)/gi;
    for (const m of content.matchAll(buyPattern)) {
      try { insertSF.run(memoryId, "user", m[1].toLowerCase(), m[3].trim().substring(0, 200), parseInt(m[2]), null, dateRef, dateApprox, userId); factCount++; } catch {}
    }

    // Pattern 2: spent $N on X
    const spentPattern = /\bspent\s+\$([\d,.]+)\s+(?:on|for)\s+(.+?)(?:\.|,|$)/gi;
    for (const m of content.matchAll(spentPattern)) {
      try { insertSF.run(memoryId, "user", "spent", m[2].trim().substring(0, 200), parseFloat(m[1].replace(",", "")), "dollars", dateRef, dateApprox, userId); factCount++; } catch {}
    }

    // Pattern 3: have/own N X
    const havePattern = /\b(?:I\s+)?(?:have|has|own|got)\s+(\d+)\s+(.+?)(?:\.|,|\s+(?:and|but|so|now))/gi;
    for (const m of content.matchAll(havePattern)) {
      try { insertSF.run(memoryId, "user", "has", m[2].trim().substring(0, 200), parseInt(m[1]), null, dateRef, dateApprox, userId); factCount++; } catch {}
    }

    // Pattern 4: N hours/minutes of X
    const durationPattern = /\b(\d+(?:\.\d+)?)\s+(hours?|minutes?|mins?|days?|weeks?|months?)\s+(?:of\s+)?(.+?)(?:\.|,|$)/gi;
    for (const m of content.matchAll(durationPattern)) {
      try { insertSF.run(memoryId, "user", "did", m[3].trim().substring(0, 200), parseFloat(m[1]), m[2].toLowerCase(), dateRef, dateApprox, userId); factCount++; } catch {}
    }

    // Pattern 5: exercised for N time
    const exercisePattern = /\b(ran|jogged|walked|hiked|swam|cycled|biked|exercised)\s+(?:for\s+)?(\d+(?:\.\d+)?)\s+(hours?|minutes?|mins?|miles?|km)/gi;
    for (const m of content.matchAll(exercisePattern)) {
      try { insertSF.run(memoryId, "user", m[1].toLowerCase(), null, parseFloat(m[2]), m[3].toLowerCase(), dateRef, dateApprox, userId); factCount++; } catch {}
    }

    // Pattern 6: made/baked/cooked X
    const madePattern = /\b(made|baked|cooked|prepared)\s+(?:a\s+|some\s+)?(.+?)(?:\.|,|\s+(?:and|but|for|from|yesterday|today|last))/gi;
    for (const m of content.matchAll(madePattern)) {
      try { insertSF.run(memoryId, "user", m[1].toLowerCase(), m[2].trim().substring(0, 200), 1, null, dateRef, dateApprox, userId); factCount++; } catch {}
    }

    // Pattern 7: earned/made $N
    const earnedPattern = /\b(?:earned|made|received|got paid)\s+\$([\d,.]+)(?:\s+(?:from|for|by)\s+(.+?))?(?:\.|,|$)/gi;
    for (const m of content.matchAll(earnedPattern)) {
      try { insertSF.run(memoryId, "user", "earned", m[2] ? m[2].trim().substring(0, 200) : null, parseFloat(m[1].replace(",", "")), "dollars", dateRef, dateApprox, userId); factCount++; } catch {}
    }

    // Pattern 8: Assistant/AI actions
    const assistantPattern = /\b(?:the\s+)?(?:assistant|AI|Claude|Engram|I)\s+(recommended|suggested|implemented|created|fixed|diagnosed|configured|deployed|built|wrote|designed|refactored|migrated|upgraded|replaced|analyzed|discovered|resolved|identified)\s+(.+?)(?:\.|,|$)/gi;
    for (const m of content.matchAll(assistantPattern)) {
      try { insertSF.run(memoryId, "assistant", m[1].toLowerCase(), m[2].trim().substring(0, 200), null, null, dateRef, dateApprox, userId); factCount++; } catch {}
    }

    // --- User Preferences ---
    const upsertPref = db.prepare(
      `INSERT INTO user_preferences (domain, preference, evidence_memory_id, user_id)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(domain, preference, user_id) DO UPDATE SET
         strength = strength + 0.5,
         evidence_memory_id = excluded.evidence_memory_id,
         updated_at = datetime('now')`
    );

    let prefCount = 0;

    // Likes/enjoys
    const likePattern = /\b(?:I\s+)?(love|like|enjoy|prefer|adore|am (?:really )?into)\s+(.+?)(?:\.|,|!|\s+(?:and|but|so|because))/gi;
    for (const m of content.matchAll(likePattern)) {
      const obj = m[2].trim().substring(0, 200);
      if (obj.length > 3 && obj.length < 100) {
        const domain = inferDomain(obj);
        try { upsertPref.run(domain, `likes ${obj}`, memoryId, userId); prefCount++; } catch {}
      }
    }

    // Dislikes
    const dislikePattern = /\b(?:I\s+)?(hate|dislike|don't like|can't stand|avoid)\s+(.+?)(?:\.|,|!|\s+(?:and|but|so|because))/gi;
    for (const m of content.matchAll(dislikePattern)) {
      const obj = m[2].trim().substring(0, 200);
      if (obj.length > 3 && obj.length < 100) {
        const domain = inferDomain(obj);
        try { upsertPref.run(domain, `dislikes ${obj}`, memoryId, userId); prefCount++; } catch {}
      }
    }

    // Favorites
    const favPattern = /\bmy favorite\s+(.+?)\s+(?:is|are)\s+(.+?)(?:\.|,|$)/gi;
    for (const m of content.matchAll(favPattern)) {
      try { upsertPref.run(m[1].trim().toLowerCase(), `favorite: ${m[2].trim()}`, memoryId, userId); prefCount++; } catch {}
    }

    // --- Current State Updates ---
    const upsertState = db.prepare(
      `INSERT INTO current_state (key, value, memory_id, user_id)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(key, user_id) DO UPDATE SET
         previous_value = current_state.value,
         previous_memory_id = current_state.memory_id,
         value = excluded.value,
         memory_id = excluded.memory_id,
         updated_count = updated_count + 1,
         updated_at = datetime('now')`
    );

    let stateCount = 0;

    // Job/role changes
    const rolePattern = /\b(?:I\s+)?(?:started|began|got promoted to|now work as|am now|just became|accepted a position as)\s+(?:a\s+|an\s+|my\s+)?(.+?)(?:\.|,|$)/gi;
    for (const m of content.matchAll(rolePattern)) {
      const val = m[1].trim().substring(0, 200);
      if (val.length > 3 && val.length < 100) {
        try { upsertState.run("current_role", val, memoryId, userId); stateCount++; } catch {}
      }
    }

    // Location changes
    const locPattern = /\b(?:I\s+)?(?:moved to|relocated to|live in|living in|staying in)\s+(.+?)(?:\.|,|$)/gi;
    for (const m of content.matchAll(locPattern)) {
      try { upsertState.run("current_location", m[1].trim().substring(0, 200), memoryId, userId); stateCount++; } catch {}
    }

    if (factCount + prefCount + stateCount > 0) {
      log.debug({ msg: "fast_extract", id: memoryId, facts: factCount, prefs: prefCount, state: stateCount });
    }
  } catch (e: any) {
    log.debug({ msg: "fast_extract_error", error: e.message });
  }
}

function inferDomain(text: string): string {
  const t = text.toLowerCase();
  if (/food|cook|bak|eat|restaurant|meal|recipe|cuisine/i.test(t)) return "food";
  if (/movie|film|show|series|watch|tv|netflix|stream/i.test(t)) return "entertainment";
  if (/book|read|novel|author|library/i.test(t)) return "reading";
  if (/music|song|album|artist|band|listen|concert/i.test(t)) return "music";
  if (/travel|trip|visit|vacation|hotel|flight|city|country/i.test(t)) return "travel";
  if (/sport|run|jog|gym|exercise|hike|swim|yoga|fitness/i.test(t)) return "fitness";
  if (/game|play|gaming|console/i.test(t)) return "gaming";
  if (/tech|code|program|software|app|computer/i.test(t)) return "technology";
  if (/garden|plant|flower|herb/i.test(t)) return "gardening";
  if (/pet|dog|cat|fish|animal|tank|aquarium/i.test(t)) return "pets";
  if (/art|paint|draw|craft|photo/i.test(t)) return "art";
  if (/shop|buy|store|brand|wear|cloth/i.test(t)) return "shopping";
  return "general";
}


function embeddingToBuffer(emb: Float32Array): Buffer {
  return Buffer.from(emb.buffer, emb.byteOffset, emb.byteLength);
}

function bufferToEmbedding(buf: Buffer | Uint8Array | ArrayBuffer): Float32Array {
  if (buf instanceof ArrayBuffer) return new Float32Array(buf);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return new Float32Array(ab);
}

/** Convert Float32Array to JSON string for libsql vector() function */
function embeddingToVectorJSON(emb: Float32Array): string {
  return "[" + Array.from(emb).join(",") + "]";
}

