// ============================================================================
// EMOTIONAL VALENCE — Sentiment/affect tracking
// Nobody in the AI memory space tracks emotional context.
//
// Each memory gets valence (-1 to +1), arousal (0 to 1), and dominant emotion.
// Enables queries like "what frustrates this user?" or "what made them happy?"
// Useful for personalization and understanding user satisfaction patterns.
// ============================================================================

import { updateValence } from "../db/index.ts";
import { log } from "../config/logger.ts";

// Emotion lexicon — fast regex-based detection (no LLM needed)
const EMOTION_PATTERNS: Array<{ pattern: RegExp; emotion: string; valence: number; arousal: number }> = [
  // Negative high arousal
  { pattern: /\b(furious|enraged|livid|outraged)\b/i, emotion: "anger", valence: -0.9, arousal: 0.9 },
  { pattern: /\b(angry|pissed|mad|frustrated|annoyed|irritated)\b/i, emotion: "anger", valence: -0.7, arousal: 0.7 },
  { pattern: /\b(terrified|panicked|horrified)\b/i, emotion: "fear", valence: -0.8, arousal: 0.9 },
  { pattern: /\b(anxious|worried|nervous|stressed|afraid|scared)\b/i, emotion: "fear", valence: -0.6, arousal: 0.6 },

  // Negative low arousal
  { pattern: /\b(devastated|heartbroken|grief|mourning)\b/i, emotion: "sadness", valence: -0.9, arousal: 0.3 },
  { pattern: /\b(sad|disappointed|depressed|miserable|upset|bummed)\b/i, emotion: "sadness", valence: -0.6, arousal: 0.3 },
  { pattern: /\b(bored|tired|exhausted|drained|burned out|burnt out)\b/i, emotion: "fatigue", valence: -0.3, arousal: 0.1 },
  { pattern: /\b(confused|lost|stuck|puzzled|stumped)\b/i, emotion: "confusion", valence: -0.3, arousal: 0.4 },

  // Negative technical (infrastructure context)
  { pattern: /\b(crashed|broken|failed|down|error|bug|issue|problem)\b/i, emotion: "frustration", valence: -0.5, arousal: 0.6 },
  { pattern: /\b(hate|worst|terrible|awful|horrible|garbage|trash)\b/i, emotion: "disgust", valence: -0.8, arousal: 0.5 },

  // Positive high arousal
  { pattern: /\b(ecstatic|thrilled|elated|overjoyed)\b/i, emotion: "joy", valence: 0.9, arousal: 0.9 },
  { pattern: /\b(excited|pumped|stoked|hyped|amazing|incredible)\b/i, emotion: "excitement", valence: 0.8, arousal: 0.8 },
  { pattern: /\b(happy|glad|pleased|delighted|great|awesome|fantastic)\b/i, emotion: "joy", valence: 0.7, arousal: 0.6 },
  { pattern: /\b(proud|accomplished|nailed|crushed it|killed it)\b/i, emotion: "pride", valence: 0.7, arousal: 0.6 },

  // Positive low arousal
  { pattern: /\b(satisfied|content|good|nice|fine|pleasant|comfortable)\b/i, emotion: "satisfaction", valence: 0.4, arousal: 0.3 },
  { pattern: /\b(calm|relaxed|peaceful|serene|chill)\b/i, emotion: "calm", valence: 0.3, arousal: 0.1 },
  { pattern: /\b(grateful|thankful|appreciate)\b/i, emotion: "gratitude", valence: 0.6, arousal: 0.3 },
  { pattern: /\b(curious|interested|intrigued|fascinated)\b/i, emotion: "curiosity", valence: 0.4, arousal: 0.5 },

  // Positive technical
  { pattern: /\b(fixed|resolved|working|deployed|shipped|launched|completed|done|finished)\b/i, emotion: "accomplishment", valence: 0.5, arousal: 0.5 },
  { pattern: /\b(love|perfect|beautiful|elegant|clean|brilliant)\b/i, emotion: "admiration", valence: 0.7, arousal: 0.4 },

  // Neutral
  { pattern: /\b(surprised|unexpected|wow|whoa)\b/i, emotion: "surprise", valence: 0.0, arousal: 0.7 },
];

export interface ValenceResult {
  valence: number;       // -1 (negative) to +1 (positive)
  arousal: number;       // 0 (calm) to 1 (intense)
  dominant_emotion: string;
  all_emotions: Array<{ emotion: string; valence: number; arousal: number }>;
}

/**
 * analyzeValence — Fast regex-based emotion detection.
 * Returns aggregate valence/arousal from all matched patterns.
 */
export function analyzeValence(content: string): ValenceResult {
  const matches: Array<{ emotion: string; valence: number; arousal: number }> = [];

  for (const pat of EMOTION_PATTERNS) {
    if (pat.pattern.test(content)) {
      matches.push({ emotion: pat.emotion, valence: pat.valence, arousal: pat.arousal });
    }
  }

  if (matches.length === 0) {
    return { valence: 0, arousal: 0, dominant_emotion: "neutral", all_emotions: [] };
  }

  // Aggregate: weighted average (stronger emotions weighted more)
  const totalWeight = matches.reduce((sum, m) => sum + Math.abs(m.valence), 0);
  const avgValence = matches.reduce((sum, m) => sum + m.valence * Math.abs(m.valence), 0) / totalWeight;
  const avgArousal = matches.reduce((sum, m) => sum + m.arousal * Math.abs(m.valence), 0) / totalWeight;

  // Dominant = strongest absolute valence
  const dominant = matches.reduce((best, m) =>
    Math.abs(m.valence) > Math.abs(best.valence) ? m : best
  );

  return {
    valence: Math.round(avgValence * 100) / 100,
    arousal: Math.round(avgArousal * 100) / 100,
    dominant_emotion: dominant.emotion,
    all_emotions: matches,
  };
}

/**
 * storeValence — Analyze and persist valence for a memory.
 */
export function storeValence(memoryId: number, content: string): ValenceResult {
  const result = analyzeValence(content);
  if (result.dominant_emotion !== "neutral") {
    updateValence.run(result.valence, result.arousal, result.dominant_emotion, memoryId);
  }
  return result;
}

/**
 * queryByEmotion — Find memories by emotional state.
 */
export function queryByEmotion(
  emotion: string,
  userId: number = 1,
  limit: number = 20
): any[] {
  return db.prepare(
    `SELECT id, content, category, importance, valence, arousal, dominant_emotion, created_at
     FROM memories
     WHERE user_id = ? AND dominant_emotion = ? AND is_forgotten = 0 AND is_archived = 0
     ORDER BY ABS(valence) DESC, created_at DESC LIMIT ?`
  ).all(userId, emotion, limit);
}

/**
 * getEmotionalProfile — Aggregate emotional stats for a user.
 */
export function getEmotionalProfile(userId: number = 1): any {
  const stats = db.prepare(
    `SELECT dominant_emotion, COUNT(*) as count, AVG(valence) as avg_valence, AVG(arousal) as avg_arousal
     FROM memories
     WHERE user_id = ? AND dominant_emotion IS NOT NULL AND is_forgotten = 0
     GROUP BY dominant_emotion ORDER BY count DESC`
  ).all(userId);

  const overall = db.prepare(
    `SELECT AVG(valence) as avg_valence, AVG(arousal) as avg_arousal,
       SUM(CASE WHEN valence > 0.2 THEN 1 ELSE 0 END) as positive_count,
       SUM(CASE WHEN valence < -0.2 THEN 1 ELSE 0 END) as negative_count,
       SUM(CASE WHEN valence BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END) as neutral_count
     FROM memories WHERE user_id = ? AND valence IS NOT NULL AND is_forgotten = 0`
  ).get(userId);

  return { emotions: stats, overall };
}
