// ============================================================================
// PREDICTIVE RECALL — Proactive memory surfacing
// Nobody in the AI memory space does pre-emptive recall.
//
// Instead of waiting for a query, this surfaces memories BEFORE asked
// based on: time of day, day of week, project context, activity patterns.
//
// "It's Monday morning in the zanverse project — here's where you left
// off Friday, issues that might need attention, and scheduled tasks."
// ============================================================================

import { db, getTemporalPatternsForNow, insertTemporalPattern } from "../db/index.ts";
import { log } from "../config/logger.ts";

interface PredictiveContext {
  time_context: string;
  predicted_categories: string[];
  predicted_project: { id: number; name: string } | null;
  proactive_memories: Array<{
    id: number;
    content: string;
    category: string;
    reason: string;
    score: number;
  }>;
  suggested_actions: string[];
}

/**
 * trackTemporalAccess — Record that a memory category was accessed at this time.
 * Builds the temporal pattern database over time.
 */
export function trackTemporalAccess(
  userId: number,
  category: string,
  projectId: number | null = null
): void {
  const now = new Date();
  const dow = now.getDay(); // 0=Sun, 6=Sat
  const hour = now.getHours();
  try {
    insertTemporalPattern.run(userId, dow, hour, category, projectId);
  } catch (e) {
    // Ignore — pattern tracking is best-effort
  }
}

/**
 * predictiveRecall — Generate proactive context for the current moment.
 * Called at session start or periodically.
 */
export function predictiveRecall(userId: number = 1): PredictiveContext {
  const now = new Date();
  const dow = now.getDay();
  const hour = now.getHours();

  // Get temporal patterns for this time slot
  const patterns = getTemporalPatternsForNow.all(userId, dow, hour, 10) as any[];

  const predictedCategories = [...new Set(patterns.map(p => p.category).filter(Boolean))];
  const predictedProject = patterns.find(p => p.project_id)
    ? { id: patterns.find(p => p.project_id).project_id, name: patterns.find(p => p.project_id).project_name }
    : null;

  // Get unfinished work (recent tasks that might need continuation)
  const unfinished = db.prepare(
    `SELECT id, content, category, created_at, importance
     FROM memories
     WHERE user_id = ? AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1
       AND category = 'task' AND is_static = 0
       AND created_at > datetime('now', '-3 days')
     ORDER BY importance DESC, created_at DESC LIMIT 5`
  ).all(userId) as any[];

  // Get active issues
  const activeIssues = db.prepare(
    `SELECT id, content, category, created_at, importance
     FROM memories
     WHERE user_id = ? AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1
       AND category = 'issue'
       AND created_at > datetime('now', '-7 days')
     ORDER BY importance DESC LIMIT 3`
  ).all(userId) as any[];

  // Get decisions that might be relevant to predicted project
  let projectMemories: any[] = [];
  if (predictedProject) {
    projectMemories = db.prepare(
      `SELECT m.id, m.content, m.category, m.created_at, m.importance
       FROM memories m
       JOIN memory_projects mp ON mp.memory_id = m.id
       WHERE mp.project_id = ? AND m.is_forgotten = 0 AND m.is_archived = 0
       ORDER BY m.created_at DESC LIMIT 5`
    ).all(predictedProject.id) as any[];
  }

  // Get memories from last session (continuity)
  const lastSession = db.prepare(
    `SELECT id, content, category, created_at, importance
     FROM memories
     WHERE user_id = ? AND is_forgotten = 0 AND is_archived = 0 AND is_latest = 1
     ORDER BY created_at DESC LIMIT 5`
  ).all(userId) as any[];

  // Assemble proactive memories with reasons
  const proactive: PredictiveContext["proactive_memories"] = [];
  const seen = new Set<number>();

  for (const m of unfinished) {
    if (!seen.has(m.id)) {
      seen.add(m.id);
      proactive.push({ ...m, reason: "unfinished_task", score: m.importance / 10 + 0.3 });
    }
  }
  for (const m of activeIssues) {
    if (!seen.has(m.id)) {
      seen.add(m.id);
      proactive.push({ ...m, reason: "active_issue", score: m.importance / 10 + 0.2 });
    }
  }
  for (const m of projectMemories) {
    if (!seen.has(m.id)) {
      seen.add(m.id);
      proactive.push({ ...m, reason: "predicted_project", score: m.importance / 10 + 0.1 });
    }
  }
  for (const m of lastSession.slice(0, 3)) {
    if (!seen.has(m.id)) {
      seen.add(m.id);
      proactive.push({ ...m, reason: "session_continuity", score: m.importance / 10 });
    }
  }

  // Sort by composite score
  proactive.sort((a, b) => b.score - a.score);

  // Generate time context string
  const dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const timeContext = `${dayNames[dow]} ${hour < 12 ? "morning" : hour < 17 ? "afternoon" : "evening"}`;

  // Suggest actions based on patterns
  const suggestions: string[] = [];
  if (unfinished.length > 0) suggestions.push(`Continue: ${unfinished[0].content.slice(0, 80)}`);
  if (activeIssues.length > 0) suggestions.push(`Address issue: ${activeIssues[0].content.slice(0, 80)}`);
  if (predictedProject) suggestions.push(`Expected project: ${predictedProject.name}`);

  return {
    time_context: timeContext,
    predicted_categories: predictedCategories,
    predicted_project: predictedProject,
    proactive_memories: proactive.slice(0, 10),
    suggested_actions: suggestions,
  };
}
