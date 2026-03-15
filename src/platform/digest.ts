// ============================================================================
// DIGESTS — Scheduled summary delivery via webhooks
// ============================================================================

import { db } from "../db/index.ts";
import { log } from "../config/logger.ts";
import { validatePublicUrlWithDNS } from "../helpers/index.ts";

export async function buildDigestPayload(digest: any, userId: number): Promise<any> {
  const now = new Date();
  const sinceStr = digest.last_sent_at || new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString().replace("T", " ").replace("Z", "");

  const payload: any = {
    type: "engram_digest",
    schedule: digest.schedule,
    generated_at: now.toISOString(),
    period_since: sinceStr,
  };

  if (digest.include_stats) {
    const total = (db.prepare("SELECT COUNT(*) as c FROM memories WHERE is_forgotten = 0 AND user_id = ?").get(userId) as any).c;
    const newCount = (db.prepare("SELECT COUNT(*) as c FROM memories WHERE created_at > ? AND user_id = ?").get(sinceStr, userId) as any).c;
    const archivedCount = (db.prepare("SELECT COUNT(*) as c FROM memories WHERE is_archived = 1 AND updated_at > ? AND user_id = ?").get(sinceStr, userId) as any).c;
    const updatedCount = (db.prepare("SELECT COUNT(*) as c FROM memories WHERE updated_at > ? AND created_at <= ? AND user_id = ?").get(sinceStr, sinceStr, userId) as any).c;

    payload.stats = { total_memories: total, new: newCount, updated: updatedCount, archived: archivedCount };
  }

  if (digest.include_new_memories) {
    const newMems = db.prepare(
      `SELECT id, content, category, importance, tags, created_at
       FROM memories WHERE created_at > ? AND is_forgotten = 0 AND user_id = ?
       ORDER BY importance DESC, created_at DESC LIMIT 20`
    ).all(sinceStr, userId) as any[];

    payload.new_memories = newMems.map((m: any) => ({
      id: m.id, content: m.content.substring(0, 300), category: m.category,
      importance: m.importance, tags: m.tags ? JSON.parse(m.tags) : [],
    }));
  }

  if (digest.include_contradictions) {
    const contras = db.prepare(
      `SELECT ml.source_id, ml.target_id,
         ms.content as a_content, mt.content as b_content
        FROM memory_links ml
        JOIN memories ms ON ml.source_id = ms.id
        JOIN memories mt ON ml.target_id = mt.id
        WHERE ml.type = 'contradicts' AND ml.created_at > ?
          AND ms.user_id = ? AND mt.user_id = ?
          AND ms.is_forgotten = 0 AND mt.is_forgotten = 0
        ORDER BY ml.created_at DESC LIMIT 10`
    ).all(sinceStr, userId, userId) as any[];

    payload.contradictions = contras.map((c: any) => ({
      memory_a: { id: c.source_id, content: c.a_content.substring(0, 200) },
      memory_b: { id: c.target_id, content: c.b_content.substring(0, 200) },
    }));
  }

  if (digest.include_reflections) {
    const refl = db.prepare(
      `SELECT content, themes, period_start, period_end, created_at FROM reflections
       WHERE user_id = ? AND created_at > ? ORDER BY created_at DESC LIMIT 1`
    ).get(userId, sinceStr) as any;

    if (refl) {
      payload.reflection = {
        content: refl.content,
        themes: refl.themes ? JSON.parse(refl.themes) : [],
        period: { start: refl.period_start, end: refl.period_end },
      };
    }
  }

  return payload;
}

export async function sendDigestWebhook(digest: any, payload: any): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (digest.webhook_secret) {
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey("raw", encoder.encode(digest.webhook_secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    const sig = await crypto.subtle.sign("HMAC", key, encoder.encode(JSON.stringify(payload)));
    headers["X-Engram-Signature"] = Array.from(new Uint8Array(sig)).map(b => b.toString(16).padStart(2, "0")).join("");
  }

  // Dispatch-time SSRF revalidation (prevents DNS rebinding)
  const urlError = await validatePublicUrlWithDNS(digest.webhook_url, "Digest webhook URL");
  if (urlError) {
    log.error({ msg: "digest_ssrf_blocked", digest_id: digest.id, error: urlError });
    throw new Error(urlError);
  }

  try {
    const resp = await fetch(digest.webhook_url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });

    if (resp.ok) {
      db.prepare("UPDATE digests SET last_sent_at = datetime('now'), next_send_at = ? WHERE id = ?").run(
        calculateNextSend(digest.schedule), digest.id
      );
    } else {
      db.prepare("UPDATE digests SET next_send_at = ? WHERE id = ?").run(
        calculateNextSend(digest.schedule), digest.id
      );
      log.error({ msg: "digest_webhook_error", digest_id: digest.id, status: resp.status });
    }
  } catch (e: any) {
    db.prepare("UPDATE digests SET next_send_at = ? WHERE id = ?").run(
      calculateNextSend(digest.schedule), digest.id
    );
    log.error({ msg: "digest_webhook_failed", digest_id: digest.id, error: e.message });
  }
}

export function calculateNextSend(schedule: string): string {
  const now = new Date();
  let next: Date;
  if (schedule === "hourly") next = new Date(now.getTime() + 60 * 60 * 1000);
  else if (schedule === "weekly") next = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
  else next = new Date(now.getTime() + 24 * 60 * 60 * 1000);
  return next.toISOString().replace("T", " ").replace("Z", "");
}

export async function processScheduledDigests(): Promise<number> {
  const nowStr = new Date().toISOString().replace("T", " ").replace("Z", "");
  const due = db.prepare(
    `SELECT * FROM digests WHERE active = 1 AND next_send_at <= ? ORDER BY next_send_at ASC LIMIT 10`
  ).all(nowStr) as any[];

  let sent = 0;
  for (const digest of due) {
    try {
      const payload = await buildDigestPayload(digest, digest.user_id);
      await sendDigestWebhook(digest, payload);
      sent++;
    } catch (e: any) {
      log.error({ msg: "digest_processing_failed", digest_id: digest.id, error: e.message });
    }
  }
  return sent;
}
