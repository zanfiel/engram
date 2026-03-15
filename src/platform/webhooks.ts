// ============================================================================
// WEBHOOKS — Event dispatch
// ============================================================================

import { db } from "../db/index.ts";
import { log } from "../config/logger.ts";
import { createHmac } from "crypto";

export async function emitWebhookEvent(
  event: string,
  payload: Record<string, unknown>,
  userId: number = 1
): Promise<void> {
  const hooks = getActiveWebhooks.all(userId) as Array<{
    id: number; url: string; events: string; secret: string | null;
  }>;

  for (const hook of hooks) {
    try {
      const events = JSON.parse(hook.events) as string[];
      if (!events.includes("*") && !events.includes(event)) continue;

      const body = JSON.stringify({ event, timestamp: new Date().toISOString(), data: payload });
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (hook.secret) {
        const hmac = createHmac("sha256", hook.secret).update(body).digest("hex");
        headers["X-Engram-Signature"] = `sha256=${hmac}`;
      }

      fetch(hook.url, { method: "POST", headers, body, signal: AbortSignal.timeout(10000) })
        .then(resp => {
          if (resp.ok) { webhookTriggered.run(hook.id); }
          else { webhookFailed.run(hook.id); }
        })
        .catch(() => { webhookFailed.run(hook.id); });
    } catch {}
  }
}
