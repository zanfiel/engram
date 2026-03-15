// ============================================================================
// HELPERS — Response builders, security headers, FTS sanitization
// ============================================================================

import { CORS_ORIGIN } from "../config/index.ts";
import { log } from "../config/logger.ts";

export function securityHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": CORS_ORIGIN,
    "Access-Control-Allow-Methods": "GET, POST, PATCH, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Space, X-Engram-Space, X-Request-Id",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'",
    ...extra,
  };
}

export function json(data: unknown, status = 200, extra: Record<string, string> = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: securityHeaders({ "Content-Type": "application/json", ...extra }),
  });
}

export function errorResponse(message: string, status = 400, requestId?: string) {
  return json({ error: message, ...(requestId ? { request_id: requestId } : {}) }, status);
}

// S7 FIX: Safe error response — logs real error, returns generic message to client
export function safeError(label: string, e: any, status = 500, requestId?: string): Response {
  log.error({ msg: `${label}_failed`, error: e?.message, stack: e?.stack?.split("\n")[1]?.trim() });
  return errorResponse(`${label} failed`, status, requestId);
}

export function sanitizeFTS(query: string): string {
  return query
    .replace(/[^\w\s-]/g, "")
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => `"${w}"`)
    .join(" OR ");
}
