// ============================================================================
// HELPERS — Response builders, security headers, FTS sanitization
// ============================================================================

import { CORS_ORIGIN } from "../config/index.ts";
import { log } from "../config/logger.ts";
import { resolve as dnsResolve } from "dns/promises";

// ── SSRF protection ──────────────────────────────────────────────────

export function isPrivateHostname(hostname: string): boolean {
  // Strip brackets from IPv6 literals like [::1]
  let h = hostname;
  if (h.startsWith("[") && h.endsWith("]")) {
    h = h.slice(1, -1);
  }
  h = h.toLowerCase();

  // IPv4 loopback + unspecified
  if (h === "localhost" || h === "127.0.0.1" || h === "0.0.0.0") return true;

  // IPv4 private ranges
  if (h.startsWith("10.") || h.startsWith("192.168.") ||
      h.startsWith("100.64.") || h.startsWith("169.254.")) return true;

  // 172.16.0.0/12
  if (h.startsWith("172.")) {
    const second = parseInt(h.split(".")[1], 10);
    if (second >= 16 && second <= 31) return true;
  }

  // IPv6 loopback and unspecified
  if (h === "::1" || h === "::") return true;

  // IPv6 ULA (fc00::/7 covers fc and fd prefixes)
  if (h.startsWith("fc") || h.startsWith("fd")) return true;

  // IPv6 link-local
  if (h.startsWith("fe80")) return true;

  // Local domain suffixes
  if (h.endsWith(".local") || h.endsWith(".internal")) return true;

  return false;
}

export async function validatePublicUrlWithDNS(rawUrl: string, label: string): Promise<string | null> {
  try {
    const parsed = new URL(rawUrl);
    if (!["http:", "https:"].includes(parsed.protocol)) return `${label} must be http or https`;

    if (isPrivateHostname(parsed.hostname)) return `${label} cannot point to private/internal addresses`;

    // DNS resolution to prevent TOCTOU rebinding attacks
    try {
      const host = parsed.hostname.replace(/^\[|\]$/g, "");
      const addresses = await dnsResolve(host);
      for (const addr of addresses) {
        if (isPrivateHostname(addr)) {
          return `${label} resolves to private/internal address`;
        }
      }
    } catch {
      return `${label} hostname could not be resolved`;
    }

    return null;
  } catch {
    return `Invalid ${label.toLowerCase()}`;
  }
}

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
