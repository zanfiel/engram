// ============================================================================
// AUTH — API key authentication, rate limiting, RBAC
// ============================================================================

import { createHash } from "crypto";
import { db } from "../db/index.ts";
import { RATE_WINDOW_MS, OPEN_ACCESS } from "../config/index.ts";

export interface AuthContext {
  user_id: number;
  space_id: number | null;
  key_id: number | null;
  agent_id: number | null;
  scopes: string[];
  is_admin: boolean;
}

// Auth error signals (rate limit, bad space, etc.) - never silently fall through to getAuthOrDefault
export interface AuthError { error: string; status: number; headers?: Record<string, string> }
export function isAuthError(r: AuthContext | AuthError | null): r is AuthError {
  return r !== null && typeof r === "object" && "error" in r;
}

const rateLimitMap = new Map<number, { count: number; reset: number }>();

export function authenticate(req: Request): AuthContext | AuthError | null {
  const authHeader = req.headers.get("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer eg_")) return null;

  const key = authHeader.slice(7); // strip "Bearer "
  const prefix = key.slice(0, 11); // "eg_" + 8 chars
  const hash = createHash("sha256").update(key).digest("hex");

  const row = db.prepare(
    `SELECT ak.id, ak.user_id, ak.scopes, ak.rate_limit, ak.agent_id, u.is_admin, u.role
     FROM api_keys ak JOIN users u ON ak.user_id = u.id
     WHERE ak.key_prefix = ? AND ak.key_hash = ? AND ak.is_active = 1`
  ).get(prefix, hash) as any;

  if (!row) return null;

  // Update last_used_at (non-blocking)
  db.prepare("UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?").run(row.id);

  // Rate limiting
  const now = Date.now();
  let rl = rateLimitMap.get(row.id);
  if (!rl || now > rl.reset) {
    rl = { count: 0, reset: now + RATE_WINDOW_MS };
    rateLimitMap.set(row.id, rl);
  }
  rl.count++;
  if (rl.count > row.rate_limit) return { error: "Rate limit exceeded", status: 429, headers: { "Retry-After": String(Math.ceil((rl.reset - now) / 1000)) } };

  // Determine space — only filter by space if explicitly requested
  let space_id: number | null = null;
  const spaceHeader = req.headers.get("X-Space") || req.headers.get("X-Engram-Space");
  if (spaceHeader) {
    const space = db.prepare("SELECT id FROM spaces WHERE user_id = ? AND name = ?").get(row.user_id, spaceHeader) as any;
    if (!space) return { error: `Space '${spaceHeader}' not found`, status: 400 };
    space_id = space.id;
  }

  // RBAC: derive scopes from role (role overrides key scopes for safety)
  const role = row.role || "writer";
  let effectiveScopes: string[];
  if (role === "admin") {
    effectiveScopes = (row.scopes || "read,write,admin").split(",");
  } else if (role === "reader") {
    // Reader can ONLY read, regardless of what the key says
    effectiveScopes = ["read"];
  } else {
    // Writer: can read + write, but not admin
    const keyScopes = (row.scopes || "read,write").split(",");
    effectiveScopes = keyScopes.filter((s: string) => s !== "admin");
  }

  // Resolve agent identity — check if key is linked to an active agent
  let agent_id: number | null = row.agent_id ?? null;
  if (agent_id) {
    const agent = db.prepare("SELECT id FROM agents WHERE id = ? AND is_active = 1").get(agent_id) as any;
    if (!agent) agent_id = null; // agent revoked, unlink
  }

  return {
    user_id: row.user_id,
    space_id,
    key_id: row.id,
    agent_id,
    scopes: effectiveScopes,
    is_admin: !!row.is_admin && role === "admin",
  };
}

export function getAuthOrDefault(req: Request, guiAuthed: (req: Request) => boolean): AuthContext | AuthError | null {
  const auth = authenticate(req);
  if (isAuthError(auth)) return auth; // propagate rate limit, bad space, etc.
  if (auth) return auth;
  // GUI cookie auth
  if (guiAuthed(req)) {
    return { user_id: 1, space_id: null, key_id: null, agent_id: null, scopes: ["read", "write"], is_admin: false };
  }
  if (OPEN_ACCESS) {
    // S7 FIX: OPEN_ACCESS grants read+write but NOT admin
    return { user_id: 1, space_id: null, key_id: null, agent_id: null, scopes: ["read", "write"], is_admin: false };
  }
  return null;
}

export function hasScope(auth: AuthContext, scope: string): boolean {
  return auth.scopes.includes("all") || auth.scopes.includes(scope) || auth.scopes.includes("admin");
}

export function generateApiKey(): { key: string; prefix: string; hash: string } {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  const raw = Array.from(bytes).map(b => b.toString(16).padStart(2, "0")).join("");
  const key = `eg_${raw}`;
  const prefix = key.slice(0, 11);
  const hash = createHash("sha256").update(key).digest("hex");
  return { key, prefix, hash };
}
