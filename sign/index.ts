// ============================================================================
// @zanverse/sign — Cryptographic identity & execution signing for AI agents
// Zero dependencies. Node >= 18.
// ============================================================================

import { createHash, createHmac, timingSafeEqual, randomUUID } from "crypto";

/** Deterministic JSON — recursive key sort for canonical representation */
export function canonicalJSON(obj: any): string {
  if (obj === null || obj === undefined) return JSON.stringify(obj);
  if (typeof obj !== "object") return JSON.stringify(obj);
  if (Array.isArray(obj)) return "[" + obj.map(canonicalJSON).join(",") + "]";
  return "{" + Object.keys(obj).sort().map(k => JSON.stringify(k) + ":" + canonicalJSON(obj[k])).join(",") + "}";
}

export function sha256(data: string): string {
  return createHash("sha256").update(data, "utf8").digest("hex");
}

export function hmacSign(secret: string, data: string): string {
  return createHmac("sha256", secret).update(data, "utf8").digest("hex");
}

export function hmacVerify(secret: string, data: string, signature: string): boolean {
  const expected = hmacSign(secret, data);
  try {
    return timingSafeEqual(Buffer.from(expected, "hex"), Buffer.from(signature, "hex"));
  } catch {
    return false;
  }
}

// ── Execution Signing ─────────────────────────────────────────────

export interface SignedExecution {
  execution_id: string;
  agent_id: number;
  action: string;
  input_hash: string;
  output_hash: string | null;
  execution_hash: string;
  signature: string;
  signed_at: string;
}

export function signExecution(
  secret: string,
  agentId: number,
  action: string,
  input: any,
  output?: any,
): SignedExecution {
  const executionId = `exec_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
  const inputHash = sha256(canonicalJSON(input));
  const outputHash = output !== undefined ? sha256(canonicalJSON(output)) : null;
  const signedAt = new Date().toISOString();
  const executionHash = sha256(`${agentId}:${action}:${inputHash}:${outputHash ?? "null"}:${signedAt}`);
  const signature = hmacSign(secret, executionHash);
  return { execution_id: executionId, agent_id: agentId, action, input_hash: inputHash, output_hash: outputHash, execution_hash: executionHash, signature, signed_at: signedAt };
}

export function verifyExecution(secret: string, execution: SignedExecution): boolean {
  return hmacVerify(secret, execution.execution_hash, execution.signature);
}

// ── Agent Passports ───────────────────────────────────────────────

export interface PassportPayload {
  agent_id: number;
  name: string;
  category: string | null;
  trust_score: number;
  code_hash: string | null;
  issued_at: string;
  expires_at: string;
}

export interface Passport extends PassportPayload {
  signature: string;
}

export function createPassport(
  secret: string,
  agent: { id: number; name: string; category?: string | null; trust_score: number; code_hash?: string | null },
  ttlMs: number = 3_600_000, // 1 hour default
): Passport {
  const payload: PassportPayload = {
    agent_id: agent.id,
    name: agent.name,
    category: agent.category ?? null,
    trust_score: agent.trust_score,
    code_hash: agent.code_hash ?? null,
    issued_at: new Date().toISOString(),
    expires_at: new Date(Date.now() + ttlMs).toISOString(),
  };
  const signature = hmacSign(secret, canonicalJSON(payload));
  return { ...payload, signature };
}

export interface PassportVerification {
  valid: boolean;
  reason: string | null;
  expired: boolean;
}

export function verifyPassport(secret: string, passport: Passport): PassportVerification {
  const { signature, ...payload } = passport;
  const expired = new Date(passport.expires_at) < new Date();
  const sigValid = hmacVerify(secret, canonicalJSON(payload), signature);
  if (!sigValid) return { valid: false, reason: "invalid_signature", expired };
  if (expired) return { valid: false, reason: "expired", expired: true };
  return { valid: true, reason: null, expired: false };
}

// ── Trust Score Calculation ───────────────────────────────────────

export interface TrustInputs {
  total_ops: number;
  successful_ops: number;
  failed_ops: number;
  guard_allows: number;
  guard_warns: number;
  guard_blocks: number;
}

/**
 * Compute behavioral trust score (0-100).
 * Starts at 50 (neutral), adjusts based on history.
 */
export function computeTrustScore(inputs: TrustInputs): number {
  const base = 50;
  // Reward successful operations (diminishing returns, cap at 25)
  const successBonus = Math.min(25, inputs.successful_ops * 0.1);
  // Reward clean guard passes (cap at 15)
  const guardBonus = Math.min(15, inputs.guard_allows * 0.5);
  // Penalize warnings and blocks
  const warnPenalty = inputs.guard_warns * 2;
  const blockPenalty = inputs.guard_blocks * 5;
  // Penalize high error rate
  const errorPenalty = inputs.total_ops > 0
    ? Math.min(20, (inputs.failed_ops / inputs.total_ops) * 40)
    : 0;

  const score = base + successBonus + guardBonus - warnPenalty - blockPenalty - errorPenalty;
  return Math.max(0, Math.min(100, Math.round(score * 10) / 10));
}

/** Generate a new HMAC signing secret */
export function generateSigningSecret(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes).map(b => b.toString(16).padStart(2, "0")).join("");
}
