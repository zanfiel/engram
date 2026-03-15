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

// ── MCP Message Signing (nonce + timestamp replay protection) ─────

export interface SignedEnvelope {
  message: any;
  agent_id: number;
  nonce: string;
  timestamp: string;
  signature: string;
}

const DEFAULT_MAX_AGE_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Wrap a JSON-RPC message in a signed envelope with nonce + timestamp.
 */
export function signMessage(secret: string, agentId: number, message: any): SignedEnvelope {
  const nonce = randomUUID();
  const timestamp = new Date().toISOString();
  const payload = canonicalJSON({ agent_id: agentId, message, nonce, timestamp });
  const signature = hmacSign(secret, payload);
  return { message, agent_id: agentId, nonce, timestamp, signature };
}

/**
 * Verify a signed envelope. Checks signature, timestamp window, and nonce replay.
 * Pass a NonceTracker to enable replay protection.
 */
export function verifyMessage(
  secret: string,
  envelope: SignedEnvelope,
  tracker?: NonceTracker,
  maxAgeMs: number = DEFAULT_MAX_AGE_MS,
): { valid: boolean; reason: string | null } {
  // Check timestamp window
  const msgTime = new Date(envelope.timestamp).getTime();
  if (isNaN(msgTime)) return { valid: false, reason: "invalid_timestamp" };
  const age = Date.now() - msgTime;
  if (age > maxAgeMs) return { valid: false, reason: "expired" };
  if (age < -30_000) return { valid: false, reason: "future_timestamp" }; // 30s clock skew tolerance

  // Check nonce replay
  if (tracker) {
    if (tracker.seen(envelope.nonce)) return { valid: false, reason: "replay_detected" };
    tracker.add(envelope.nonce);
  }

  // Verify signature
  const payload = canonicalJSON({ agent_id: envelope.agent_id, message: envelope.message, nonce: envelope.nonce, timestamp: envelope.timestamp });
  const sigValid = hmacVerify(secret, payload, envelope.signature);
  if (!sigValid) return { valid: false, reason: "invalid_signature" };

  return { valid: true, reason: null };
}

// ── Nonce Tracker (replay protection) ─────────────────────────────

export class NonceTracker {
  private nonces = new Map<string, number>(); // nonce → timestamp
  private maxAgeMs: number;
  private sweepInterval: ReturnType<typeof setInterval> | null = null;

  constructor(maxAgeMs: number = DEFAULT_MAX_AGE_MS) {
    this.maxAgeMs = maxAgeMs;
    // Sweep expired nonces every minute
    this.sweepInterval = setInterval(() => this.sweep(), 60_000);
    if (this.sweepInterval.unref) this.sweepInterval.unref();
  }

  seen(nonce: string): boolean {
    return this.nonces.has(nonce);
  }

  add(nonce: string): void {
    this.nonces.set(nonce, Date.now());
  }

  sweep(): void {
    const cutoff = Date.now() - this.maxAgeMs;
    for (const [nonce, ts] of this.nonces) {
      if (ts < cutoff) this.nonces.delete(nonce);
    }
  }

  close(): void {
    if (this.sweepInterval) clearInterval(this.sweepInterval);
  }
}

// ── Tool Integrity Binding ────────────────────────────────────────

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: any;
}

export interface SignedToolManifest {
  tools: Array<ToolDefinition & { hash: string }>;
  manifest_hash: string;
  signature: string;
  signed_at: string;
}

/** Hash a single tool definition (name + description + schema) */
export function hashTool(tool: ToolDefinition): string {
  return sha256(canonicalJSON({ name: tool.name, description: tool.description, inputSchema: tool.inputSchema }));
}

/**
 * Sign a manifest of tool definitions.
 * Clients can verify the manifest to detect tool poisoning.
 */
export function signToolManifest(secret: string, tools: ToolDefinition[]): SignedToolManifest {
  const hashed = tools.map(t => ({ ...t, hash: hashTool(t) }));
  const manifestHash = sha256(hashed.map(t => t.hash).join(":"));
  const signedAt = new Date().toISOString();
  const signature = hmacSign(secret, `${manifestHash}:${signedAt}`);
  return { tools: hashed, manifest_hash: manifestHash, signature, signed_at: signedAt };
}

/** Verify a tool manifest hasn't been tampered with */
export function verifyToolManifest(secret: string, manifest: SignedToolManifest): { valid: boolean; tampered: string[] } {
  // Verify overall signature
  const sigValid = hmacVerify(secret, `${manifest.manifest_hash}:${manifest.signed_at}`, manifest.signature);
  if (!sigValid) return { valid: false, tampered: ["manifest_signature"] };

  // Verify individual tool hashes
  const tampered: string[] = [];
  for (const tool of manifest.tools) {
    const expected = hashTool(tool);
    if (expected !== tool.hash) tampered.push(tool.name);
  }

  // Verify manifest hash
  const expectedManifest = sha256(manifest.tools.map(t => t.hash).join(":"));
  if (expectedManifest !== manifest.manifest_hash) tampered.push("manifest_hash");

  return { valid: tampered.length === 0, tampered };
}
