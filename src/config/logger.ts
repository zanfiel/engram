// ============================================================================
// LOGGER — Structured JSON logging
// ============================================================================

import { LOG_LEVEL } from "../config/index.ts";

function logPayload(args: any[]): Record<string, any> {
  if (args.length === 1 && typeof args[0] === "object") return args[0];
  return { msg: args.map(a => typeof a === "string" ? a : JSON.stringify(a)).join(" ") };
}

export const log = {
  debug: (...args: any[]) => { if (LOG_LEVEL <= 0) console.log(JSON.stringify({ level: "debug", ts: new Date().toISOString(), ...logPayload(args) })); },
  info: (...args: any[]) => { if (LOG_LEVEL <= 1) console.log(JSON.stringify({ level: "info", ts: new Date().toISOString(), ...logPayload(args) })); },
  warn: (...args: any[]) => { if (LOG_LEVEL <= 2) console.warn(JSON.stringify({ level: "warn", ts: new Date().toISOString(), ...logPayload(args) })); },
  error: (...args: any[]) => { if (LOG_LEVEL <= 3) console.error(JSON.stringify({ level: "error", ts: new Date().toISOString(), ...logPayload(args) })); },
};
