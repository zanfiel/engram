// ============================================================================
// GUI — Web GUI authentication, cookie signing, HTML serving
// ============================================================================

import { createHash, randomUUID, timingSafeEqual } from "crypto";
import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { DATA_DIR, GUI_AUTH_MAX_ATTEMPTS, GUI_AUTH_WINDOW_MS, GUI_AUTH_LOCKOUT_MS } from "../config/index.ts";
import { log } from "../config/logger.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SERVER_DIR = resolve(__dirname, "../..");

// GUI password resolution
export const GUI_PASSWORD = (() => {
  if (process.env.ENGRAM_GUI_PASSWORD) return process.env.ENGRAM_GUI_PASSWORD;
  if (process.env.MEGAMIND_GUI_PASSWORD) {
    log.warn({ msg: "deprecated_env", var: "MEGAMIND_GUI_PASSWORD", use: "ENGRAM_GUI_PASSWORD" });
    return process.env.MEGAMIND_GUI_PASSWORD;
  }
  log.warn({ msg: "WARNING_default_gui_password", detail: "ENGRAM_GUI_PASSWORD not set — using 'changeme'. Set a strong password for production!" });
  return "changeme";
})();

// HMAC secret for cookie signing (top-level await)
const GUI_HMAC_SECRET = await (async () => {
  if (process.env.ENGRAM_HMAC_SECRET) return process.env.ENGRAM_HMAC_SECRET;
  const secretFile = resolve(DATA_DIR, ".hmac_secret");
  try {
    return readFileSync(secretFile, "utf-8");
  } catch {
    const secret = randomUUID() + randomUUID();
    writeFileSync(secretFile, secret);
    log.info({ msg: "generated_hmac_secret", path: secretFile });
    return secret;
  }
})();

export const GUI_COOKIE_MAX_AGE = 7 * 24 * 60 * 60;

export function guiSignCookie(ts: number): string {
  const h = createHash("sha256");
  h.update(GUI_HMAC_SECRET + ":" + String(ts));
  return ts + "." + h.digest("hex");
}

export function guiVerifyCookie(cookie: string): boolean {
  const dot = cookie.indexOf(".");
  if (dot < 1) return false;
  const ts = cookie.substring(0, dot), sig = cookie.substring(dot + 1);
  const t = parseInt(ts);
  if (isNaN(t) || Date.now() / 1000 - t > GUI_COOKIE_MAX_AGE) return false;
  const h = createHash("sha256");
  h.update(GUI_HMAC_SECRET + ":" + ts);
  const expected = h.digest("hex");
  if (expected.length !== sig.length) return false;
  return timingSafeEqual(Buffer.from(expected), Buffer.from(sig));
}

export function guiAuthed(req: Request): boolean {
  const ck = (req.headers.get("cookie") || "")
    .split(";").map(c => c.trim())
    .find(c => c.startsWith("engram_auth="));
  if (!ck) return false;
  return guiVerifyCookie(ck.split("=").slice(1).join("="));
}

// HTML serving
let GUI_HTML = readFileSync(resolve(SERVER_DIR, "engram-gui.html"), "utf-8");
let LOGIN_HTML = readFileSync(resolve(SERVER_DIR, "engram-login.html"), "utf-8");
const GUI_HOT_RELOAD = process.env.ENGRAM_HOT_RELOAD === "1";

export function getGuiHtml(): string {
  if (GUI_HOT_RELOAD) return readFileSync(resolve(SERVER_DIR, "engram-gui.html"), "utf-8");
  return GUI_HTML;
}

export function getLoginHtml(): string {
  if (GUI_HOT_RELOAD) return readFileSync(resolve(SERVER_DIR, "engram-login.html"), "utf-8");
  return LOGIN_HTML;
}

export function reloadGuiHtml(): void {
  GUI_HTML = readFileSync(resolve(SERVER_DIR, "engram-gui.html"), "utf-8");
  LOGIN_HTML = readFileSync(resolve(SERVER_DIR, "engram-login.html"), "utf-8");
  log.info({ msg: "gui_reloaded", trigger: "SIGHUP" });
}
