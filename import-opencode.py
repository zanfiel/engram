#!/usr/bin/env python3
"""
Import OpenCode sessions from SQLite DB into Engram conversations.
Reads directly from the OpenCode SQLite database (no CLI dependency).
Deduplicates by session_id — safe to run repeatedly.

Usage:
  python3 import-opencode.py                          # Windows local DB
  python3 import-opencode.py --agent opencode-rocky   # Override agent name
  python3 import-opencode.py --db /path/to/opencode.db
  python3 import-opencode.py --engram-url http://host:4200
  python3 import-opencode.py --dry-run                # Count only, no import
"""

import argparse
import json
import os
import platform
import sqlite3
import sys
import time
import urllib.request
import urllib.error

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Defaults
DEFAULT_ENGRAM_URL = "http://127.0.0.1:4200"
DEFAULT_AGENT = "opencode-windows" if platform.system() == "Windows" else "opencode-rocky"
BATCH_SIZE = 50  # messages per upsert batch
RATE_DELAY = 0.6  # seconds between API calls (Engram: 120 req/min)
MAX_BODY_SIZE = 900_000  # stay under 1MB limit
MAX_RETRIES = 3


def find_opencode_db():
    """Find the OpenCode SQLite database."""
    candidates = [
        os.path.expanduser("~/.local/share/opencode/opencode.db"),
    ]
    if platform.system() == "Windows":
        candidates.insert(0, os.path.join(os.environ.get("LOCALAPPDATA", ""), "opencode", "opencode.db"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def engram_get(base_url, endpoint, cookie=None):
    """GET from Engram."""
    headers = {}
    if cookie:
        headers["Cookie"] = f"engram_auth={cookie}"
    req = urllib.request.Request(f"{base_url}{endpoint}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200] if e.fp else ""
        print(f"  GET {endpoint} → HTTP {e.code}: {body}")
        return None
    except Exception as e:
        print(f"  GET {endpoint} → Error: {e}")
        return None


def engram_post(base_url, endpoint, data, cookie=None):
    """POST JSON to Engram."""
    body = json.dumps(data).encode()
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = f"engram_auth={cookie}"
    req = urllib.request.Request(f"{base_url}{endpoint}", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()[:300] if e.fp else ""
        print(f"  POST {endpoint} → HTTP {e.code}: {err_body}")
        return None
    except Exception as e:
        print(f"  POST {endpoint} → Error: {e}")
        return None


def get_gui_cookie(base_url, password="changeme"):
    """Get a GUI auth cookie for Engram v5.4+ (which requires auth)."""
    try:
        body = json.dumps({"password": password}).encode()
        req = urllib.request.Request(
            f"{base_url}/gui/auth", data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            for h in resp.getheaders():
                if h[0].lower() == "set-cookie" and "engram_auth=" in h[1]:
                    return h[1].split("engram_auth=")[1].split(";")[0]
    except Exception as e:
        print(f"Warning: Could not get GUI cookie: {e}")
    return None


def get_existing_session_ids(base_url, cookie=None):
    """Get set of session_ids already in Engram."""
    existing = set()
    data = engram_get(base_url, "/conversations?limit=10000", cookie)
    if data and "results" in data:
        for c in data["results"]:
            sid = c.get("session_id")
            if sid:
                existing.add(sid)
    return existing


def ts_to_iso(ms_or_s):
    """Convert timestamp (ms or seconds) to ISO string."""
    if ms_or_s is None:
        return None
    ts = ms_or_s / 1000 if ms_or_s > 1e12 else ms_or_s
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def main():
    parser = argparse.ArgumentParser(description="Import OpenCode sessions into Engram")
    parser.add_argument("--db", help="Path to opencode.db")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help=f"Agent name (default: {DEFAULT_AGENT})")
    parser.add_argument("--engram-url", default=DEFAULT_ENGRAM_URL, help=f"Engram URL (default: {DEFAULT_ENGRAM_URL})")
    parser.add_argument("--password", default="changeme", help="Engram GUI password")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't import")
    parser.add_argument("--force", action="store_true", help="Re-import even if session exists")
    args = parser.parse_args()

    # Find DB
    db_path = args.db or find_opencode_db()
    if not db_path or not os.path.exists(db_path):
        print(f"ERROR: OpenCode database not found. Tried: {db_path}")
        sys.exit(1)
    print(f"Database: {db_path} ({os.path.getsize(db_path) / 1024 / 1024:.1f} MB)")

    # Connect to OpenCode DB (read-only)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Count sessions and messages
    total_sessions = conn.execute("SELECT COUNT(*) FROM session").fetchone()[0]
    total_messages = conn.execute("SELECT COUNT(*) FROM message").fetchone()[0]
    total_parts = conn.execute("SELECT COUNT(*) FROM part").fetchone()[0]
    print(f"OpenCode DB: {total_sessions} sessions, {total_messages} messages, {total_parts} parts")

    # Check Engram health
    health = engram_get(args.engram_url, "/health")
    if not health or health.get("status") != "ok":
        print(f"ERROR: Engram not reachable at {args.engram_url}")
        sys.exit(1)
    print(f"Engram: v{health.get('version')}, {health.get('memories')} memories")

    # Get auth cookie
    cookie = get_gui_cookie(args.engram_url, args.password)
    if cookie:
        print(f"Auth: GUI cookie obtained")
    else:
        print(f"Auth: no cookie (using open access)")

    # Get existing session IDs
    existing_sids = get_existing_session_ids(args.engram_url, cookie)
    print(f"Already imported: {len(existing_sids)} sessions")

    # Get all sessions
    sessions = conn.execute(
        "SELECT id, title, time_created, time_updated FROM session ORDER BY time_created ASC"
    ).fetchall()

    # Filter to new sessions
    if args.force:
        new_sessions = sessions
    else:
        new_sessions = [s for s in sessions if s["id"] not in existing_sids]
    print(f"New sessions to import: {len(new_sessions)}")

    if args.dry_run:
        print("\n[DRY RUN] Would import:")
        for s in new_sessions[:20]:
            title = s["title"] or "untitled"
            ts = ts_to_iso(s["time_created"])
            msg_count = conn.execute(
                "SELECT COUNT(*) FROM message WHERE session_id = ?", (s["id"],)
            ).fetchone()[0]
            print(f"  {s['id'][:20]}... | {ts} | {msg_count} msgs | {title[:60]}")
        if len(new_sessions) > 20:
            print(f"  ... and {len(new_sessions) - 20} more")
        conn.close()
        return

    if not new_sessions:
        print("Nothing to import.")
        conn.close()
        return

    # Import each session
    imported = 0
    failed = 0
    total_msgs = 0

    for i, session in enumerate(new_sessions):
        sid = session["id"]
        title = session["title"] or "untitled"
        started_at = ts_to_iso(session["time_created"])

        # Get messages for this session with their parts
        messages = conn.execute(
            "SELECT id, data, time_created FROM message WHERE session_id = ? ORDER BY time_created ASC",
            (sid,)
        ).fetchall()

        if not messages:
            continue

        # Build conversation messages
        conv_messages = []
        for msg in messages:
            msg_data = json.loads(msg["data"])
            role = msg_data.get("role", "user")

            # Get parts for this message
            parts = conn.execute(
                "SELECT data FROM part WHERE message_id = ? ORDER BY time_created ASC",
                (msg["id"],)
            ).fetchall()

            # Extract text content from parts
            text_parts = []
            for part in parts:
                part_data = json.loads(part["data"])
                ptype = part_data.get("type", "")
                if ptype == "text" and part_data.get("text"):
                    text_parts.append(part_data["text"])
                elif ptype == "tool-invocation":
                    tool_name = part_data.get("toolName", "unknown")
                    tool_input = part_data.get("input", {})
                    # Compact tool call representation
                    if isinstance(tool_input, dict):
                        summary = json.dumps(tool_input)[:200]
                    else:
                        summary = str(tool_input)[:200]
                    text_parts.append(f"[Tool: {tool_name}] {summary}")
                elif ptype == "tool-result":
                    result_text = ""
                    for item in part_data.get("result", []):
                        if isinstance(item, dict) and item.get("type") == "text":
                            result_text += item.get("text", "")[:500]
                    if result_text:
                        text_parts.append(f"[Result] {result_text[:500]}")

            content = "\n".join(text_parts)
            if not content.strip():
                continue

            # Truncate very large messages (tool results can be huge)
            if len(content) > 8000:
                content = content[:8000] + "\n... [truncated]"

            conv_messages.append({
                "role": role,
                "content": content,
                "timestamp": ts_to_iso(msg["time_created"]),
            })

        if not conv_messages:
            continue

        # Truncate messages if body would exceed limit
        payload = {
            "agent": args.agent,
            "session_id": sid,
            "title": title,
            "messages": conv_messages,
        }
        body_size = len(json.dumps(payload).encode())
        if body_size > MAX_BODY_SIZE:
            # Progressively truncate message content
            for trunc in [4000, 2000, 1000, 500]:
                for m in payload["messages"]:
                    if len(m["content"]) > trunc:
                        m["content"] = m["content"][:trunc] + "\n... [truncated]"
                body_size = len(json.dumps(payload).encode())
                if body_size <= MAX_BODY_SIZE:
                    break
            if body_size > MAX_BODY_SIZE:
                # Last resort: drop oldest messages keeping first and last 20
                msgs = payload["messages"]
                if len(msgs) > 40:
                    payload["messages"] = msgs[:20] + msgs[-20:]
                    print(f"  Warning: truncated {len(msgs)} msgs to 40 for {sid}")

        # Upsert with retry
        result = None
        for attempt in range(MAX_RETRIES):
            result = engram_post(args.engram_url, "/conversations/upsert", payload, cookie)
            if result and result.get("id"):
                break
            if result is None:
                # Check if it was a 429
                wait = (attempt + 1) * 30
                print(f"  Retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)

        if result and result.get("id"):
            imported += 1
            total_msgs += len(conv_messages)
            status = "created" if result.get("created") else "updated"
            print(f"  [{i+1}/{len(new_sessions)}] {status} #{result['id']}: {len(conv_messages)} msgs — {title[:50]}")
        else:
            failed += 1
            print(f"  [{i+1}/{len(new_sessions)}] FAILED: {sid} — {title[:50]}")

        time.sleep(RATE_DELAY)

    conn.close()
    print(f"\nDone! Imported: {imported}, Failed: {failed}, Total messages: {total_msgs}")


if __name__ == "__main__":
    main()
