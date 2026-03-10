#!/usr/bin/env python3
"""Bulk import all OpenCode sessions into Engram conversations."""

import json
import subprocess
import sys
import time
import urllib.request

MEGAMIND_URL = "http://127.0.0.1:4200"
OPENCODE_BIN = "/usr/local/bin/opencode"
AGENT_NAME = "opencode-rocky"


def api_post(endpoint, data):
    """POST JSON to Engram."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{MEGAMIND_URL}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        print(f"  HTTP {e.code}: {err_body[:200]}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def api_get(endpoint):
    """GET from Engram."""
    req = urllib.request.Request(f"{MEGAMIND_URL}{endpoint}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  GET Error: {e}")
        return None


def get_all_sessions():
    """Get list of all OpenCode sessions."""
    result = subprocess.run(
        [OPENCODE_BIN, "session", "list", "--format", "json", "-n", "200"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"Error listing sessions: {result.stderr}")
        return []
    return json.loads(result.stdout)


def export_session(session_id):
    """Export a single session, returns parsed dict or None."""
    import tempfile
    import os
    # Write to a temp file to avoid pipe buffer truncation
    tmpfile = f"/tmp/oc_export_{session_id}.json"
    try:
        result = subprocess.run(
            f'{OPENCODE_BIN} export {session_id} > "{tmpfile}" 2>/dev/null',
            shell=True, timeout=120
        )
        if not os.path.exists(tmpfile) or os.path.getsize(tmpfile) == 0:
            return None
        with open(tmpfile, "r") as f:
            raw = f.read()
        # Skip the "Exporting session: ..." prefix line
        idx = raw.index("{")
        return json.loads(raw[idx:])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  Parse error: {e}")
        return None
    except Exception as e:
        print(f"  Export error: {e}")
        return None
    finally:
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)


def extract_messages(export_data):
    """Convert OpenCode export messages to Engram format (role + content)."""
    messages = []
    for msg in export_data.get("messages", []):
        role = msg["info"]["role"]
        # Collect all text parts
        texts = []
        for part in msg.get("parts", []):
            if part.get("type") == "text" and part.get("text"):
                text = part["text"].strip()
                if text and text != "NO_REPLY":
                    texts.append(text)
            elif part.get("type") == "tool-invocation":
                tool = part.get("toolName", "unknown")
                args = part.get("args", {})
                # Summarize tool calls briefly
                if tool == "bash":
                    cmd = args.get("command", "")[:200]
                    texts.append(f"[Tool: bash] {cmd}")
                elif tool in ("read", "write", "edit", "glob", "grep"):
                    path = args.get("filePath", args.get("path", args.get("pattern", "")))
                    texts.append(f"[Tool: {tool}] {path}")
                elif tool in ("memory_store", "memory_search", "memory_list"):
                    texts.append(f"[Tool: {tool}]")
                elif tool == "webfetch":
                    url = args.get("url", "")
                    texts.append(f"[Tool: webfetch] {url}")
                elif tool == "task":
                    desc = args.get("description", "")
                    texts.append(f"[Tool: task] {desc}")
                else:
                    texts.append(f"[Tool: {tool}]")
            elif part.get("type") == "tool-result":
                # Skip tool results to keep size manageable
                pass

        if texts:
            content = "\n".join(texts)
            # Truncate very long messages (Engram has limits)
            if len(content) > 50000:
                content = content[:50000] + "\n... [truncated]"
            messages.append({"role": role, "content": content})
    return messages


def get_existing_sessions():
    """Get set of session_ids already in Engram."""
    result = api_get("/conversations?limit=500")
    if not result:
        return set()
    convs = result.get("results", result) if isinstance(result, dict) else result
    existing = set()
    for conv in convs:
        if isinstance(conv, dict):
            sid = conv.get("session_id", "")
            if sid:
                existing.add(sid)
    return existing


def main():
    # Check Engram health
    health = api_get("/health")
    if not health:
        print("ERROR: Engram not responding at", MEGAMIND_URL)
        sys.exit(1)
    print(f"Engram OK: {health['memories']} memories, {health['conversations']} conversations, {health['messages']} messages")

    # Get existing to avoid duplicates
    existing = get_existing_sessions()
    print(f"Already imported: {len(existing)} sessions")

    # Get all sessions
    sessions = get_all_sessions()
    print(f"Found {len(sessions)} OpenCode sessions")

    imported = 0
    skipped = 0
    failed = 0

    for i, sess in enumerate(sessions):
        sid = sess["id"]
        title = sess.get("title", "Untitled")

        if sid in existing:
            print(f"[{i+1}/{len(sessions)}] SKIP (exists): {title[:60]}")
            skipped += 1
            continue

        print(f"[{i+1}/{len(sessions)}] Exporting: {title[:60]}...", end="", flush=True)

        export = export_session(sid)
        if not export:
            print(" FAILED (export)")
            failed += 1
            continue

        messages = extract_messages(export)
        if not messages:
            print(f" SKIP (no messages)")
            skipped += 1
            continue

        # Build conversation payload
        created_ts = sess.get("created", 0)
        # Convert ms timestamp to ISO string if available
        payload = {
            "agent": AGENT_NAME,
            "session_id": sid,
            "title": title,
            "messages": messages,
        }

        result = api_post("/conversations/bulk", payload)
        if result:
            msg_count = len(messages)
            print(f" OK ({msg_count} msgs)")
            imported += 1
        else:
            print(" FAILED (API)")
            failed += 1

        # Small delay to not hammer the API
        time.sleep(0.1)

    print(f"\nDone! Imported: {imported}, Skipped: {skipped}, Failed: {failed}")

    # Final stats
    health = api_get("/health")
    if health:
        print(f"Engram now: {health['conversations']} conversations, {health['messages']} messages")


if __name__ == "__main__":
    main()
