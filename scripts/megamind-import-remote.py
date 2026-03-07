#!/usr/bin/env python3
"""Import OpenCode sessions from remote servers into MegaMind."""

import json
import subprocess
import sys
import time
import urllib.request

MEGAMIND_URL = "http://127.0.0.1:4200"

REMOTE_SERVERS = [
    {
        "name": "bav-apps",
        "agent": "opencode-bav-apps",
        "ssh": "ssh -i /home/zan/.ssh/ZanSSH -o ConnectTimeout=10 -o StrictHostKeyChecking=no zan@15.204.88.223",
        "opencode": "/usr/local/bin/opencode",
    },
]


def api_post(endpoint, data):
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
    req = urllib.request.Request(f"{MEGAMIND_URL}{endpoint}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  GET Error: {e}")
        return None


def get_existing_sessions(agent):
    result = api_get(f"/conversations?limit=500&agent={agent}")
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


def extract_messages(export_data):
    messages = []
    for msg in export_data.get("messages", []):
        role = msg["info"]["role"]
        texts = []
        for part in msg.get("parts", []):
            if part.get("type") == "text" and part.get("text"):
                text = part["text"].strip()
                if text and text != "NO_REPLY":
                    texts.append(text)
            elif part.get("type") == "tool-invocation":
                tool = part.get("toolName", "unknown")
                args = part.get("args", {})
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
        if texts:
            content = "\n".join(texts)
            if len(content) > 50000:
                content = content[:50000] + "\n... [truncated]"
            messages.append({"role": role, "content": content})
    return messages


def import_from_server(server):
    name = server["name"]
    agent = server["agent"]
    ssh_cmd = server["ssh"]
    oc_bin = server["opencode"]

    print(f"\n=== {name} ({agent}) ===")

    # Get existing to skip duplicates
    existing = get_existing_sessions(agent)
    print(f"Already in MegaMind: {len(existing)} sessions")

    # List sessions on remote
    list_cmd = f'{ssh_cmd} "{oc_bin} session list --format json -n 200 > /tmp/oc_sessions.json 2>/dev/null; cat /tmp/oc_sessions.json"'
    result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"  ERROR listing sessions: {result.stderr[:200]}")
        return 0

    try:
        sessions = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"  ERROR parsing session list: {e}")
        return 0

    print(f"Found {len(sessions)} sessions on {name}")

    imported = 0
    for i, sess in enumerate(sessions):
        sid = sess["id"]
        title = sess.get("title", "Untitled")

        if sid in existing:
            print(f"  [{i+1}/{len(sessions)}] SKIP: {title[:55]}")
            continue

        print(f"  [{i+1}/{len(sessions)}] Exporting: {title[:55]}...", end="", flush=True)

        # Export session on remote, cat it back
        export_cmd = f'{ssh_cmd} "{oc_bin} export {sid} > /tmp/oc_export_{sid}.json 2>/dev/null; cat /tmp/oc_export_{sid}.json; rm -f /tmp/oc_export_{sid}.json"'
        result = subprocess.run(export_cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not result.stdout.strip():
            print(" FAILED (export)")
            continue

        raw = result.stdout
        try:
            idx = raw.index("{")
            export_data = json.loads(raw[idx:])
        except (ValueError, json.JSONDecodeError) as e:
            print(f" FAILED (parse: {e})")
            continue

        messages = extract_messages(export_data)
        if not messages:
            print(" SKIP (empty)")
            continue

        payload = {
            "agent": agent,
            "session_id": sid,
            "title": title,
            "messages": messages,
        }

        resp = api_post("/conversations/bulk", payload)
        if resp:
            print(f" OK ({len(messages)} msgs)")
            imported += 1
        else:
            print(" FAILED (API)")

        time.sleep(0.1)

    return imported


def main():
    health = api_get("/health")
    if not health:
        print("ERROR: MegaMind not responding")
        sys.exit(1)
    print(f"MegaMind: {health['conversations']} conversations, {health['messages']} messages")

    total = 0
    for server in REMOTE_SERVERS:
        total += import_from_server(server)

    print(f"\nTotal imported: {total}")
    health = api_get("/health")
    if health:
        print(f"MegaMind now: {health['conversations']} conversations, {health['messages']} messages")


if __name__ == "__main__":
    main()
