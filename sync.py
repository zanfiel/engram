#!/usr/bin/env python3
"""Engram bidirectional sync between primary and failover nodes."""

import json
import sys
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone

NODES = {
    "windows": "http://127.0.0.1:4200",
    "bav-apps": "http://100.64.0.3:4200",
    "forge-box": "http://100.64.0.7:4200",
    "rocky": "http://100.64.0.2:4200",
}

BATCH_SIZE = 500


def api(url, method="GET", data=None):
    """Make HTTP request to Engram API."""
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"  ERROR {e.code}: {err[:200]}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def get_all_changes(base_url, since="1970-01-01T00:00:00"):
    """Fetch all changes from a node with pagination."""
    all_changes = []
    cursor = since
    while True:
        encoded_cursor = urllib.parse.quote(cursor)
        url = f"{base_url}/sync/changes?since={encoded_cursor}&limit={BATCH_SIZE}"
        result = api(url)
        if not result or not result.get("changes"):
            break
        batch = result["changes"]
        all_changes.extend(batch)
        if len(batch) < BATCH_SIZE:
            break
        # Use last item's updated_at as next cursor
        cursor = batch[-1].get("updated_at", cursor)
        if cursor == since:
            break
        since = cursor
    return all_changes


def push_changes(target_url, memories):
    """Push memories to target node in batches."""
    total_created = 0
    total_updated = 0
    total_skipped = 0
    for i in range(0, len(memories), BATCH_SIZE):
        batch = memories[i:i + BATCH_SIZE]
        result = api(f"{target_url}/sync/receive", method="POST", data={"memories": batch})
        if result and result.get("synced"):
            total_created += result.get("created", 0)
            total_updated += result.get("updated", 0)
            total_skipped += result.get("skipped", 0)
        else:
            print(f"  Failed to push batch {i//BATCH_SIZE + 1}")
    return total_created, total_updated, total_skipped


def sync_pair(source_name, source_url, target_name, target_url):
    """Sync changes from source to target."""
    print(f"\n  {source_name} -> {target_name}")
    changes = get_all_changes(source_url)
    if not changes:
        print(f"    No changes to sync")
        return
    print(f"    Fetched {len(changes)} changes from {source_name}")
    created, updated, skipped = push_changes(target_url, changes)
    print(f"    Result: {created} created, {updated} updated, {skipped} skipped")


def check_health(name, url):
    """Check if node is healthy and return memory count."""
    result = api(f"{url}/health")
    if result and result.get("status") == "ok":
        return result.get("memories", 0)
    return -1


def main():
    primary_name = "windows"
    primary_url = NODES[primary_name]

    print("=== Engram Sync ===")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")

    # Health check all nodes
    print("\nHealth check:")
    healthy = {}
    for name, url in NODES.items():
        count = check_health(name, url)
        status = f"{count} memories" if count >= 0 else "DOWN"
        marker = " (primary)" if name == primary_name else ""
        print(f"  {name}: {status}{marker}")
        if count >= 0:
            healthy[name] = url

    if primary_name not in healthy:
        print("\nERROR: Primary node is down. Aborting.")
        sys.exit(1)

    failovers = {k: v for k, v in healthy.items() if k != primary_name}
    if not failovers:
        print("\nNo failover nodes available.")
        sys.exit(0)

    # Sync primary -> each failover
    print("\n--- Primary -> Failovers ---")
    for name, url in failovers.items():
        sync_pair(primary_name, primary_url, name, url)

    # Sync each failover -> primary (pick up anything unique on failovers)
    print("\n--- Failovers -> Primary ---")
    for name, url in failovers.items():
        sync_pair(name, url, primary_name, primary_url)

    # Final health check
    print("\n--- Final State ---")
    for name, url in healthy.items():
        count = check_health(name, url)
        marker = " (primary)" if name == primary_name else ""
        print(f"  {name}: {count} memories{marker}")

    print("\nSync complete.")


if __name__ == "__main__":
    main()
