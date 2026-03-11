"""
Engram SDK — Persistent memory for AI agents

Usage:
    from engram import Engram

    client = Engram("http://localhost:4200", api_key="eg_your_key")

    # Store a memory
    client.store("User prefers dark mode", category="decision")

    # Search
    results = client.search("user preferences")

    # Recall context for an agent
    context = client.recall("What does the user like?")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import httpx

Category = Literal["task", "discovery", "decision", "state", "issue", "general"]


class EngramError(Exception):
    """Error from the Engram API."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


@dataclass
class Memory:
    id: int
    content: str
    category: str
    source: str
    importance: int
    created_at: str
    version: int = 1
    is_latest: bool = True
    is_static: bool = False
    is_forgotten: bool = False
    is_archived: bool = False
    is_inference: bool = False
    session_id: Optional[str] = None
    parent_memory_id: Optional[int] = None
    root_memory_id: Optional[int] = None
    source_count: int = 1
    forget_after: Optional[str] = None
    score: Optional[float] = None  # Search results only

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class Engram:
    """Engram client for persistent AI agent memory."""

    def __init__(
        self,
        url: str = "http://localhost:4200",
        api_key: Optional[str] = None,
        space: Optional[str] = None,
        source: str = "python-sdk",
        timeout: float = 30.0,
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.space = space
        self.source = source
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.space:
            h["X-Space"] = self.space
        return h

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._client.request(
            method, f"{self.url}{path}", headers=self._headers(), **kwargs
        )
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error", resp.text)
            except Exception:
                err = resp.text
            raise EngramError(str(err), resp.status_code)
        return resp.json()

    # ── Core Memory Operations ──────────────────────────────────────────

    def store(
        self,
        content: str,
        category: Category = "general",
        importance: int = 5,
        source: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """Store a new memory. Returns {stored, id, created_at, importance, linked, embedded}."""
        return self._request(
            "POST",
            "/store",
            json={
                "content": content,
                "category": category,
                "importance": importance,
                "source": source or self.source,
                "session_id": session_id,
            },
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[Memory]:
        """Semantic search across memories."""
        data = self._request(
            "POST", "/search", json={"query": query, "limit": limit, "threshold": threshold}
        )
        return [Memory.from_dict(r) for r in data.get("results", [])]

    def recall(self, query: str, limit: int = 25) -> dict:
        """Contextual recall — optimized for agent context injection.

        Returns {memories: [...], breakdown: {static_facts, semantic_matches, important, recent}}.
        """
        return self._request("POST", "/recall", json={"query": query, "limit": limit})

    def list(
        self, limit: int = 20, category: Optional[Category] = None
    ) -> list[Memory]:
        """List recent memories."""
        params: dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        data = self._request("GET", "/list", params=params)
        return [Memory.from_dict(r) for r in data.get("results", [])]

    def profile(self, summary: bool = False) -> dict:
        """Get user profile (static facts + recent activity)."""
        params = {"summary": "true"} if summary else {}
        return self._request("GET", "/profile", params=params)

    def graph(self) -> dict:
        """Get full memory graph {memories, links}."""
        return self._request("GET", "/graph")

    # ── Memory Management ───────────────────────────────────────────────

    def update(self, id: int, content: str, category: Optional[Category] = None) -> dict:
        """Update a memory (creates a new version)."""
        body: dict[str, Any] = {"content": content}
        if category:
            body["category"] = category
        return self._request("POST", f"/memory/{id}/update", json=body)

    def forget(self, id: int) -> dict:
        """Forget a memory (soft delete)."""
        return self._request("POST", f"/memory/{id}/forget")

    def archive(self, id: int) -> dict:
        """Archive a memory."""
        return self._request("POST", f"/memory/{id}/archive")

    def unarchive(self, id: int) -> dict:
        """Unarchive a memory."""
        return self._request("POST", f"/memory/{id}/unarchive")

    def delete(self, id: int) -> dict:
        """Permanently delete a memory."""
        return self._request("DELETE", f"/memory/{id}")

    # ── Data Operations ─────────────────────────────────────────────────

    def export(self, format: str = "json") -> dict:
        """Export all memories and links."""
        return self._request("GET", f"/export?format={format}")

    def import_memories(
        self, memories: list[dict[str, Any]]
    ) -> dict:
        """Bulk import memories. Returns {imported, failed, total}."""
        return self._request("POST", "/import", json={"memories": memories})

    # ── Spaces ──────────────────────────────────────────────────────────

    def list_spaces(self) -> list[dict]:
        """List memory spaces."""
        return self._request("GET", "/spaces").get("spaces", [])

    def create_space(self, name: str, description: Optional[str] = None) -> dict:
        """Create a memory space."""
        return self._request("POST", "/spaces", json={"name": name, "description": description})

    def delete_space(self, id: int) -> dict:
        """Delete a memory space."""
        return self._request("DELETE", f"/spaces/{id}")

    def use_space(self, name: str) -> "Engram":
        """Switch to a different space. Returns self for chaining."""
        self.space = name
        return self

    # ── API Keys ────────────────────────────────────────────────────────

    def list_keys(self) -> list[dict]:
        """List API keys (values not returned)."""
        return self._request("GET", "/keys").get("keys", [])

    def create_key(
        self,
        name: str = "default",
        scopes: str = "read,write",
        rate_limit: int = 120,
    ) -> dict:
        """Create a new API key. Key value is returned only once."""
        return self._request(
            "POST", "/keys", json={"name": name, "scopes": scopes, "rate_limit": rate_limit}
        )

    def revoke_key(self, id: int) -> dict:
        """Revoke an API key."""
        return self._request("DELETE", f"/keys/{id}")

    # ── Maintenance ─────────────────────────────────────────────────────

    def duplicates(self, threshold: float = 0.92, limit: int = 20) -> dict:
        """Find duplicate memory clusters."""
        return self._request("GET", f"/duplicates?threshold={threshold}&limit={limit}")

    def deduplicate(self, threshold: float = 0.92, dry_run: bool = False) -> dict:
        """Auto-deduplicate memories."""
        return self._request(
            "POST", "/deduplicate", json={"threshold": threshold, "dry_run": dry_run}
        )

    def sweep(self) -> dict:
        """Trigger forget sweep."""
        return self._request("POST", "/sweep")

    def backfill(self) -> dict:
        """Backfill missing embeddings."""
        return self._request("POST", "/backfill")

    # ── Conversations ───────────────────────────────────────────────────

    def create_conversation(self, agent: str, session_id: Optional[str] = None, model: Optional[str] = None, title: Optional[str] = None) -> dict:
        """Create a conversation."""
        return self._request("POST", "/conversations", json={"agent": agent, "session_id": session_id, "model": model, "title": title})

    def list_conversations(self, limit: int = 20, offset: int = 0, agent: Optional[str] = None) -> dict:
        """List conversations."""
        params = f"?limit={limit}&offset={offset}"
        if agent: params += f"&agent={agent}"
        return self._request("GET", f"/conversations{params}")

    def conversations_bulk(self, agent: str, messages: list[dict], session_id: Optional[str] = None, model: Optional[str] = None) -> dict:
        """Bulk insert a conversation with messages."""
        return self._request("POST", "/conversations/bulk", json={"agent": agent, "messages": messages, "session_id": session_id, "model": model})

    def search_messages(self, query: str, limit: int = 20) -> dict:
        """Search conversation messages."""
        return self._request("POST", "/messages/search", json={"query": query, "limit": limit})

    # ── Episodes ────────────────────────────────────────────────────────

    def create_episode(self, name: str, memory_ids: list[int]) -> dict:
        """Create an episode from memory IDs."""
        return self._request("POST", "/episodes", json={"name": name, "memory_ids": memory_ids})

    def list_episodes(self, limit: int = 20) -> dict:
        """List episodes."""
        return self._request("GET", f"/episodes?limit={limit}")

    # ── Tags ────────────────────────────────────────────────────────────

    def set_tags(self, id: int, tags: list[str]) -> dict:
        """Set tags on a memory."""
        return self._request("PUT", f"/memory/{id}/tags", json={"tags": tags})

    def list_tags(self) -> dict:
        """List all tags."""
        return self._request("GET", "/tags")

    def search_by_tag(self, tag: str) -> dict:
        """Search memories by tag."""
        return self._request("POST", "/tags/search", json={"tag": tag})

    # ── FSRS & Decay ───────────────────────────────────────────────────

    def decay_scores(self, limit: int = 20) -> dict:
        """Get decay scores."""
        return self._request("GET", f"/decay/scores?limit={limit}")

    def decay_refresh(self) -> dict:
        """Refresh decay calculations."""
        return self._request("POST", "/decay/refresh")

    def fsrs_review(self, id: int, grade: int) -> dict:
        """Record an FSRS review (grade 1-4)."""
        return self._request("POST", "/fsrs/review", json={"id": id, "grade": grade})

    def fsrs_state(self, id: int) -> dict:
        """Get FSRS state for a memory."""
        return self._request("GET", f"/fsrs/state?id={id}")

    # ── Pack & Context ──────────────────────────────────────────────────

    def pack(self, query: str, max_tokens: int = 4000) -> dict:
        """Pack memories into a token-budgeted context string."""
        return self._request("POST", "/pack", json={"query": query, "max_tokens": max_tokens})

    def prompt(self, template: str, query: str) -> dict:
        """Get a formatted prompt template."""
        return self._request("GET", f"/prompt?template={template}&query={query}")

    def context(self, query: str, max_tokens: int = 4000) -> dict:
        """Get smart context with blocks and token budget."""
        return self._request("POST", "/context", json={"query": query, "max_tokens": max_tokens})

    # ── Time Travel & Contradictions ────────────────────────────────────

    def time_travel(self, as_of: str) -> dict:
        """View memory state at a point in time."""
        return self._request("POST", "/timetravel", json={"as_of": as_of})

    def contradictions(self, limit: int = 20) -> dict:
        """List detected contradictions."""
        return self._request("GET", f"/contradictions?limit={limit}")

    # ── Webhooks ────────────────────────────────────────────────────────

    def create_webhook(self, url: str, events: list[str]) -> dict:
        """Register a webhook."""
        return self._request("POST", "/webhooks", json={"url": url, "events": events})

    def list_webhooks(self) -> dict:
        """List webhooks."""
        return self._request("GET", "/webhooks")

    # ── Sync ────────────────────────────────────────────────────────────

    def sync_changes(self, since: str) -> dict:
        """Get changes since a timestamp for syncing."""
        return self._request("GET", f"/sync/changes?since={since}")

    # ── Inbox & Audit ───────────────────────────────────────────────────

    def inbox(self, limit: int = 20) -> dict:
        """Get pending inbox items."""
        return self._request("GET", f"/inbox?limit={limit}")

    def audit(self, limit: int = 50) -> dict:
        """Get audit log."""
        return self._request("GET", f"/audit?limit={limit}")

    # ── Entities & Projects ─────────────────────────────────────────────

    def create_entity(self, name: str, type: str, description: Optional[str] = None) -> dict:
        """Create an entity."""
        return self._request("POST", "/entities", json={"name": name, "type": type, "description": description})

    def list_entities(self) -> dict:
        """List entities."""
        return self._request("GET", "/entities")

    def create_project(self, name: str, description: Optional[str] = None) -> dict:
        """Create a project."""
        return self._request("POST", "/projects", json={"name": name, "description": description})

    def list_projects(self) -> dict:
        """List projects."""
        return self._request("GET", "/projects")

    # ── Digests ─────────────────────────────────────────────────────────

    def create_digest(self, name: str, cron: str, webhook_url: str) -> dict:
        """Create a scheduled digest."""
        return self._request("POST", "/digests", json={"name": name, "cron": cron, "webhook_url": webhook_url})

    def list_digests(self) -> dict:
        """List digest schedules."""
        return self._request("GET", "/digests")

    # ── Users ───────────────────────────────────────────────────────────

    def create_user(self, username: str) -> dict:
        """Create a user."""
        return self._request("POST", "/users", json={"username": username})

    def list_users(self) -> dict:
        """List users."""
        return self._request("GET", "/users")

    # ── Version History & Links ─────────────────────────────────────────

    def versions(self, root_id: int) -> dict:
        """Get version chain for a root memory."""
        return self._request("GET", f"/versions/{root_id}")

    def links(self, id: int) -> dict:
        """Get links for a memory."""
        return self._request("GET", f"/links/{id}")

    # ── URL Ingest ──────────────────────────────────────────────────────

    def ingest(self, url: str) -> dict:
        """Ingest content from a URL."""
        return self._request("POST", "/ingest", json={"url": url})

    # ── System ──────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Health check."""
        return self._request("GET", "/health")

    def stats(self) -> dict:
        """Detailed statistics."""
        return self._request("GET", "/stats")

    # ── Context Manager ─────────────────────────────────────────────────

    def __enter__(self) -> "Engram":
        return self

    def __exit__(self, *_: Any) -> None:
        self._client.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
