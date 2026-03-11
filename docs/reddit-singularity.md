GitHub: https://github.com/zanfiel/engram

Demo: https://demo.engram.lol/gui

Open source memory system for AI agents. Gives them persistent memory across sessions — they store what they learn and can recall it later by meaning.

Runs entirely local, embeddings included. No OpenAI key, no external vector database, just a single file. Docker compose to deploy.

Recently added FSRS-6 spaced repetition (same algorithm behind Anki) so memories decay realistically based on usage instead of a flat timer. Also moved to native vector search with libsql so it actually scales past a few thousand memories.

Some of the other stuff it does: auto-links related memories into a knowledge graph, detects contradictions between memories, time-travel queries to see what the agent knew at any point, conversation extraction, a WebGL graph visualization, multi-tenant with API keys. MCP server and TypeScript/Python SDKs for integration.

6,200 lines of TypeScript, MIT licensed. Been running it for my own agents for a while now.
