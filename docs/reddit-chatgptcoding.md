GitHub: https://github.com/zanfiel/engram

Demo: https://demo.engram.lol/gui

Got tired of my coding agents forgetting everything between sessions. Built Engram to fix it — it's a memory server that agents can store to and recall from. Runs locally, single file database, no API keys needed for embeddings.

The part that actually made the biggest difference for me was adding FSRS-6 (the spaced repetition algorithm from Anki). Memories that my agents keep accessing build up stability and stick around. Stuff that was only relevant once fades out on its own. Before this it was just a flat decay timer which wasn't great.

It also does auto-linking between related memories so you end up with a knowledge graph, contradiction detection if memories conflict, versioning so you don't lose history, and a context builder that packs relevant memories into a token budget for recall.

Has an MCP server so you can wire it into whatever agent setup you're using. TypeScript and Python SDKs too.

Self-hosted, MIT, `docker compose up` to run it.
