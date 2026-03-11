---
title: I replaced exponential decay with spaced repetition in my AI memory server
published: true
tags: ai, opensource, typescript, machinelearning
---

I've been building [Engram](https://github.com/zanfiel/engram), an open source memory server for AI agents. The idea is simple — agents store what they learn, and recall it later by meaning. Embeddings run locally, single file database, self-hosted.

One of the first things I implemented was memory decay. Memories shouldn't all live forever — some stuff is only relevant for a day, other things matter for months. So I added a basic exponential decay with a 30-day half-life. Memories faded over time, access bumped them up a bit, done.

It worked okay. But it had a stupid problem.

## The problem with flat decay

Every memory decayed on the same curve. Something an agent accessed 50 times a day got roughly the same treatment as something stored once and never touched again. I added boosts for access count and recency but it was band-aids on a fundamentally wrong model.

The other issue — exponential decay drops off too fast initially and too slow later. A memory at day 2 has already lost a noticeable chunk of its score, but then it takes forever to actually hit zero. Real memory doesn't work that way.

I kept running into situations where my agents would recall stale, barely-relevant stuff because it happened to be stored recently, while genuinely useful memories that were a few weeks old had already decayed below the threshold.

## FSRS-6

I found [FSRS](https://github.com/open-spaced-repetition/fsrs4anki) (Free Spaced Repetition Scheduler) while looking at how Anki handles this. It's a 21-parameter model trained on hundreds of millions of real review logs. The forgetting curve is power-law instead of exponential, which aligns with what the research actually says about memory.

The formula:

```
R = (1 + factor × t/S)^(-w₂₀)
```

Where `S` is stability (days until 90% recall probability) and `w₂₀` is a trained decay parameter. The key difference — power-law curves have a sharp initial drop followed by a long tail. Memories that aren't reinforced fade quickly at first, but the ones that survive the initial period stick around much longer. Exponential just gradually slides everything toward zero at the same rate.

## Implicit reviews

In Anki, you explicitly rate cards (Again/Hard/Good/Easy). For an AI memory server nobody's manually grading anything, so I made it implicit — every time a memory gets accessed through search or recall, that's a "Good" review. Archiving or forgetting a memory is an "Again."

Frequently accessed memories build stability. A memory that gets hit every day might have stability measured in months — it takes a long time to decay even if access stops. Something stored once and never looked at drops off within days.

No config, no manual intervention. It just tracks usage patterns.

## Dual-strength model

While I was at it I also implemented the Bjork & Bjork (1992) dual-strength model. Each memory now tracks two independent values:

**Storage strength** goes up every time a memory is accessed and never decays. This represents how deeply encoded something is.

**Retrieval strength** decays via power law and resets on access. This represents how accessible it is right now.

The combined retention score is `0.7 × retrieval + 0.3 × (storage/10)`.

Why this matters: a memory can have high storage strength but low retrieval strength. It's deeply known but temporarily hard to find. When the agent accesses it again, retrieval strength jumps back and the high storage strength means it decays slower next time. This is the "spacing effect" — the same mechanism that makes spaced repetition work for flashcards.

## The actual migration

Porting the algorithm from the [reference implementation](https://github.com/open-spaced-repetition) to TypeScript wasn't bad. Added new columns to the schema:

```
fsrs_stability, fsrs_difficulty, fsrs_storage_strength,
fsrs_retrieval_strength, fsrs_learning_state, fsrs_reps,
fsrs_lapses, fsrs_last_review_at
```

Kept the old `decay_score` column populated from FSRS retrievability for backward compatibility. Backfilled existing memories with a `/fsrs/init` endpoint.

The `calculateDecayScore()` function went from this:

```typescript
// Old — flat exponential
Math.pow(0.5, daysSinceCreation / effectiveHalfLife)
```

To using the FSRS retrievability formula, which factors in the memory's actual stability built from its access history.

## Results

The difference in recall quality was noticeable pretty much immediately. Memories that my agents actively use stay high in the rankings. Old context that was only relevant to a specific task fades out properly instead of cluttering results for weeks. The scoring just makes more sense now.

I also moved from SQLite to libsql around the same time for native vector search (FLOAT32 columns with ANN indexing instead of cosine similarity loops in JS), which helped with search speed, but the FSRS change had way more impact on the actual quality of what comes back from recall.

## Try it

Engram is open source, MIT licensed. Single file, `docker compose up`.

- GitHub: [github.com/zanfiel/engram](https://github.com/zanfiel/engram)
- Live demo: [demo.engram.lol/gui](https://demo.engram.lol/gui)

If you're building agents and they keep forgetting everything between sessions, this might be useful. Happy to answer questions about the FSRS implementation or anything else.
