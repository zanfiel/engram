// ============================================================================
// FSRS-6 ENGINE — Spaced repetition for AI memory
// ============================================================================

import { FSRS_DEFAULT_RETENTION } from "../config/index.ts";

// ============================================================================

// 21 default weights trained on millions of Anki reviews
const FSRS6_WEIGHTS: number[] = [
  0.212,   // w0:  Initial stability for Again
  1.2931,  // w1:  Initial stability for Hard
  2.3065,  // w2:  Initial stability for Good
  8.2956,  // w3:  Initial stability for Easy
  6.4133,  // w4:  Initial difficulty base
  0.8334,  // w5:  Initial difficulty grade modifier
  3.0194,  // w6:  Difficulty delta
  0.001,   // w7:  Difficulty mean reversion
  1.8722,  // w8:  Stability increase base
  0.1666,  // w9:  Stability saturation
  0.796,   // w10: Retrievability influence on stability
  1.4835,  // w11: Forget stability base
  0.0614,  // w12: Forget difficulty influence
  0.2629,  // w13: Forget stability influence
  1.6483,  // w14: Forget retrievability influence
  0.6014,  // w15: Hard penalty
  1.8729,  // w16: Easy bonus
  0.5425,  // w17: Same-day review base (FSRS-6)
  0.0912,  // w18: Same-day review grade modifier (FSRS-6)
  0.0658,  // w19: Same-day review stability influence (FSRS-6)
  0.1542,  // w20: Forgetting curve decay (FSRS-6, personalizable)
];

const FSRS_MIN_STABILITY = 0.1;
const FSRS_MAX_STABILITY = 36500; // 100 years
const FSRS_MIN_DIFFICULTY = 1.0;
const FSRS_MAX_DIFFICULTY = 10.0;
const FSRS_MAX_STORAGE = 10.0;

export const FSRSRating = { Again: 1, Hard: 2, Good: 3, Easy: 4 } as const;
export type FSRSRating = typeof FSRSRating[keyof typeof FSRSRating];
const FSRSState = { New: 0, Learning: 1, Review: 2, Relearning: 3 } as const;

/** Forgetting factor: 0.9^(-1/w20) - 1 */
function fsrsForgettingFactor(w20: number = FSRS6_WEIGHTS[20]): number {
  return Math.pow(0.9, -1.0 / w20) - 1.0;
}

/**
 * FSRS-6 Retrievability — probability of recall at time t
 * R = (1 + factor * t / S)^(-w20)  — power law, more accurate than exponential
 */
export function fsrsRetrievability(stability: number, elapsedDays: number, w20: number = FSRS6_WEIGHTS[20]): number {
  if (stability <= 0) return 0;
  if (elapsedDays <= 0) return 1;
  const factor = fsrsForgettingFactor(w20);
  return Math.max(0, Math.min(1, Math.pow(1 + factor * elapsedDays / stability, -w20)));
}

/** Initial difficulty: D0(G) = w4 - e^(w5*(G-1)) + 1 */
function fsrsInitialDifficulty(grade: FSRSRating): number {
  const d = FSRS6_WEIGHTS[4] - Math.exp(FSRS6_WEIGHTS[5] * (grade - 1)) + 1;
  return Math.max(FSRS_MIN_DIFFICULTY, Math.min(FSRS_MAX_DIFFICULTY, d));
}

/** Initial stability: S0(G) = w[G-1] */
function fsrsInitialStability(grade: FSRSRating): number {
  return Math.max(FSRS_MIN_STABILITY, FSRS6_WEIGHTS[grade - 1]);
}

/** Next difficulty with mean reversion: D' = w7*D0(4) + (1-w7)*(D + delta*((10-D)/9)) */
function fsrsNextDifficulty(currentD: number, grade: FSRSRating): number {
  const d0 = fsrsInitialDifficulty(FSRSRating.Easy);
  const delta = -FSRS6_WEIGHTS[6] * (grade - 3);
  const meanReversionScale = (10 - currentD) / 9;
  const newD = currentD + delta * meanReversionScale;
  const finalD = FSRS6_WEIGHTS[7] * d0 + (1 - FSRS6_WEIGHTS[7]) * newD;
  return Math.max(FSRS_MIN_DIFFICULTY, Math.min(FSRS_MAX_DIFFICULTY, finalD));
}

/** Stability after successful recall: S' = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) * HP * EB + 1) */
function fsrsRecallStability(S: number, D: number, R: number, grade: FSRSRating): number {
  if (grade === FSRSRating.Again) return fsrsForgetStability(D, S, R);
  const hardPenalty = grade === FSRSRating.Hard ? FSRS6_WEIGHTS[15] : 1;
  const easyBonus = grade === FSRSRating.Easy ? FSRS6_WEIGHTS[16] : 1;
  const factor = Math.exp(FSRS6_WEIGHTS[8]) * (11 - D)
    * Math.pow(S, -FSRS6_WEIGHTS[9])
    * (Math.exp(FSRS6_WEIGHTS[10] * (1 - R)) - 1)
    * hardPenalty * easyBonus + 1;
  return Math.max(FSRS_MIN_STABILITY, Math.min(FSRS_MAX_STABILITY, S * factor));
}

/** Stability after lapse: S'f = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R)) */
function fsrsForgetStability(D: number, S: number, R: number): number {
  const newS = FSRS6_WEIGHTS[11] * Math.pow(D, -FSRS6_WEIGHTS[12])
    * (Math.pow(S + 1, FSRS6_WEIGHTS[13]) - 1)
    * Math.exp(FSRS6_WEIGHTS[14] * (1 - R));
  return Math.max(FSRS_MIN_STABILITY, Math.min(Math.min(newS, S), FSRS_MAX_STABILITY));
}

/** Same-day review stability (FSRS-6): S'(S,G) = S * e^(w17*(G-3+w18)) * S^(-w19) */
function fsrsSameDayStability(S: number, grade: FSRSRating): number {
  const newS = S * Math.exp(FSRS6_WEIGHTS[17] * (grade - 3 + FSRS6_WEIGHTS[18]))
    * Math.pow(S, -FSRS6_WEIGHTS[19]);
  return Math.max(FSRS_MIN_STABILITY, Math.min(FSRS_MAX_STABILITY, newS));
}

/** Optimal review interval: t = S/factor * (R^(-1/w20) - 1) */
export function fsrsNextInterval(stability: number, desiredR: number = FSRS_DEFAULT_RETENTION): number {
  if (stability <= 0 || desiredR >= 1 || desiredR <= 0) return 0;
  const factor = fsrsForgettingFactor();
  return Math.max(0, Math.round(stability / factor * (Math.pow(desiredR, -1 / FSRS6_WEIGHTS[20]) - 1)));
}

// --- Dual-Strength Model (Bjork & Bjork 1992) ---

interface DualStrength { storage: number; retrieval: number; }

function dualStrengthRetention(ds: DualStrength): number {
  return (ds.retrieval * 0.7) + ((ds.storage / FSRS_MAX_STORAGE) * 0.3);
}

function dualStrengthOnRecall(ds: DualStrength): DualStrength {
  return { storage: Math.min(ds.storage + 0.1, FSRS_MAX_STORAGE), retrieval: 1.0 };
}

function dualStrengthOnLapse(ds: DualStrength): DualStrength {
  return { storage: Math.min(ds.storage + 0.3, FSRS_MAX_STORAGE), retrieval: 1.0 };
}

function dualStrengthDecay(ds: DualStrength, elapsedDays: number, stability: number): DualStrength {
  if (elapsedDays <= 0 || stability <= 0) return ds;
  const retrieval = Math.max(0, Math.min(1,
    Math.pow(1 + elapsedDays / (9.0 * stability), -1.0 / 0.5)
  ));
  return { storage: ds.storage, retrieval };
}

// --- Review processor: maps memory access events to FSRS state updates ---

export interface FSRSMemoryState {
  stability: number;
  difficulty: number;
  storage_strength: number;
  retrieval_strength: number;
  learning_state: number; // FSRSState enum
  reps: number;
  lapses: number;
  last_review_at: string;
}

export function fsrsProcessReview(
  state: FSRSMemoryState | null,
  grade: FSRSRating,
  elapsedDays: number
): FSRSMemoryState {
  const now = new Date().toISOString().replace("T", " ").slice(0, 19);

  // First review (new memory)
  if (!state || state.learning_state === FSRSState.New) {
    const d = fsrsInitialDifficulty(grade);
    const s = fsrsInitialStability(grade);
    return {
      stability: s, difficulty: d,
      storage_strength: 1.0, retrieval_strength: 1.0,
      learning_state: grade <= FSRSRating.Hard ? FSRSState.Learning : FSRSState.Review,
      reps: 1, lapses: grade === FSRSRating.Again ? 1 : 0,
      last_review_at: now,
    };
  }

  const R = fsrsRetrievability(state.stability, elapsedDays);
  const isSameDay = elapsedDays < 1;
  let ds: DualStrength = { storage: state.storage_strength, retrieval: state.retrieval_strength };

  let newS: number, newD: number, newState: number, newLapses: number;

  if (isSameDay) {
    // Same-day review — FSRS-6 special handling
    newS = fsrsSameDayStability(state.stability, grade);
    newD = fsrsNextDifficulty(state.difficulty, grade);
    newState = state.learning_state;
    newLapses = state.lapses;
    ds = dualStrengthOnRecall(ds);
  } else if (grade === FSRSRating.Again) {
    // Lapse — forgot it
    newS = fsrsForgetStability(state.difficulty, state.stability, R);
    newD = fsrsNextDifficulty(state.difficulty, FSRSRating.Again);
    newState = FSRSState.Relearning;
    newLapses = state.lapses + 1;
    ds = dualStrengthOnLapse(ds);
  } else {
    // Successful recall
    newS = fsrsRecallStability(state.stability, state.difficulty, R, grade);
    newD = fsrsNextDifficulty(state.difficulty, grade);
    newState = FSRSState.Review;
    newLapses = state.lapses;
    ds = dualStrengthOnRecall(ds);
  }

  return {
    stability: Math.round(newS * 1000) / 1000,
    difficulty: Math.round(newD * 1000) / 1000,
    storage_strength: Math.round(ds.storage * 1000) / 1000,
    retrieval_strength: Math.round(ds.retrieval * 1000) / 1000,
    learning_state: newState,
    reps: state.reps + 1,
    lapses: newLapses,
    last_review_at: now,
  };
}

// --- Backward-compatible decay score using FSRS retrievability ---

export function calculateDecayScore(
  importance: number,
  createdAt: string,
  accessCount: number = 0,
  lastAccessedAt: string | null = null,
  isStatic: boolean = false,
  sourceCount: number = 1,
  stability?: number
): number {
  if (isStatic) return importance;

  const now = Date.now();
  // Use last_accessed_at if available, otherwise created_at
  const refStr = lastAccessedAt || createdAt;
  if (!refStr) return importance * 0.5; // No date info — return neutral score
  const refTime = new Date(refStr + (refStr.includes("Z") ? "" : "Z")).getTime();
  if (isNaN(refTime)) return importance * 0.5; // Invalid date — return neutral score
  const elapsedDays = (now - refTime) / (1000 * 60 * 60 * 24);

  // Use FSRS stability if available, otherwise estimate from access patterns
  const effectiveStability = stability && stability > 0
    ? stability
    : fsrsInitialStability(FSRSRating.Good) * (1 + Math.min(accessCount * 0.3, 3) + Math.min((sourceCount - 1) * 0.2, 1));

  // FSRS-6 power-law retrievability instead of exponential decay
  const R = fsrsRetrievability(effectiveStability, elapsedDays);
  return importance * R;
}

