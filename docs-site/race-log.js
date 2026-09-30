// Race log: everything needed to replay a race exactly (level seed, the human's flap frames, and
// every brain poke with the fly frame it took effect on). Game and brain are deterministic per
// seed, so tools/verify-race.mjs can re-run both birds from this log and check the claimed scores.
// No imports: shared by the browser and Node.

export const RACE_LOG_FORMAT = 1;

/** The brain seed a level starts from (live-app startLevel and "Reset brain"). */
export const brainSeed = (seed) => (seed ^ 0x5bd1e995) >>> 0;

const POKE_KEYS = ["blindLeft", "blindRight", "cutGiantFiber", "gain", "inhibitionOff"];

const snapshot = (pokes) => Object.fromEntries(POKE_KEYS.map((k) => [k, pokes[k]]));

/** A new log for a race on `seed`, starting with the pokes already set (they carry over between levels). */
export function newRaceLog(seed, pokes) {
  return { seed, flaps: [], pokeLog: [{ f: 0, p: snapshot(pokes), r: false }], pauses: 0, started: false, sent: false };
}

/**
 * Record the pokes in force from fly frame `frame` on. `reset` = the brain was also reset
 * ("Reset brain"). Every change is kept, even several in one frame: cutting and restoring the
 * giant fiber puts those neurons back to rest, so an on-off pair is not a no-op.
 */
export function logPokes(log, frame, pokes, reset) {
  log.pokeLog.push({ f: frame, p: snapshot(pokes), r: reset });
}

/** Plain-language list of the ways the fly was handicapped during the race (empty = a fair race). */
export function handicaps(log) {
  const out = new Set();
  for (const { f, p, r } of log.pokeLog) {
    if (p.blindLeft) out.add("left eye blind");
    if (p.blindRight) out.add("right eye blind");
    if (p.cutGiantFiber) out.add("giant fiber cut");
    if (p.gain < 1) out.add("eyes dimmed");
    if (p.inhibitionOff) out.add("inhibition removed");
    if (r && f > 0) out.add("brain reset mid-race");
  }
  return [...out];
}

/** 32-bit FNV-1a over a string. */
function fnv1a(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Short code shown on the result screen and sent with the race, e.g. "4821-0K7Q2M". Identifies
 * the race in PostHog; it is a checksum of the log, not a secret (the replay is the real check).
 */
export function raceCode({ seed, flaps, pokeLog, humanFrames, flyFrames }) {
  const h = fnv1a(JSON.stringify([seed, flaps, pokeLog, humanFrames, flyFrames]));
  return `${seed}-${h.toString(36).toUpperCase().padStart(7, "0")}`;
}
