// Check a claimed race against the fly by replaying it.
//
//   node tools/verify-race.mjs <event.json>     race_finished properties copied from PostHog
//   node tools/verify-race.mjs --code 4821-0K7Q2M   fetch the race from PostHog, then replay it
//   node tools/verify-race.mjs --seed 4821      the fair (unpoked) fly's score on that level
//
// --code needs POSTHOG_PERSONAL_API_KEY (Settings > Personal API keys, scope query:read) in the
// environment. POSTHOG_PROJECT_ID defaults to 289642, POSTHOG_APP_HOST to https://eu.posthog.com.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { decodeCsr } from "../docs-site/live-loader.js";
import { replayRace, fairFlyRun } from "../docs-site/race-replay.js";

const LIVE = join(dirname(fileURLToPath(import.meta.url)), "..", "docs-site", "live");

async function loadModel() {
  const model = JSON.parse(await readFile(join(LIVE, "model.json"), "utf-8"));
  const bufs = await Promise.all(
    ["csr_indptr.bin", "csr_indices.bin", "csr_weights.bin"].map((f) => readFile(join(LIVE, f)))
  );
  const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  return { model, csr: decodeCsr(model, ...bufs.map(ab)) };
}

async function fetchByCode(code) {
  if (!/^\d{1,5}-[0-9A-Z]{7}$/.test(code)) throw new Error(`"${code}" is not a race code (like 4821-0K7Q2M)`);
  const key = process.env.POSTHOG_PERSONAL_API_KEY;
  const project = process.env.POSTHOG_PROJECT_ID ?? "289642";
  const host = process.env.POSTHOG_APP_HOST ?? "https://eu.posthog.com";
  if (!key) throw new Error("set POSTHOG_PERSONAL_API_KEY to fetch by code");
  const res = await fetch(`${host}/api/projects/${project}/query/`, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      query: {
        kind: "HogQLQuery",
        // code is validated above, so it is safe inside the query string
        query: `SELECT properties, timestamp FROM events WHERE event = 'race_finished' AND properties.race_code = '${code}' ORDER BY timestamp DESC LIMIT 5`,
      },
    }),
  });
  if (!res.ok) throw new Error(`PostHog query failed: HTTP ${res.status} ${await res.text()}`);
  const rows = (await res.json()).results ?? [];
  if (rows.length === 0) return null;
  if (rows.length > 1) console.log(`Note: ${rows.length} events carry this code (same race sent twice, or a copy). Checking the newest.`);
  const [props, ts] = rows[0];
  console.log(`Race logged at ${ts}`);
  return typeof props === "string" ? JSON.parse(props) : props;
}

function printVerdict(ev, result) {
  console.log(`Level #${ev.seed}, race code ${ev.race_code}`);
  for (const c of result.checks) {
    console.log(`  ${c.ok ? "ok  " : "BAD "} ${c.name}: claimed ${c.claimed}, replay ${c.replayed}`);
  }
  if (!result.ok) {
    console.log("\nMISMATCH: the logged flaps and pokes do not produce the claimed race. Treat the claim as false.");
    console.log("(Exception: the fly's numbers alone differing on a non-Chrome browser can be float rounding; the human's numbers never can.)");
    return;
  }
  const who = ev.human_pipes > ev.fly_pipes ? "The human passed more pipes" : ev.human_pipes < ev.fly_pipes ? "The fly passed more pipes" : "Same number of pipes";
  if (ev.finished) console.log(`\nVERIFIED: the race replays exactly. ${who} (${ev.human_pipes} vs ${ev.fly_pipes}).`);
  else {
    const still = [ev.human_alive && "the human", ev.fly_alive && "the fly"].filter(Boolean).join(" and ");
    console.log(`\nUNFINISHED: the log replays exactly, but ${still} ${ev.human_alive && ev.fly_alive ? "were" : "was"} still flying when it was sent (page closed or new level).`);
    console.log(`Score at that point: human ${ev.human_pipes}, fly ${ev.fly_pipes}. Look for a later event from the same session.`);
  }
  if (result.handicaps.length) console.log(`But the fly was handicapped: ${result.handicaps.join(", ")}. Not a fair race.`);
  console.log("Limit: a replay proves the log is consistent, not that a person played it (a bot could). The session recording shows that.");
  if (ev.pauses) console.log(`Paused ${ev.pauses} time(s) during the race.`);
}

const args = process.argv.slice(2);
const { model, csr } = await loadModel();

if (args[0] === "--seed") {
  const seed = Number(args[1]);
  if (!Number.isInteger(seed) || seed < 1) throw new Error("usage: --seed <level number>");
  const r = fairFlyRun(model, csr, seed);
  const secs = ((r.frames * model.game.frame_ms) / 1000).toFixed(1);
  console.log(`Level #${seed}: the fair fly ${r.alive ? "was still flying after" : "crashed after"} ${r.pipes} pipes, ${secs} s of game time.`);
  console.log("A player beat it on this level only with more pipes than that and no pokes.");
} else {
  let ev;
  if (args[0] === "--code") {
    ev = await fetchByCode(args[1] ?? "");
    if (!ev) {
      console.log("No race with that code in PostHog. Either it was never played, or an ad blocker stopped the event.");
      console.log(`Check the level itself: node tools/verify-race.mjs --seed ${String(args[1]).split("-")[0]}`);
      process.exit(1);
    }
  } else if (args[0]) {
    const raw = JSON.parse(await readFile(args[0], "utf-8"));
    ev = raw.properties ?? raw; // a whole exported event, or just its properties
  } else {
    console.log("usage: node tools/verify-race.mjs <event.json> | --code <race code> | --seed <level>");
    process.exit(2);
  }
  const result = replayRace(model, csr, ev);
  printVerdict(ev, result);
  process.exit(result.ok ? 0 : 1);
}
