// Worker persistant qui exécute des regex JS sur des tests, avec un délai maximal par regex.
//
// Protocole : une requête JSON par ligne sur stdin, une réponse JSON par ligne sur stdout (pas forcément dans l'ordre,
// d'où l'`id`).
//   requête : {"id": 0, "pattern": "^a+$", "flags": "i", "match": ["aa"], "no_match": ["b"], "timeout_ms": 1000}
//   réponse : {"id": 0, "match": [true], "no_match": [true]}   un booléen par test : true = test réussi
//             {"id": 0, "error": "Invalid regular expression: …"}
//             {"id": 0, "timeout": true}
//
// Un `RegExp` JS est synchrone et ne s'interrompt pas : une regex en backtracking catastrophique (ReDoS) bloquerait le
// thread. Chaque regex tourne donc dans un worker_thread, qu'on tue et remplace s'il dépasse le délai.

import { availableParallelism } from "node:os";
import { createInterface } from "node:readline";
import { Worker } from "node:worker_threads";

const DEFAULT_TIMEOUT_MS = 1000;
const POOL_SIZE = Math.max(1, availableParallelism() - 2);

const WORKER_SOURCE = `
const { parentPort } = require("node:worker_threads");
parentPort.on("message", ({ pattern, flags, match, no_match }) => {
  try {
    const regex = new RegExp(pattern, flags);
    // Avec les flags g ou y, test() part de lastIndex et le modifie : on repart de 0 pour chaque chaîne.
    const test = (s) => {
      regex.lastIndex = 0;
      return regex.test(s);
    };
    parentPort.postMessage({ match: match.map(test), no_match: no_match.map((s) => !test(s)) });
  } catch (e) {
    parentPort.postMessage({ error: String(e.message) });
  }
});
`;

const idle = [];
const queue = [];
let inFlight = 0;
let inputClosed = false;

function spawn() {
  idle.push(new Worker(WORKER_SOURCE, { eval: true }));
}

function reply(response) {
  process.stdout.write(JSON.stringify(response) + "\n");
  inFlight--;
  if (inputClosed && inFlight === 0) process.exit(0);
}

function dispatch() {
  while (idle.length && queue.length) {
    const worker = idle.pop();
    const { id, timeout_ms = DEFAULT_TIMEOUT_MS, ...job } = queue.shift();
    const timer = setTimeout(() => {
      worker.removeAllListeners("message");
      worker.terminate();
      spawn();
      reply({ id, timeout: true });
      dispatch();
    }, timeout_ms);
    worker.once("message", (result) => {
      clearTimeout(timer);
      idle.push(worker);
      reply({ id, ...result });
      dispatch();
    });
    worker.postMessage(job);
  }
}

for (let i = 0; i < POOL_SIZE; i++) spawn();

const input = createInterface({ input: process.stdin });
input.on("line", (line) => {
  if (!line.trim()) return;
  inFlight++;
  queue.push(JSON.parse(line));
  dispatch();
});
input.on("close", () => {
  inputClosed = true;
  if (inFlight === 0) process.exit(0);
});
