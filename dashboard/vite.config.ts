import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { readDifficulty } from "./difficulty";
import { readTrainingRuns } from "./training";

const SCORES = fileURLToPath(new URL("../results/scores.jsonl", import.meta.url));
const LOGS = fileURLToPath(new URL("../logs", import.meta.url));
const ROOT = fileURLToPath(new URL("..", import.meta.url));

// Sert results/scores.jsonl tel quel sur /api/scores, relu à chaque requête : les nouvelles évals
// apparaissent sans rebuild. Serveur de dev uniquement.
function scoresApi(): Plugin {
  return {
    name: "scores-api",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/api/scores", async (_req, res) => {
        try {
          const body = await readFile(SCORES, "utf8");
          res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
          res.setHeader("Cache-Control", "no-store");
          res.end(body);
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });
    },
  };
}

// Ventilation du succès par difficulté (simple/medium/hard), recalculée à chaque requête : voir difficulty.ts.
function difficultyApi(): Plugin {
  return {
    name: "difficulty-api",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/api/difficulty", async (_req, res) => {
        try {
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.setHeader("Cache-Control", "no-store");
          res.end(JSON.stringify(await readDifficulty(ROOT, SCORES)));
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });
    },
  };
}

// Progression des entraînements, extraite des logs/*.log (SFT, DPO, GRPO) à chaque requête : voir training.ts.
function trainingApi(): Plugin {
  return {
    name: "training-api",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/api/training", async (_req, res) => {
        try {
          const runs = await readTrainingRuns(LOGS);
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.setHeader("Cache-Control", "no-store");
          res.end(JSON.stringify({ now: Date.now(), runs }));
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), scoresApi(), difficultyApi(), trainingApi()],
  // Port fixe, voisin de celui du playground (5299).
  server: { port: 5297, strictPort: true },
});
