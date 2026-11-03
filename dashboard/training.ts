// Lecture des logs d'entraînement (logs/*.log) pour /api/training. Côté serveur de dev uniquement : importé par
// vite.config.ts ; la page n'en importe que les types.
import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";

export type Point = [iter: number, value: number];

export type TrainingRun = {
  name: string;
  /** sft : mlx_lm.lora ; mlx_lm_lora : DPO/ORPO… ; grpo : script maison (reward au lieu d'une loss). */
  kind: "sft" | "mlx_lm_lora" | "grpo";
  iter: number;
  total: number | null;
  train: Point[];
  val: Point[];
  /** Dernier it/s rapporté par le log (hors temps de validation). */
  itPerSec: number | null;
  /** it/s moyen observé depuis la création du fichier, validations et chargement compris (null après une reprise). */
  observedItPerSec: number | null;
  peakGb: number | null;
  finished: boolean;
  error: string | null;
  mtime: number;
};

const ANSI = /\x1b\[[0-9;]*[A-Za-z]/g;
const num = (s: string | undefined) => (s == null ? null : Number(s));

/** Analyse un log ; null s'il ne ressemble à aucun entraînement connu (évals, génération…). */
export function parseLog(name: string, text: string): Omit<TrainingRun, "mtime" | "observedItPerSec"> & { startIter: number } | null {
  // tqdm réécrit sa ligne avec \r : on en fait des lignes à part entière.
  const lines = text.replace(ANSI, "").split(/\r\n|\r|\n/);
  let kind: TrainingRun["kind"] | null = null;
  let total: number | null = null;
  let iter = 0;
  let startIter = 0;
  let itPerSec: number | null = null;
  let peakGb: number | null = null;
  let finished = false;
  let error: string | null = null;
  const train: Point[] = [];
  const val: Point[] = [];
  let m: RegExpMatchArray | null;

  for (const raw of lines) {
    const l = raw.trim();
    if (!l) continue;
    if ((m = l.match(/^Starting training\.\.\., iters: (\d+)/))) {
      kind = "sft";
      total = Number(m[1]);
    } else if ((m = l.match(/^Training:.*?\|\s*(\d+)\/(\d+) \[/))) {
      kind ??= "mlx_lm_lora";
      iter = Math.max(iter, Number(m[1]));
      total = Number(m[2]);
    } else if ((m = l.match(/^\d+ exemples d'entraînement, (\d+) pas/))) {
      kind = "grpo";
      total = Number(m[1]);
    } else if ((m = l.match(/^reprise au pas (\d+)/))) {
      startIter = Number(m[1]);
    } else if ((m = l.match(/^Iter (\d+): Val loss ([\d.eE+-]+)/))) {
      val.push([Number(m[1]), Number(m[2])]);
      iter = Math.max(iter, Number(m[1]));
    } else if ((m = l.match(/^Iter (\d+): (?:Train )?loss ([\d.eE+-]+)/))) {
      const i = Number(m[1]);
      train.push([i, Number(m[2])]);
      iter = Math.max(iter, i);
      itPerSec = num(l.match(/(?:It\/sec|it\/s) ([\d.]+)/i)?.[1]) ?? itPerSec;
      peakGb = num(l.match(/(?:Peak mem|peak_mem) ([\d.]+) ?GB/i)?.[1]) ?? peakGb;
    } else if ((m = l.match(/^step (\d+) reward ([\d.eE+-]+)/))) {
      const i = Number(m[1]);
      train.push([i, Number(m[2])]);
      iter = Math.max(iter, i);
      const s = num(l.match(/step_s ([\d.]+)/)?.[1]);
      if (s) itPerSec = 1 / s;
      peakGb = num(l.match(/peak_gb ([\d.]+)/)?.[1]) ?? peakGb;
    } else if (/^Saved final weights/.test(l)) {
      finished = true;
    } else if (/^\w*(Error|Exception):|^libc\+\+abi/.test(l)) {
      error = l; // la dernière ligne d'une traceback est la plus parlante
    }
  }
  if (!kind) return null;
  if (kind === "grpo" && total != null && iter >= total) finished = true;
  return { name, kind, iter, total, train, val, itPerSec, peakGb, finished, error, startIter };
}

type Cached = { key: string; run: TrainingRun | null };
const cache = new Map<string, Cached>();

/** Tous les logs d'entraînement de `dir`, du plus récemment modifié au plus ancien. Relit seulement les fichiers modifiés. */
export async function readTrainingRuns(dir: string): Promise<TrainingRun[]> {
  const files = (await readdir(dir)).filter((f) => f.endsWith(".log"));
  const runs = await Promise.all(
    files.map(async (f) => {
      const path = join(dir, f);
      const st = await stat(path);
      const key = `${st.mtimeMs}:${st.size}`;
      const hit = cache.get(path);
      if (hit?.key === key) return hit.run;
      const parsed = parseLog(f.replace(/\.log$/, ""), await readFile(path, "utf8"));
      let run: TrainingRun | null = null;
      if (parsed) {
        const { startIter, ...rest } = parsed;
        // Après une reprise ajoutée au même fichier, la date de création ne dit plus rien de la vitesse.
        const elapsed = (st.mtimeMs - st.birthtimeMs) / 1000;
        const observed = startIter === 0 && elapsed > 60 && rest.iter > 0 ? rest.iter / elapsed : null;
        run = { ...rest, mtime: st.mtimeMs, observedItPerSec: observed };
      }
      cache.set(path, { key, run });
      return run;
    }),
  );
  return runs.filter((r): r is TrainingRun => r != null).sort((a, b) => b.mtime - a.mtime);
}
