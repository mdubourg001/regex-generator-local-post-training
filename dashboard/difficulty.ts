import { readFile } from "node:fs/promises";
import path from "node:path";

export const LEVELS = ["simple", "medium", "hard"] as const;
export type Level = (typeof LEVELS)[number];
/** Par niveau : nombre de cas et nombre de succès. */
export type Breakdown = Record<Level, { n: number; ok: number }>;

async function readJsonl(file: string): Promise<Record<string, unknown>[]> {
  const rows: Record<string, unknown>[] = [];
  for (const l of (await readFile(file, "utf8")).split("\n")) {
    if (!l.trim()) continue;
    try {
      rows.push(JSON.parse(l));
    } catch {
      // ligne en cours d'écriture
    }
  }
  return rows;
}

/**
 * Pour chaque run (dernière ligne de scores.jsonl par nom), croise results/<name>.jsonl avec le jeu d'éval
 * (champ `difficulty`) par `id`. results/<name>.jsonl est écrasé à chaque run : seul le dernier run d'un nom
 * a une ventilation fiable. Les runs sans fichier de détail ou sans difficulté dans le jeu sont omis.
 */
export async function readDifficulty(root: string, scoresFile: string): Promise<Record<string, Breakdown>> {
  const latest = new Map<string, string>(); // name -> data
  for (const s of await readJsonl(scoresFile)) latest.set(String(s.name), String(s.data));

  const levelsByData = new Map<string, Map<string, Level> | null>();
  const levelsOf = async (data: string) => {
    if (!levelsByData.has(data)) {
      try {
        const m = new Map<string, Level>();
        for (const r of await readJsonl(path.resolve(root, data)))
          if (LEVELS.includes(r.difficulty as Level)) m.set(String(r.id), r.difficulty as Level);
        levelsByData.set(data, m.size ? m : null);
      } catch {
        levelsByData.set(data, null);
      }
    }
    return levelsByData.get(data)!;
  };

  const out: Record<string, Breakdown> = {};
  await Promise.all(
    [...latest].map(async ([name, data]) => {
      const levels = await levelsOf(data);
      if (!levels) return;
      let rows;
      try {
        rows = await readJsonl(path.join(root, "results", `${name}.jsonl`));
      } catch {
        return;
      }
      const b: Breakdown = { simple: { n: 0, ok: 0 }, medium: { n: 0, ok: 0 }, hard: { n: 0, ok: 0 } };
      for (const r of rows) {
        const lvl = levels.get(String(r.id));
        if (!lvl) continue;
        b[lvl].n++;
        if (r.success === true) b[lvl].ok++;
      }
      if (LEVELS.some((l) => b[l].n > 0)) out[name] = b;
    }),
  );
  return out;
}
