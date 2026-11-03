import { useEffect, useMemo, useRef, useState } from "react";
import Training from "./Training";

type Score = {
  name: string;
  model: string;
  adapter: string | null;
  few_shot: boolean;
  thinking?: boolean;
  data: string;
  n: number;
  success_rate: number;
  mean_pass_rate: number;
  malformed_rate: number;
  timeout_rate: number;
  generation_seconds: number;
};

/** Une ligne de scores.jsonl, avec sa position dans le fichier (append-only : plus grand = plus récent). */
type Run = Score & { line: number; latest: boolean };

const POLL_MS = 5000;

const DATASETS = [
  { id: "regex101", label: "regex101 « réel »", match: (d: string) => d === "data/eval/benchmark-regex101.jsonl" },
  { id: "heldout", label: "held-out", match: (d: string) => d === "data/prepared/heldout.jsonl" },
  { id: "benchmark", label: "benchmark (écrit par LLM, biaisé)", match: (d: string) => d === "data/eval/benchmark.jsonl" },
  { id: "gpt", label: "benchmark-gpt (écrit par LLM, biaisé)", match: (d: string) => d === "data/eval/benchmark-gpt.jsonl" },
] as const;

type DatasetId = (typeof DATASETS)[number]["id"] | "autre" | "tous";

function datasetOf(data: string): DatasetId {
  return DATASETS.find((d) => d.match(data))?.id ?? "autre";
}

function datasetLabel(id: DatasetId): string {
  if (id === "autre") return "autre (sous-ensembles, échantillons…)";
  if (id === "tous") return "tous";
  return DATASETS.find((d) => d.id === id)!.label;
}

const basename = (p: string) => p.split("/").filter(Boolean).pop() ?? p;
/** Baseline : modèle du Hub sans adapter (un modèle fusionné local, ex. models/sft-0.6b, n'en est pas une). */
const isBaseline = (r: Score) => r.adapter == null && !r.model.startsWith("models/") && !r.model.startsWith("results/");

const pctFmt = new Intl.NumberFormat("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const pct = (x: number) => `${pctFmt.format(x * 100)} %`;
function duration(s: number): string {
  if (s < 60) return `${pctFmt.format(s)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${String(Math.round(s % 60)).padStart(2, "0")}`;
}

function parse(text: string): { runs: Run[]; invalid: number } {
  const scores: (Score & { line: number })[] = [];
  let invalid = 0;
  text.split("\n").forEach((l, i) => {
    if (!l.trim()) return;
    try {
      scores.push({ ...(JSON.parse(l) as Score), line: i + 1 });
    } catch {
      invalid++; // ligne en cours d'écriture ou corrompue
    }
  });
  const lastLine = new Map<string, number>();
  for (const s of scores) lastLine.set(s.name, s.line);
  return { runs: scores.map((s) => ({ ...s, latest: lastLine.get(s.name) === s.line })), invalid };
}

type SortKey = "name" | "model" | "adapter" | "few_shot" | "n" | "success_rate" | "mean_pass_rate" | "malformed_rate" | "timeout_rate" | "generation_seconds";

const COLUMNS: { key: SortKey; label: string; numeric?: boolean }[] = [
  { key: "name", label: "Nom" },
  { key: "model", label: "Modèle" },
  { key: "adapter", label: "Adapter" },
  { key: "few_shot", label: "Few-shot" },
  { key: "n", label: "n", numeric: true },
  { key: "success_rate", label: "Succès", numeric: true },
  { key: "mean_pass_rate", label: "Tests réussis", numeric: true },
  { key: "malformed_rate", label: "Mal formées", numeric: true },
  { key: "timeout_rate", label: "Timeouts", numeric: true },
  { key: "generation_seconds", label: "Durée", numeric: true },
];

function sortValue(r: Run, k: SortKey): string | number {
  if (k === "model") return basename(r.model);
  if (k === "adapter") return r.adapter ? basename(r.adapter) : "";
  if (k === "few_shot") return r.few_shot ? 1 : 0;
  return r[k];
}

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [invalid, setInvalid] = useState(0);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const initialLines = useRef<number | null>(null);

  const [dataset, setDataset] = useState<DatasetId>("regex101");
  const [query, setQuery] = useState("");
  const [baselines, setBaselines] = useState(true);
  const [history, setHistory] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "success_rate", desc: true });

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const res = await fetch("/api/scores", { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
        const parsed = parse(await res.text());
        if (!alive) return;
        initialLines.current ??= Math.max(0, ...parsed.runs.map((r) => r.line));
        setRuns(parsed.runs);
        setInvalid(parsed.invalid);
        setUpdatedAt(new Date());
        setError(null);
      } catch (e) {
        if (alive) setError(String(e));
      }
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const isNew = (r: Run) => initialLines.current != null && r.line > initialLines.current;

  const counts = useMemo(() => {
    const c = new Map<DatasetId, number>();
    for (const r of runs) if (r.latest) c.set(datasetOf(r.data), (c.get(datasetOf(r.data)) ?? 0) + 1);
    return c;
  }, [runs]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = runs.filter(
      (r) =>
        (history || r.latest) &&
        (dataset === "tous" || datasetOf(r.data) === dataset) &&
        (baselines || !isBaseline(r)) &&
        (!q || r.name.toLowerCase().includes(q) || r.model.toLowerCase().includes(q)),
    );
    const dir = sort.desc ? -1 : 1;
    return rows.sort((a, b) => {
      const va = sortValue(a, sort.key);
      const vb = sortValue(b, sort.key);
      const c = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb), "fr");
      return c * dir || b.line - a.line;
    });
  }, [runs, dataset, query, baselines, history, sort]);

  const olderCount = runs.length - runs.filter((r) => r.latest).length;
  const maxSuccess = Math.max(0.01, ...filtered.map((r) => r.success_rate));
  const byName = [...filtered].sort((a, b) => b.success_rate - a.success_rate);

  return (
    <div className="min-h-screen bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      <div className="mx-auto max-w-7xl px-4 py-6">
        <Training />

        <header className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          <h1 className="text-xl font-semibold">Évaluations NL → regex</h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            {error ? (
              <span className="text-red-600 dark:text-red-400">Erreur de lecture : {error}</span>
            ) : updatedAt ? (
              <>
                Dernière mise à jour : {updatedAt.toLocaleTimeString("fr-FR")} · {runs.length} lignes
                {invalid > 0 && ` · ${invalid} ligne(s) illisible(s)`}
              </>
            ) : (
              "Chargement…"
            )}
          </p>
        </header>

        <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
          <label className="flex items-center gap-2">
            Jeu d'éval
            <select
              className="rounded border border-zinc-300 bg-white px-2 py-1 dark:border-zinc-700 dark:bg-zinc-900"
              value={dataset}
              onChange={(e) => setDataset(e.target.value as DatasetId)}
            >
              {([...DATASETS.map((d) => d.id), "autre", "tous"] as DatasetId[]).map((id) => (
                <option key={id} value={id}>
                  {datasetLabel(id)}
                  {id !== "tous" && ` (${counts.get(id) ?? 0})`}
                </option>
              ))}
            </select>
          </label>
          <input
            className="w-56 rounded border border-zinc-300 bg-white px-2 py-1 dark:border-zinc-700 dark:bg-zinc-900"
            placeholder="Rechercher nom ou modèle…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={baselines} onChange={(e) => setBaselines(e.target.checked)} />
            baselines (sans adapter)
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={history} onChange={(e) => setHistory(e.target.checked)} />
            runs précédents ({olderCount} remplacés par un run plus récent du même nom)
          </label>
        </div>

        {(dataset === "benchmark" || dataset === "gpt") && (
          <p className="mb-4 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
            Ce benchmark a été écrit par un LLM, comme une partie des données d'entraînement : les scores y sont biaisés à la hausse.
            Préférer regex101 « réel » ou held-out.
          </p>
        )}

        <section className="mb-6">
          <h2 className="mb-2 text-sm font-medium text-zinc-500 dark:text-zinc-400">Taux de succès</h2>
          {byName.length === 0 ? (
            <p className="text-sm text-zinc-500">Aucun run pour ces filtres.</p>
          ) : (
            <div className="space-y-1">
              {byName.map((r) => (
                <div key={r.line} className="flex items-center gap-2 text-xs">
                  <div className={`w-64 shrink-0 truncate text-right ${r.latest ? "" : "text-zinc-400"}`} title={r.name}>
                    {isNew(r) && <span className="mr-1 font-semibold text-amber-600 dark:text-amber-400">nouveau</span>}
                    {r.name}
                  </div>
                  <div className="h-4 flex-1 rounded-sm bg-zinc-100 dark:bg-zinc-900">
                    <div
                      className={`h-full rounded-sm ${isNew(r) ? "bg-amber-500" : isBaseline(r) ? "bg-zinc-400 dark:bg-zinc-600" : "bg-sky-600 dark:bg-sky-500"}`}
                      style={{ width: `${(r.success_rate / maxSuccess) * 100}%` }}
                    />
                  </div>
                  <div className="w-14 shrink-0 tabular-nums">{pct(r.success_rate)}</div>
                </div>
              ))}
              <p className="pt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Échelle relative au meilleur run affiché. <span className="text-sky-600 dark:text-sky-500">■</span> entraîné ·{" "}
                <span className="text-zinc-400 dark:text-zinc-600">■</span> baseline ·{" "}
                <span className="text-amber-500">■</span> nouveau depuis l'ouverture de la page
              </p>
            </div>
          )}
        </section>

        <div className="overflow-x-auto rounded border border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-50 text-left dark:bg-zinc-900">
              <tr>
                {COLUMNS.map((c) => (
                  <th
                    key={c.key}
                    className={`cursor-pointer select-none whitespace-nowrap px-3 py-2 font-medium hover:text-sky-600 ${c.numeric ? "text-right" : ""}`}
                    onClick={() =>
                      setSort((s) => (s.key === c.key ? { key: c.key, desc: !s.desc } : { key: c.key, desc: !!c.numeric }))
                    }
                  >
                    {c.label}
                    {sort.key === c.key && (sort.desc ? " ↓" : " ↑")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr
                  key={r.line}
                  className={`border-t border-zinc-200 dark:border-zinc-800 ${r.latest ? "" : "text-zinc-400 dark:text-zinc-500"}`}
                  style={isNew(r) ? { animation: "flash 4s ease-out" } : undefined}
                >
                  <td className="px-3 py-1.5 font-mono text-xs" title={`ligne ${r.line} · ${r.data}`}>
                    {isNew(r) && <span className="mr-1 rounded bg-amber-500 px-1 text-[10px] font-semibold text-white">nouveau</span>}
                    {!r.latest && <span className="mr-1 text-[10px]">(ancien)</span>}
                    {r.name}
                    {dataset === "tous" && <span className="ml-2 text-zinc-400">{datasetLabel(datasetOf(r.data))}</span>}
                  </td>
                  <td className="px-3 py-1.5 whitespace-nowrap" title={r.model}>
                    {basename(r.model)}
                  </td>
                  <td className="px-3 py-1.5 whitespace-nowrap" title={r.adapter ?? undefined}>
                    {r.adapter ? basename(r.adapter) : "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    {r.few_shot ? "oui" : "non"}
                    {r.thinking && " · thinking"}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{r.n}</td>
                  <td className="px-3 py-1.5 text-right font-semibold tabular-nums">{pct(r.success_rate)}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{pct(r.mean_pass_rate)}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{pct(r.malformed_rate)}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{pct(r.timeout_rate)}</td>
                  <td className="px-3 py-1.5 text-right whitespace-nowrap tabular-nums">{duration(r.generation_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
          Succès : toutes les assertions de l'exemple passent. Tests réussis : part moyenne d'assertions passées. Source :{" "}
          <code>results/scores.jsonl</code>, relu toutes les {POLL_MS / 1000} s.
        </p>
      </div>
    </div>
  );
}
