import { useEffect, useMemo, useState } from "react";
import type { Point, TrainingRun } from "../training";

const POLL_MS = 5000;
/** Sans nouvelle ligne depuis ce délai et sans fin propre, un run est considéré comme interrompu. */
const STALE_MS = 5 * 60_000;
/** Par défaut, on n'affiche que les logs modifiés récemment. */
const RECENT_MS = 24 * 3600_000;

type Status = "actif" | "terminé" | "interrompu";

function statusOf(r: TrainingRun, now: number): Status {
  if (r.finished) return "terminé";
  if (r.error || now - r.mtime > STALE_MS) return "interrompu";
  return "actif";
}

const KIND_LABEL: Record<TrainingRun["kind"], string> = { sft: "SFT", mlx_lm_lora: "DPO / préférences", grpo: "GRPO" };

const numFmt = (digits: number) => new Intl.NumberFormat("fr-FR", { maximumFractionDigits: digits });
const fmt3 = numFmt(3);
const fmt2 = numFmt(2);
const fmt1 = numFmt(1);

function span(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h} h ${String(m % 60).padStart(2, "0")}`;
  return `${Math.floor(h / 24)} j`;
}

export default function Training() {
  const [data, setData] = useState<{ now: number; runs: TrainingRun[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [all, setAll] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const res = await fetch("/api/training", { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
        const json = await res.json();
        if (!alive) return;
        setData(json);
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

  const now = data?.now ?? Date.now();
  const runs = data?.runs ?? [];
  const shown = useMemo(
    () => (all ? runs : runs.filter((r) => statusOf(r, now) === "actif" || now - r.mtime < RECENT_MS)),
    [runs, all, now],
  );
  const active = runs.filter((r) => statusOf(r, now) === "actif").length;

  return (
    <section className="mb-8">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-xl font-semibold">
          Entraînements
          {active > 0 && <span className="ml-2 text-sm font-normal text-emerald-600 dark:text-emerald-400">{active} en cours</span>}
        </h2>
        <div className="flex items-center gap-3 text-sm text-zinc-500 dark:text-zinc-400">
          {error && <span className="text-red-600 dark:text-red-400">Erreur de lecture : {error}</span>}
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
            tous les logs ({runs.length})
          </label>
        </div>
      </div>
      {!data ? (
        <p className="text-sm text-zinc-500">Chargement…</p>
      ) : shown.length === 0 ? (
        <p className="text-sm text-zinc-500">Aucun entraînement depuis 24 h.</p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {shown.map((r) => (
            <RunCard key={r.name} run={r} now={now} />
          ))}
        </div>
      )}
      <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
        Source : <code>logs/*.log</code>, relus toutes les {POLL_MS / 1000} s. Itérations en micro-batchs (avec accumulation de
        gradient, plusieurs par mise à jour). Pic mémoire : celui que rapporte MLX, en dessous de la pression réelle sur la machine.
        Un run sans nouvelle ligne depuis {STALE_MS / 60_000} min et sans fin propre est marqué interrompu.
      </p>
    </section>
  );
}

const BADGE: Record<Status, string> = {
  actif: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  terminé: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300",
  interrompu: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
};

function RunCard({ run: r, now }: { run: TrainingRun; now: number }) {
  const status = statusOf(r, now);
  const progress = r.total ? Math.min(1, r.iter / r.total) : null;
  // La vitesse observée (validations comprises) est plus fiable pour l'ETA que le dernier it/s rapporté, qui les exclut.
  const rate = r.observedItPerSec ?? r.itPerSec;
  const eta = status === "actif" && r.total && rate ? ((r.total - r.iter) / rate) * 1000 : null;
  const metric = r.kind === "grpo" ? "reward" : "loss";
  const lastTrain = r.train.at(-1)?.[1];
  const lastVal = r.val.at(-1)?.[1];

  return (
    <div className="rounded border border-zinc-200 p-3 dark:border-zinc-800">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-medium">{r.name}</span>
        <span className="text-xs text-zinc-500 dark:text-zinc-400">{KIND_LABEL[r.kind]}</span>
        <span className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${BADGE[status]}`}>
          {status === "actif" && <span className="mr-1 inline-block size-1.5 animate-pulse rounded-full bg-emerald-500 align-middle" />}
          {status}
        </span>
        <span className="ml-auto text-xs text-zinc-500 dark:text-zinc-400" title={new Date(r.mtime).toLocaleString("fr-FR")}>
          dernière ligne il y a {span(now - r.mtime)}
        </span>
      </div>

      <div className="mb-1 h-2 rounded-full bg-zinc-100 dark:bg-zinc-900">
        <div
          className={`h-full rounded-full ${status === "actif" ? "bg-emerald-500" : status === "terminé" ? "bg-sky-600 dark:bg-sky-500" : "bg-red-400 dark:bg-red-500"}`}
          style={{ width: `${(progress ?? 0) * 100}%` }}
        />
      </div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-0.5 text-xs tabular-nums text-zinc-600 dark:text-zinc-300">
        <span>
          {r.iter} / {r.total ?? "?"} it{progress != null && ` · ${fmt1.format(progress * 100)} %`}
        </span>
        {eta != null && (
          <span title="Vitesse moyenne observée depuis le début du log">
            fin vers {new Date(now + eta).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })} (dans {span(eta)})
          </span>
        )}
        {r.itPerSec != null && <span>{fmt3.format(r.itPerSec)} it/s</span>}
        {r.peakGb != null && <span>pic {fmt2.format(r.peakGb)} Go</span>}
        {lastTrain != null && (
          <span>
            {metric} train {fmt3.format(lastTrain)}
            {lastVal != null && ` · val ${fmt3.format(lastVal)}`}
          </span>
        )}
      </div>

      <Chart train={r.train} val={r.val} total={r.total ?? r.iter} log={metric === "loss"} />
      {r.error && (
        <p className="mt-1 truncate font-mono text-[11px] text-red-600 dark:text-red-400" title={r.error}>
          {r.error}
        </p>
      )}
    </div>
  );
}

const W = 400;
const H = 90;
const PAD = 2;

/** Courbe train (ligne) et val (points reliés), axe x = itérations jusqu'au total prévu ; y log pour une loss. */
function Chart({ train, val, total, log }: { train: Point[]; val: Point[]; total: number; log: boolean }) {
  const values = [...train, ...val].map(([, v]) => v).filter((v) => !log || v > 0);
  if (values.length < 2) return <div className="h-[90px] text-xs text-zinc-400">Pas encore de points.</div>;
  const t = (v: number) => (log ? Math.log(v) : v);
  const lo = t(Math.min(...values));
  const hi = t(Math.max(...values));
  const x = (i: number) => PAD + (i / Math.max(1, total)) * (W - 2 * PAD);
  const y = (v: number) => H - PAD - ((t(v) - lo) / (hi - lo || 1)) * (H - 2 * PAD);
  const path = (pts: Point[]) => pts.filter(([, v]) => !log || v > 0).map(([i, v]) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-[90px] w-full">
        <polyline points={path(train)} fill="none" className="stroke-sky-600 dark:stroke-sky-400" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
        {val.length > 1 && (
          <polyline points={path(val)} fill="none" className="stroke-amber-500" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      <div className="pointer-events-none absolute inset-0 flex flex-col justify-between text-[10px] tabular-nums text-zinc-400">
        <span>{fmt3.format(Math.max(...values))}</span>
        <span>{fmt3.format(Math.min(...values))}</span>
      </div>
      <div className="mt-0.5 flex gap-3 text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>
          <span className="text-sky-600 dark:text-sky-400">━</span> train
        </span>
        {val.length > 0 && (
          <span>
            <span className="text-amber-500">━</span> val
          </span>
        )}
        {log && <span>échelle log</span>}
      </div>
    </div>
  );
}
