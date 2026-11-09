import { useEffect, useRef, useState } from "react";
import type { LineResult, MatchRequest, MatchResponse } from "./matcher.worker";

export type { LineResult };

/** Au-delà, le worker est terminé et la regex considérée comme trop lente (backtracking catastrophique). */
export const TIMEOUT_MS = 1500;

type Inflight = { key: string; lines: string[]; timer: ReturnType<typeof setTimeout> };

/**
 * Résultats de `regex` sur `lines`, calculés dans un worker. Ils sont gardés par ligne pour la regex courante :
 * pendant la frappe, seule la ligne modifiée est recalculée et les autres restent affichées. Une ligne sans résultat
 * (le temps d'un aller-retour avec le worker, ~1 ms) s'affiche sans surlignage.
 */
export function useMatches(regex: RegExp | null, lines: string[]) {
  const key = regex ? `/${regex.source}/${regex.flags}` : null;
  const [state, setState] = useState({ key: null as string | null, results: new Map<string, LineResult>(), timedOut: false });
  const ctx = useRef({
    worker: null as Worker | null,
    key: null as string | null, // regex des lignes déjà demandées
    requested: new Set<string>(),
    inflight: new Map<number, Inflight>(),
    nextId: 0,
  });
  const current = state.key === key ? state : { key, results: new Map<string, LineResult>(), timedOut: false };

  useEffect(() => {
    const c = ctx.current;
    return () => {
      c.worker?.terminate();
      c.inflight.forEach((r) => clearTimeout(r.timer));
    };
  }, []);

  // Sans dépendances : à chaque rendu, ne demande que les lignes jamais demandées pour cette regex.
  useEffect(() => {
    const c = ctx.current;
    if (!regex || !key || current.timedOut) return;
    if (c.key !== key) {
      c.key = key;
      c.requested = new Set();
    }
    const missing = [...new Set(lines)].filter((l) => !c.requested.has(l));
    if (!missing.length) return;
    missing.forEach((l) => c.requested.add(l));

    if (!c.worker) {
      c.worker = new Worker(new URL("./matcher.worker.ts", import.meta.url), { type: "module" });
      c.worker.onmessage = ({ data }: MessageEvent<MatchResponse>) => {
        const request = c.inflight.get(data.id);
        if (!request) return;
        clearTimeout(request.timer);
        c.inflight.delete(data.id);
        if (request.key !== c.key) return; // regex remplacée entre-temps
        setState((s) => {
          const results = new Map(s.key === request.key ? s.results : undefined);
          request.lines.forEach((line, i) => results.set(line, data.results[i]));
          return { key: request.key, results, timedOut: false };
        });
      };
    }

    const id = ++c.nextId;
    const timer = setTimeout(() => {
      // Le worker est bloqué : on le termine, avec toutes ses requêtes. La prochaine regex en recrée un.
      c.worker?.terminate();
      c.worker = null;
      c.inflight.forEach((r) => clearTimeout(r.timer));
      c.inflight.clear();
      c.key = null;
      setState((s) => ({ key, results: s.key === key ? s.results : new Map(), timedOut: true }));
    }, TIMEOUT_MS);
    c.inflight.set(id, { key, lines: missing, timer });
    c.worker.postMessage({ id, source: regex.source, flags: regex.flags, lines: missing } satisfies MatchRequest);
  });

  return { results: current.results, timedOut: current.timedOut };
}
