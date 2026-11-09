import { useMemo, useRef } from "react";
import { type LineResult, TIMEOUT_MS, useMatches } from "./useMatches";

type Segment = { text: string; match?: number };
/** Chaîne d'exemple et résultat attendu de RegExp.prototype.test. */
export type Example = { text: string; expected: boolean };

/** Découpe une ligne selon les correspondances, numérotées pour alterner les couleurs. */
function segments(line: string, result: LineResult | undefined): Segment[] {
  if (!result) return [{ text: line }];
  const out: Segment[] = [];
  let last = 0;
  result.ranges.forEach(([start, end], i) => {
    if (start > last) out.push({ text: line.slice(last, start) });
    out.push({ text: line.slice(start, end), match: i });
    last = Math.max(last, end);
  });
  if (last < line.length) out.push({ text: line.slice(last) });
  return out;
}

const MARKS = ["bg-sky-300/70 dark:bg-sky-500/50", "bg-sky-200/70 dark:bg-sky-300/35"];
const ROW = "h-6 leading-6 whitespace-pre";
const LABEL = "text-xs font-semibold tracking-wide text-zinc-500 uppercase";

/** Segments surlignés ; `transparent` : texte invisible, pour le calque derrière la zone de saisie. */
function Highlighted({ parts, transparent = false }: { parts: Segment[]; transparent?: boolean }) {
  return parts.map((s, j) =>
    s.match === undefined ? (
      <span key={j}>{s.text}</span>
    ) : s.text === "" ? (
      <span key={j} className="relative inline-block h-full w-0 border-l-2 border-sky-500 align-top" />
    ) : (
      <mark key={j} className={`rounded-sm ${transparent ? "text-transparent" : "text-inherit"} ${MARKS[s.match % 2]}`}>
        {s.text}
      </mark>
    ),
  );
}

export function TestArea(props: {
  value: string;
  onChange: (v: string) => void;
  regex: RegExp | null;
  examples: Example[];
}) {
  const { value, onChange, regex, examples } = props;
  const lines = useMemo(() => value.split("\n"), [value]);
  const all = useMemo(() => [...examples.map((e) => e.text), ...lines], [examples, lines]);
  const { results, timedOut } = useMatches(regex, all);
  const active = regex !== null && !timedOut;

  const backdrop = useRef<HTMLDivElement>(null);
  const gutter = useRef<HTMLDivElement>(null);
  const syncScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    const { scrollTop, scrollLeft } = e.currentTarget;
    backdrop.current!.style.transform = `translate(${-scrollLeft}px, ${-scrollTop}px)`;
    gutter.current!.style.transform = `translateY(${-scrollTop}px)`;
  };

  const tested = active ? lines.filter((l) => l !== "" && results.has(l)) : [];
  const matching = tested.filter((l) => results.get(l)!.ok).length;
  const checked = active ? examples.filter((e) => results.has(e.text)) : [];
  const passing = checked.filter((e) => results.get(e.text)!.ok === e.expected).length;

  return (
    <div className="space-y-5">
      {timedOut && (
        <p role="alert" className="text-sm text-rose-600">
          Testing stopped after {TIMEOUT_MS / 1000} s: this regex may backtrack catastrophically (ReDoS) on these strings.
        </p>
      )}

      {examples.length > 0 && (
        <div>
          <div className="mb-2 flex items-baseline justify-between">
            <span className={LABEL}>
              Examples <span className="font-normal normal-case">— checked against what you expect</span>
            </span>
            {checked.length === examples.length && (
              <span
                className={`text-xs font-medium tabular-nums ${passing === examples.length ? "text-emerald-600" : "text-rose-600"}`}
              >
                {passing} / {examples.length} pass
              </span>
            )}
          </div>
          <ul className="divide-y divide-zinc-200 overflow-hidden rounded-lg border border-zinc-300 bg-white font-mono text-sm dark:divide-zinc-800 dark:border-zinc-700 dark:bg-zinc-900">
            {examples.map((e, i) => {
              const result = active ? results.get(e.text) : undefined;
              const pass = result && result.ok === e.expected;
              return (
                <li key={i} className="flex items-center gap-3 px-3 py-1">
                  <span
                    aria-label={result ? (pass ? "pass" : "fail") : undefined}
                    className={`w-4 shrink-0 text-center text-xs ${pass ? "text-emerald-600" : "text-rose-500"}`}
                  >
                    {result ? (pass ? "✓" : "✗") : ""}
                  </span>
                  <span className="w-28 shrink-0 font-sans text-xs text-zinc-500">
                    {e.expected ? "should match" : "should not match"}
                  </span>
                  <span className={`min-w-0 flex-1 overflow-x-auto ${ROW}`}>
                    <Highlighted parts={segments(e.text, result)} />
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div>
        <div className="mb-2 flex items-baseline justify-between">
          <label htmlFor="test" className={LABEL}>
            Test strings <span className="font-normal normal-case">— one per line, tested with RegExp.prototype.test</span>
          </label>
          {tested.length > 0 && (
            <span className="text-xs text-zinc-500 tabular-nums">
              {matching} / {tested.length} lines match
            </span>
          )}
        </div>
        <div className="flex h-80 overflow-hidden rounded-lg border border-zinc-300 bg-white font-mono text-sm focus-within:border-sky-500 focus-within:ring-2 focus-within:ring-sky-500/20 dark:border-zinc-700 dark:bg-zinc-900">
          {/* Neutre : « correspond » n'est ni un succès ni un échec sans résultat attendu (voir Examples). */}
          <div className="w-8 shrink-0 overflow-hidden border-r border-zinc-200 bg-zinc-50 py-3 text-center select-none dark:border-zinc-800 dark:bg-zinc-950">
            <div ref={gutter}>
              {lines.map((l, i) => {
                const r = active && l !== "" ? results.get(l) : undefined;
                return (
                  <div
                    key={i}
                    title={r ? (r.ok ? "matches" : "does not match") : undefined}
                    className={`${ROW} text-xs ${r?.ok ? "text-sky-600 dark:text-sky-400" : "text-zinc-400"}`}
                  >
                    {r ? (r.ok ? "●" : "○") : ""}
                  </div>
                );
              })}
            </div>
          </div>
          <div className="relative min-w-0 flex-1 overflow-hidden">
            <div ref={backdrop} aria-hidden className="pointer-events-none absolute top-0 left-0 w-max min-w-full py-3 text-transparent">
              {lines.map((l, i) => {
                const r = active && l !== "" ? results.get(l) : undefined;
                return (
                  <div key={i} className={`${ROW} px-3 ${r?.ok ? "bg-sky-500/8" : ""}`}>
                    <Highlighted parts={segments(l, r)} transparent />
                  </div>
                );
              })}
            </div>
            <textarea
              id="test"
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onScroll={syncScroll}
              spellCheck={false}
              wrap="off"
              className="relative h-full w-full resize-none overflow-auto bg-transparent px-3 py-3 leading-6 whitespace-pre text-zinc-900 caret-zinc-900 outline-none dark:text-zinc-100 dark:caret-zinc-100"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
