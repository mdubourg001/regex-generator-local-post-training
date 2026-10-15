import { useMemo, useRef } from "react";

type Segment = { text: string; match?: number };
type Line = { segments: Segment[]; ok: boolean | null; empty: boolean };

/** Découpe une ligne en segments, les correspondances de `global` étant numérotées (pour alterner les couleurs). */
function highlight(line: string, global: RegExp): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  let count = 0;
  global.lastIndex = 0;
  for (let m = global.exec(line); m && count < 1000; m = global.exec(line), count++) {
    if (m.index > last) segments.push({ text: line.slice(last, m.index) });
    segments.push({ text: m[0], match: count });
    last = m.index + m[0].length;
    if (m[0] === "") global.lastIndex++; // correspondance vide : on avance pour ne pas boucler
  }
  if (last < line.length) segments.push({ text: line.slice(last) });
  return segments;
}

function analyse(text: string, regex: RegExp | null): Line[] {
  const global = regex && new RegExp(regex.source, regex.flags.includes("g") ? regex.flags : regex.flags + "g");
  return text.split("\n").map((line) => {
    if (!regex || !global) return { segments: [{ text: line }], ok: null, empty: line === "" };
    // Une regex neuve par ligne : avec les flags g ou y, test() dépend de lastIndex.
    const ok = new RegExp(regex.source, regex.flags).test(line);
    return { segments: highlight(line, global), ok, empty: line === "" };
  });
}

const MARKS = ["bg-sky-300/70 dark:bg-sky-500/50", "bg-sky-200/70 dark:bg-sky-300/35"];
const ROW = "h-6 leading-6 whitespace-pre";

export function TestArea({ value, onChange, regex }: { value: string; onChange: (v: string) => void; regex: RegExp | null }) {
  const lines = useMemo(() => analyse(value, regex), [value, regex]);
  const backdrop = useRef<HTMLDivElement>(null);
  const gutter = useRef<HTMLDivElement>(null);
  const tested = lines.filter((l) => !l.empty && l.ok !== null);
  const passed = tested.filter((l) => l.ok).length;

  const syncScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    const { scrollTop, scrollLeft } = e.currentTarget;
    backdrop.current!.style.transform = `translate(${-scrollLeft}px, ${-scrollTop}px)`;
    gutter.current!.style.transform = `translateY(${-scrollTop}px)`;
  };

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <label htmlFor="test" className="text-xs font-semibold tracking-wide text-zinc-500 uppercase">
          Test strings <span className="font-normal normal-case">— one per line, tested with RegExp.prototype.test</span>
        </label>
        {regex && tested.length > 0 && (
          <span className="text-xs text-zinc-500 tabular-nums">
            {passed} / {tested.length} lines match
          </span>
        )}
      </div>
      <div className="flex h-80 overflow-hidden rounded-lg border border-zinc-300 bg-white font-mono text-sm focus-within:border-sky-500 focus-within:ring-2 focus-within:ring-sky-500/20 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="w-8 shrink-0 overflow-hidden border-r border-zinc-200 bg-zinc-50 py-3 text-center select-none dark:border-zinc-800 dark:bg-zinc-950">
          <div ref={gutter}>
            {lines.map((l, i) => (
              <div key={i} className={`${ROW} text-xs ${l.ok ? "text-emerald-600" : "text-rose-500"}`}>
                {l.empty || l.ok === null ? "" : l.ok ? "✓" : "✗"}
              </div>
            ))}
          </div>
        </div>
        <div className="relative min-w-0 flex-1 overflow-hidden">
          <div ref={backdrop} aria-hidden className="pointer-events-none absolute top-0 left-0 w-max min-w-full py-3 text-transparent">
            {lines.map((l, i) => (
              <div
                key={i}
                className={`${ROW} px-3 ${l.empty || l.ok === null ? "" : l.ok ? "bg-emerald-500/8" : "bg-rose-500/8"}`}
              >
                {l.segments.map((s, j) =>
                  s.match === undefined ? (
                    <span key={j}>{s.text}</span>
                  ) : s.text === "" ? (
                    <span key={j} className="relative inline-block h-full w-0 border-l-2 border-sky-500 align-top" />
                  ) : (
                    <mark key={j} className={`rounded-sm text-transparent ${MARKS[s.match % 2]}`}>
                      {s.text}
                    </mark>
                  ),
                )}
              </div>
            ))}
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
  );
}
