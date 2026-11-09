/// <reference lib="webworker" />
// Tests des lignes hors du thread principal : une regex générée peut backtracker de façon catastrophique (ReDoS).
// Le worker ne peut pas s'interrompre lui-même, c'est la page qui le termine après un délai (voir useMatches.ts).

export type MatchRequest = { id: number; source: string; flags: string; lines: string[] };
/** `ok` : RegExp.prototype.test ; `ranges` : correspondances [début, fin[ de la copie en flag g (vides comprises). */
export type LineResult = { ok: boolean; ranges: [number, number][] };
export type MatchResponse = { id: number; results: LineResult[] };

self.onmessage = ({ data: { id, source, flags, lines } }: MessageEvent<MatchRequest>) => {
  const global = new RegExp(source, flags.includes("g") ? flags : flags + "g");
  const results = lines.map((line): LineResult => {
    // Une regex neuve par ligne : avec les flags g ou y, test() dépend de lastIndex.
    const ok = new RegExp(source, flags).test(line);
    const ranges: [number, number][] = [];
    global.lastIndex = 0;
    for (let m = global.exec(line); m && ranges.length < 1000; m = global.exec(line)) {
      ranges.push([m.index, m.index + m[0].length]);
      if (m[0] === "") global.lastIndex++; // correspondance vide : on avance pour ne pas boucler
    }
    return { ok, ranges };
  });
  self.postMessage({ id, results } satisfies MatchResponse);
};
