import { useEffect, useMemo, useRef, useState } from "react";
import { cacheKey } from "./cache";
import { parseAnswer, type Request } from "./prompt";
import { type Benchmark, HowItWorks } from "./HowItWorks";
import { type Example, TestArea } from "./TestArea";
import { pickVariant, variantLabel } from "./variant";
import type { FromWorker, ModelSource, ToWorker, Variant } from "./worker";

/** `owner/repo@revision` (variable VITE_HUB_*, voir .env.example) → source Hub ; absente → public/models/ en local. */
function source(hub: string | undefined): ModelSource {
  if (!hub) return { kind: "local" };
  const [repo, revision] = hub.split("@");
  if (!repo || !revision) throw new Error(`Expected owner/repo@revision, got "${hub}"`);
  return { kind: "hub", repo, revision };
}

const REPO_URL = "https://github.com/mdubourg001/regex-generator-local-post-training";

// `files` : dossier de l'export (public/models/<files>/ ou repo du Hub) ; `sizes` : téléchargement par variante ;
// `benchmark` : succès complets au benchmark regex101 par difficulté, affichés dans « Limits ».
const ALL_MODELS: {
  id: string;
  label: string;
  files: string;
  sizes: Partial<Record<Variant["dtype"], string>>;
  source: ModelSource;
  benchmark: Benchmark;
}[] = [
  // SFT de Qwen3-1.7B (configs/sft.yaml), q4f16 seulement (scripts/export_qwen3_q4f16.py) : meilleur au benchmark
  // regex101 (25,8 % en 4 bits) et ~4 fois plus rapide sur WebGPU (graphe fusionné, sans If/Scan), mais 1,43 Go.
  // Premier de la liste : proposé par défaut.
  {
    id: "sft-1.7b",
    label: "SFT Qwen3-1.7B",
    files: "sft-1.7b",
    sizes: { q4f16: "1.43 GB" },
    source: source(import.meta.env.VITE_HUB_SFT_QWEN3_1_7B),
    // MLX 4 bits (results/sft-1.7b-mlx-q4-real.jsonl), en approximation de l'export q4f16, évalué sur 40 cas seulement.
    benchmark: { simple: [73, 162], medium: [28, 199], hard: [2, 39] },
  },
  // SFT de Qwen3.5-0.8B (configs/sft-qwen3.5-0.8b.yaml), quantizé en 4 bits : remplace le Qwen3-0.6B en fp16
  // (26,0 % contre 22,8 % au benchmark regex101, pour un téléchargement 2,5 fois plus léger).
  {
    id: "sft-qwen3.5-0.8b",
    label: "SFT Qwen3.5-0.8B",
    files: "sft-qwen3.5-0.8b",
    sizes: { q4f16: "584 MB", q4: "648 MB" },
    source: source(import.meta.env.VITE_HUB_SFT_QWEN3_5_0_8B),
    // Export q4f16 (results/onnx-qwen3.5-0.8b-q4f16-real.jsonl).
    benchmark: { simple: [67, 162], medium: [26, 199], hard: [3, 39] },
  },
];
// En production, public/models/ n'est pas publié : seuls les modèles hébergés sur le Hub sont proposés.
const MODELS = import.meta.env.PROD
  ? ALL_MODELS.filter((m) => m.source.kind === "hub")
  : ALL_MODELS;

// Model cards des modèles hébergés sur le Hub (aucune en dev, avec les exports locaux).
const CARDS = MODELS.flatMap(({ id, label, source }) =>
  source.kind === "hub" ? [{ id, label, url: `https://huggingface.co/${source.repo}` }] : [],
);

const ua = navigator.userAgent;
const isFirefox = /firefox|fxios/i.test(ua);
const isSafari = /safari/i.test(ua) && !/chrome|chromium|crios|edg|android/i.test(ua);
// Sans WebGPU, le modèle tourne sur WASM (CPU), plus lentement. Firefox et Safari restent exclus : non testés.
const UNSUPPORTED = isFirefox || isSafari;

const lines = (text: string) => text.split("\n").filter((l) => l !== "");

/** Premier modèle (le préféré, sinon dans l'ordre de la liste) qui existe dans la variante du navigateur. */
function resolveModel(preferred: string, variant: Variant | null) {
  const available = (m: (typeof MODELS)[number]) => !variant || m.sizes[variant.dtype] !== undefined;
  const wanted = MODELS.find((m) => m.id === preferred);
  return (wanted && available(wanted) ? wanted : (MODELS.find(available) ?? MODELS[0])).id;
}

// Connexion mobile, lente ou en économie de données (Network Information API, Chromium) : on confirme le téléchargement.
type NetworkInformation = { saveData?: boolean; type?: string; effectiveType?: string };
function meteredConnection() {
  const c = (navigator as Navigator & { connection?: NetworkInformation }).connection;
  return !!c && (c.saveData === true || c.type === "cellular" || ["slow-2g", "2g", "3g"].includes(c.effectiveType ?? ""));
}

async function isCached(files: string, version: string) {
  try {
    return await caches.has(cacheKey(files, version));
  } catch {
    return false;
  }
}

type Preset = { label: string; description: string; test: string[]; match?: string[]; noMatch?: string[] };
// Exemples cliquables ; le premier est l'état initial de la page.
const PRESETS: Preset[] = [
  {
    label: "Hex color",
    description: "a hexadecimal color code: a # followed by 3 or 6 hex digits, any case",
    test: ["#fff", "#A1B2C3", "#abcd", "fff", "#12345g", "body { color: #c0ffee; }"],
  },
  {
    label: "ISO date",
    description: "a date in YYYY-MM-DD format, with a month from 01 to 12 and a day from 01 to 31",
    test: ["2024-02-29", "1999-12-31", "2024-13-01", "2024-1-05", "24-01-05", "2024-01-32"],
  },
  {
    label: "IPv4 address",
    description: "an IPv4 address: four numbers from 0 to 255 separated by dots, without leading zeros",
    test: ["192.168.1.1", "0.0.0.0", "255.255.255.255", "256.1.1.1", "192.168.01.1", "1.2.3"],
  },
  {
    label: "Email",
    description:
      "a simple email address: letters, digits, dots, underscores or hyphens, then an @, then a domain name containing at least one dot",
    test: ["jane.doe@example.com", "j_d-42@mail.co.uk", "jane@localhost", "@example.com", "jane doe@example.com"],
  },
  {
    label: "URL slug",
    description: "a URL slug: groups of lowercase letters and digits separated by single hyphens",
    test: ["hello-world", "post-2024", "hello--world", "-hello", "Hello-World", "hello_world"],
  },
  {
    label: "24h time (with examples)",
    description: "a time in 24-hour format",
    match: ["00:00", "09:30", "23:59"],
    noMatch: ["24:00", "9:30", "12:60"],
    test: ["13:45", "7:05", "19:5", "1345"],
  },
];

// État partagé par l'URL : description (q), exemples (match, nomatch), tests (test) et modèle choisi (model). Les
// paramètres absents valent l'état initial de la page.
const URL_STATE = new URLSearchParams(location.search);
const DEFAULT_TEST = PRESETS[0].test.join("\n");

type Status =
  | { kind: "unloaded" }
  | { kind: "loading"; loaded: number; total: number }
  | { kind: "idle" }
  | { kind: "generating" }
  | { kind: "done"; ms: number }
  | { kind: "error"; message: string };

export default function App() {
  // Modèle choisi (URL ou sélecteur) ; `model` est celui qui est utilisé, le premier disponible s'il n'existe pas
  // dans la variante du navigateur (le 1.7B, en q4f16 seulement, sans WebGPU ou sans shader-f16).
  const [preferred, setPreferred] = useState(URL_STATE.get("model") ?? MODELS[0].id);
  const [ready, setReady] = useState<string | null>(null);
  // Variante (dtype, backend) détectée au démarrage : null tant que la détection de WebGPU n'a pas abouti.
  const [variant, setVariant] = useState<Variant | null>(null);
  const model = resolveModel(preferred, variant);
  const selected = MODELS.find((m) => m.id === model)!;
  const [status, setStatus] = useState<Status>({ kind: "unloaded" });
  const [description, setDescription] = useState(URL_STATE.get("q") ?? PRESETS[0].description);
  const [showExamples, setShowExamples] = useState(URL_STATE.has("match") || URL_STATE.has("nomatch"));
  const [match, setMatch] = useState(URL_STATE.get("match") ?? "");
  const [noMatch, setNoMatch] = useState(URL_STATE.get("nomatch") ?? "");
  const [output, setOutput] = useState("");
  const [test, setTest] = useState(URL_STATE.get("test") ?? DEFAULT_TEST);
  const [copied, setCopied] = useState<"copied" | "failed" | null>(null);
  // Modèle dont le téléchargement attend une confirmation (connexion mobile ou lente).
  const [confirming, setConfirming] = useState<string | null>(null);
  const [autoGenerate, setAutoGenerate] = useState(true);
  // Entrée (description + exemples) de la dernière génération lancée, pour détecter une regex périmée.
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);
  const worker = useRef<Worker>(null);
  const requestId = useRef(0);

  useEffect(() => {
    if (UNSUPPORTED) return;
    pickVariant().then(setVariant);
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (event: MessageEvent<FromWorker>) => {
      const m = event.data;
      if (m.type === "progress") setStatus({ kind: "loading", loaded: m.loaded, total: m.total });
      else if (m.type === "unloaded") {
        setReady(null);
        setStatus({ kind: "unloaded" });
      }
      else if (m.type === "ready") {
        setReady(m.model);
        setStatus({ kind: "idle" });
      } else if (m.type === "error") setStatus({ kind: "error", message: m.message });
      else if (m.id === requestId.current) {
        setOutput(m.text);
        if (m.type === "done") setStatus({ kind: "done", ms: m.ms });
      }
    };
    worker.current = w;
    return () => w.terminate();
  }, []);

  // L'état est repris dans l'URL (remplacée, pas ajoutée à l'historique), pour partager un cas.
  useEffect(() => {
    const timer = setTimeout(() => {
      const params = new URLSearchParams();
      if (description !== PRESETS[0].description) params.set("q", description);
      if (showExamples && match) params.set("match", match);
      if (showExamples && noMatch) params.set("nomatch", noMatch);
      if (test !== DEFAULT_TEST) params.set("test", test);
      if (preferred !== MODELS[0].id) params.set("model", preferred);
      const query = params.toString();
      history.replaceState(null, "", query ? `?${query}` : location.pathname);
    }, 300);
    return () => clearTimeout(timer);
  }, [description, showExamples, match, noMatch, test, preferred]);

  // Chargement à la demande : le modèle pèse plusieurs Go.
  const load = async () => {
    if (!variant) return;
    const { files, source, sizes } = selected;
    if (!sizes[variant.dtype]) {
      setStatus({ kind: "error", message: `${model} only exists in ${Object.keys(sizes).join(", ")} (WebGPU with shader-f16)` });
      return;
    }
    // Sur une connexion mobile ou lente, un premier clic demande confirmation, sauf si les poids sont déjà en cache.
    if (source.kind === "hub" && confirming !== model && meteredConnection() && !(await isCached(files, source.revision))) {
      setConfirming(model);
      return;
    }
    setConfirming(null);
    setReady(null);
    setOutput("");
    setGeneratedKey(null);
    setStatus({ kind: "loading", loaded: 0, total: 0 });
    const known = MODELS.map((m) => m.files);
    worker.current?.postMessage({ type: "load", model, files, variant, source, known } satisfies ToWorker);
  };

  // Libère la mémoire GPU et supprime les poids du modèle chargé (même s'il n'est plus sélectionné) du cache du
  // navigateur : son prochain chargement les retélécharge. Les poids des autres modèles restent en cache.
  const unload = () => {
    requestId.current++; // ignore la fin d'une génération en cours
    setReady(null);
    setOutput("");
    setGeneratedKey(null);
    setStatus({ kind: "unloaded" });
    worker.current?.postMessage({ type: "unload" } satisfies ToWorker);
  };

  const selectModel = (id: string) => {
    setPreferred(id);
    setConfirming(null);
    requestId.current++; // ignore la fin d'une génération du modèle précédent
    if (ready !== id) {
      setOutput("");
      setGeneratedKey(null);
    }
    setStatus(ready === id ? { kind: "idle" } : { kind: "unloaded" });
  };

  const loaded = ready === model;
  const request: Request = {
    description: description.trim(),
    match: showExamples ? lines(match) : [],
    noMatch: showExamples ? lines(noMatch) : [],
  };
  const requestKey = JSON.stringify(request);

  const generate = () => {
    if (!loaded || !request.description) return;
    const id = ++requestId.current;
    setStatus({ kind: "generating" });
    setOutput("");
    setGeneratedKey(requestKey);
    worker.current?.postMessage({ type: "generate", id, ...request } satisfies ToWorker);
  };

  // Génération automatique : debounce de 400 ms entre la frappe et l'appel au modèle.
  useEffect(() => {
    if (!autoGenerate || requestKey === generatedKey) return;
    const timer = setTimeout(generate, 400);
    return () => clearTimeout(timer);
  }, [autoGenerate, loaded, requestKey, generatedKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const generating = status.kind === "generating";
  const loading = status.kind === "loading";
  const outOfSync =
    !autoGenerate && !generating && generatedKey !== null && requestKey !== generatedKey;
  const parsed = useMemo(() => (generating ? null : parseAnswer(output)), [output, generating]);
  const compiled = useMemo((): { regex: RegExp | null; error: string | null } => {
    if (!parsed) return { regex: null, error: null };
    try {
      return { regex: new RegExp(parsed[0], parsed[1]), error: null };
    } catch (e) {
      return { regex: null, error: (e as Error).message };
    }
  }, [parsed]);

  const examples = useMemo(
    (): Example[] =>
      showExamples
        ? [...lines(match).map((text) => ({ text, expected: true })), ...lines(noMatch).map((text) => ({ text, expected: false }))]
        : [],
    [showExamples, match, noMatch],
  );

  // regex101 teste une seule chaîne : les flags g et m y font tester chaque ligne séparément, comme ici.
  const regex101 =
    parsed && compiled.regex
      ? "https://regex101.com/?" +
        Object.entries({
          regex: parsed[0],
          flags: [...new Set(parsed[1] + "gm")].join(""),
          flavor: "javascript",
          testString: [...examples.map((e) => e.text), ...lines(test)].join("\n"),
        })
          .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
          .join("&")
      : null;

  const applyPreset = (preset: Preset) => {
    setDescription(preset.description);
    setTest(preset.test.join("\n"));
    setMatch(preset.match?.join("\n") ?? "");
    setNoMatch(preset.noMatch?.join("\n") ?? "");
    setShowExamples(preset.match !== undefined || preset.noMatch !== undefined);
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(output);
      setCopied("copied");
    } catch {
      setCopied("failed");
    }
    setTimeout(() => setCopied(null), 1200);
  };

  return (
    <div className="min-h-screen bg-zinc-100 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {UNSUPPORTED && (
        <div
          role="alert"
          className="bg-amber-400 px-4 py-2.5 text-center text-sm font-medium text-amber-950"
        >
          Firefox and Safari are not supported. Please use a Chromium-based browser (Chrome, Edge, Brave, Arc…).
        </div>
      )}

      <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3">
          <div>
            <h1 className="font-mono text-lg leading-tight font-semibold tracking-tight">
              <span className="text-zinc-400">/</span>regex<span className="text-zinc-400">/</span>
              <span className="text-sky-600">vibes</span>
            </h1>
            <p className="text-xs text-zinc-500 pt-2">Generate regexes from natural language.</p>
          </div>
          <div className="flex w-full flex-wrap items-center gap-3 sm:ml-auto sm:w-auto">
            <ModelStatus status={status} variant={variant} />
            <select
              value={model}
              onChange={(e) => selectModel(e.target.value)}
              disabled={UNSUPPORTED || loading}
              aria-label="Model"
              className="min-w-0 flex-1 rounded-md sm:flex-none border border-zinc-300 bg-white px-2.5 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-800"
            >
              {MODELS.map((m) => (
                <option key={m.id} value={m.id} disabled={!!variant && !m.sizes[variant.dtype]}>
                  {m.label}
                  {variant &&
                    (m.sizes[variant.dtype]
                      ? ` · ${variant.dtype} · ${m.sizes[variant.dtype]}`
                      : ` · ${Object.keys(m.sizes).join(", ")} only, unavailable here`)}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={load}
              disabled={UNSUPPORTED || !variant || loading || loaded}
              title={confirming === model ? "You seem to be on a mobile, slow or data-saving connection" : undefined}
              className={`inline-flex min-w-28 items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:text-zinc-600 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-400 ${confirming === model ? "bg-amber-600 hover:bg-amber-700" : "bg-sky-600 hover:bg-sky-700"}`}
            >
              {loading && <Spinner />}
              {loading
                ? "Loading…"
                : loaded
                  ? "Loaded"
                  : confirming === model && variant
                    ? `Download ${selected.sizes[variant.dtype]} anyway`
                    : "Load model"}
            </button>
            {ready !== null && (
              <button
                type="button"
                onClick={unload}
                title={`Free GPU memory and delete the cached weights of ${MODELS.find((m) => m.id === ready)?.label} from this browser`}
                className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                Unload
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-6 px-4 py-6">
        <section>
          <label
            htmlFor="description"
            className="mb-2 block text-xs font-semibold tracking-wide text-zinc-500 uppercase"
          >
            Description <span className="font-normal normal-case">— in English</span>
          </label>
          <textarea
            id="description"
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                generate();
              }
            }}
            placeholder="e.g. a French postal code: exactly five digits"
            className="w-full resize-y rounded-lg border border-zinc-300 bg-white px-3 py-2.5 outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 dark:border-zinc-700 dark:bg-zinc-900"
          />
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <span className="mr-0.5 text-xs text-zinc-500">Try:</span>
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => applyPreset(preset)}
                aria-pressed={description === preset.description}
                className="rounded-full border border-zinc-300 bg-white px-2.5 py-0.5 text-xs text-zinc-700 hover:border-sky-500 hover:text-sky-700 aria-pressed:border-sky-500 aria-pressed:bg-sky-50 aria-pressed:text-sky-700 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:text-sky-400 dark:aria-pressed:bg-sky-950 dark:aria-pressed:text-sky-400"
              >
                {preset.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setShowExamples((v) => !v)}
            aria-expanded={showExamples}
            className="mt-2 text-sm text-sky-700 hover:underline dark:text-sky-400"
          >
            {/* Masqués, les exemples sont gardés mais exclus du prompt et des tests. */}
            {showExamples ? "− Don't use examples" : "+ Add example strings (optional)"}
          </button>
          {showExamples && (
            <div className="mt-3 grid gap-4 sm:grid-cols-2">
              <ExampleList id="match" label="Should match" value={match} onChange={setMatch} />
              <ExampleList
                id="no-match"
                label="Should not match"
                value={noMatch}
                onChange={setNoMatch}
              />
            </div>
          )}
        </section>

        <section>
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-xs font-semibold tracking-wide text-zinc-500 uppercase">
              Regular expression
            </span>
            <div className="flex items-center gap-3">
              {status.kind === "done" && (
                <span className="text-xs text-zinc-500 tabular-nums">
                  {Math.round(status.ms)} ms
                </span>
              )}
              <Toggle label="Auto-generate" checked={autoGenerate} onChange={setAutoGenerate} />
              <button
                type="button"
                onClick={generate}
                disabled={!loaded || generating || !request.description}
                title="⌘/Ctrl + Enter in the description"
                className="rounded-md bg-sky-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Generate
              </button>
            </div>
          </div>
          <div
            aria-busy={generating}
            className={`flex items-center gap-2 rounded-lg border bg-zinc-50 py-2 pr-2 pl-3 transition-colors dark:bg-zinc-900/60 ${generating ? "border-sky-400 dark:border-sky-600" : outOfSync ? "border-amber-400 dark:border-amber-600" : "border-zinc-300 dark:border-zinc-700"}`}
          >
            {generating && <Spinner className="text-sky-600 dark:text-sky-400" />}
            {/* Annonce la regex finale seulement : sur la sortie elle-même, chaque token serait lu pendant le streaming. */}
            <span className="sr-only" aria-live="polite">
              {parsed && !generating ? `Generated regex: /${parsed[0]}/${parsed[1]}` : ""}
            </span>
            <output
              className="min-h-6 min-w-0 flex-1 overflow-x-auto font-mono text-[15px] leading-6 whitespace-pre"
            >
              {parsed && !generating ? (
                <>
                  <span className="text-zinc-400">/</span>
                  {parsed[0]}
                  <span className="text-zinc-400">/</span>
                  <span className="text-fuchsia-600 dark:text-fuchsia-400">{parsed[1]}</span>
                </>
              ) : generating ? (
                <span className="text-zinc-500">{output || "Generating…"}</span>
              ) : !loaded ? (
                <span className="font-sans text-sm text-zinc-400">
                  Load a model to generate a regex.
                </span>
              ) : !output && !autoGenerate ? (
                <span className="font-sans text-sm text-zinc-400">
                  Click Generate to get a regex.
                </span>
              ) : (
                output
              )}
            </output>
            <button
              type="button"
              onClick={copy}
              disabled={!output || generating}
              className="shrink-0 rounded-md px-2.5 py-1 text-sm text-zinc-600 hover:bg-zinc-200 disabled:opacity-40 dark:text-zinc-300 dark:hover:bg-zinc-800"
            >
              {copied === "copied" ? "Copied" : copied === "failed" ? "Copy failed" : "Copy"}
            </button>
            {regex101 ? (
              <a
                href={regex101}
                target="_blank"
                rel="noopener noreferrer"
                title="Opens the regex and all strings in regex101, with flags g and m to test each line separately"
                className="shrink-0 rounded-md px-2.5 py-1 text-sm whitespace-nowrap text-zinc-600 hover:bg-zinc-200 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                Open in regex101 ↗
              </a>
            ) : (
              <span className="shrink-0 px-2.5 py-1 text-sm whitespace-nowrap text-zinc-600 opacity-40 dark:text-zinc-300">
                Open in regex101 ↗
              </span>
            )}
          </div>
          {outOfSync && (
            <p className="mt-1.5 text-sm text-amber-700 dark:text-amber-400">
              Out of sync: the description or examples changed since this regex was generated.
            </p>
          )}
          {!generating && output && !parsed && (
            <p className="mt-1.5 text-sm text-rose-600">
              Malformed answer: the model did not return a single regex literal.
            </p>
          )}
          {compiled.error && (
            <p className="mt-1.5 text-sm text-rose-600">Invalid regex: {compiled.error}</p>
          )}
        </section>

        <section>
          <TestArea value={test} onChange={setTest} regex={compiled.regex} examples={examples} />
        </section>

        <HowItWorks model={selected.label} benchmark={selected.benchmark} />

        <p className="text-xs text-zinc-500">
          The model runs locally in your browser
          {variant && (
            <>
              {" "}
              ({variantLabel(variant)}
              {variant.device === "wasm" && ": no WebGPU here, so generation is slower"})
            </>
          )}
          . Once loaded, the weights of each model are cached for this origin (scheme, host and port): Unload
          frees the memory and deletes the weights of the loaded model.
        </p>
        <p className="text-xs text-zinc-500">
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="text-sky-700 hover:underline dark:text-sky-400">
            Source code on GitHub
          </a>
          {CARDS.length > 0 && " · Model cards on Hugging Face: "}
          {CARDS.map(({ id, label, url }, i) => (
            <span key={id}>
              {i > 0 && ", "}
              <a href={url} target="_blank" rel="noopener noreferrer" className="text-sky-700 hover:underline dark:text-sky-400">
                {label}
              </a>
            </span>
          ))}
        </p>
      </main>
    </div>
  );
}

function ModelStatus({ status, variant }: { status: Status; variant: Variant | null }) {
  if (status.kind === "unloaded")
    return <span className="text-xs whitespace-nowrap text-zinc-500">Not loaded</span>;
  if (status.kind === "loading") {
    // Sans total connu (avant la première réponse HTTP), la barre est indéterminée.
    const pct = status.total ? (100 * status.loaded) / status.total : null;
    return (
      <div className="flex items-center gap-2 text-xs text-zinc-500 tabular-nums">
        <div className="relative h-1.5 w-28 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
          {pct === null ? (
            <div className="absolute inset-y-0 w-1/3 animate-[indeterminate_1.2s_ease-in-out_infinite] bg-sky-500" />
          ) : (
            <div className="h-full bg-sky-500 transition-[width]" style={{ width: `${pct}%` }} />
          )}
        </div>
        {pct === null
          ? "Fetching…"
          : pct < 100
            ? `${Math.round(status.loaded / 1e6)} / ${Math.round(status.total / 1e6)} MB`
            : "Initializing…"}
      </div>
    );
  }
  if (status.kind === "error") {
    return (
      <span title={status.message} className="max-w-xs truncate text-xs text-rose-600">
        Error: {status.message}
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-xs whitespace-nowrap text-zinc-500">
      <span
        className={`size-2 rounded-full ${status.kind === "generating" ? "animate-pulse bg-sky-500" : "bg-emerald-500"}`}
      />
      {status.kind === "generating" ? "Generating" : `Ready · ${variant ? variantLabel(variant) : ""}`}
    </span>
  );
}

function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
      className={`size-4 shrink-0 animate-spin ${className}`}
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function Toggle(props: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
      {props.label}
      <button
        type="button"
        role="switch"
        aria-checked={props.checked}
        onClick={() => props.onChange(!props.checked)}
        className={`relative h-5 w-9 rounded-full transition-colors ${props.checked ? "bg-sky-600" : "bg-zinc-300 dark:bg-zinc-700"}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 size-4 rounded-full bg-white shadow transition-transform ${props.checked ? "translate-x-4" : ""}`}
        />
      </button>
    </label>
  );
}

function ExampleList(props: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label htmlFor={props.id} className="mb-1 block text-xs text-zinc-500">
        {props.label} <span className="text-zinc-400">— one per line</span>
      </label>
      <textarea
        id={props.id}
        rows={3}
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
        spellCheck={false}
        className="w-full resize-y rounded-lg border border-zinc-300 bg-white px-3 py-2 font-mono text-sm outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 dark:border-zinc-700 dark:bg-zinc-900"
      />
    </div>
  );
}
