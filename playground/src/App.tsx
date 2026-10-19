import { useEffect, useMemo, useRef, useState } from "react";
import { parseAnswer, type Request } from "./prompt";
import { HowItWorks } from "./HowItWorks";
import { TestArea } from "./TestArea";
import type { Dtype, FromWorker, ModelSource, ToWorker } from "./worker";

/** `owner/repo@revision` (variable VITE_HUB_*, voir .env.example) → source Hub ; absente → public/models/ en local. */
function source(hub: string | undefined): ModelSource {
  if (!hub) return { kind: "local" };
  const [repo, revision] = hub.split("@");
  if (!repo || !revision) throw new Error(`Expected owner/repo@revision, got "${hub}"`);
  return { kind: "hub", repo, revision };
}

// `files` : dossier de l'export (public/models/<files>/ ou repo du Hub) ; `dtype` : variante des poids dans onnx/.
const ALL_MODELS: { id: string; label: string; size: string; files: string; dtype: Dtype; source: ModelSource }[] = [
  // Meilleur 0.6B au benchmark indépendant (GPT) : SFT sur données enrichies, étape 9 ter.
  {
    id: "sft-0.6b-plus",
    label: "SFT Qwen3-0.6B",
    size: "1.6 GB",
    files: "sft-0.6b-plus",
    dtype: "fp16",
    source: source(import.meta.env.VITE_HUB_SFT_0_6B_PLUS),
  },
  // Même modèle quantizé en 4 bits (scripts/quantize_models.py) : held-out inchangé, ~10 points de moins sur les
  // cas difficiles du benchmark. Local uniquement pour l'instant : absent du build de production.
  {
    id: "sft-0.6b-plus-q4f16",
    label: "SFT Qwen3-0.6B, light (4-bit)",
    size: "0.65 GB",
    files: "sft-0.6b-plus",
    dtype: "q4f16",
    source: { kind: "local" },
  },
  // { id: "sft-1.7b", label: "SFT Qwen3-1.7B", size: "4.7 GB", source: source(import.meta.env.VITE_HUB_SFT_1_7B) },
];
// En production, public/models/ n'est pas publié : seuls les modèles hébergés sur le Hub sont proposés.
const MODELS = import.meta.env.PROD
  ? ALL_MODELS.filter((m) => m.source.kind === "hub")
  : ALL_MODELS;

const ua = navigator.userAgent;
const isFirefox = /firefox|fxios/i.test(ua);
const isSafari = /safari/i.test(ua) && !/chrome|chromium|crios|edg|android/i.test(ua);
const UNSUPPORTED = isFirefox || isSafari || !("gpu" in navigator);

const lines = (text: string) => text.split("\n").filter((l) => l !== "");

type Status =
  | { kind: "unloaded" }
  | { kind: "loading"; loaded: number; total: number }
  | { kind: "idle" }
  | { kind: "generating" }
  | { kind: "done"; ms: number }
  | { kind: "error"; message: string };

export default function App() {
  const [model, setModel] = useState(MODELS[0].id);
  const [ready, setReady] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "unloaded" });
  const [description, setDescription] = useState(
    "a hexadecimal color code: a # followed by 3 or 6 hex digits, any case",
  );
  const [showExamples, setShowExamples] = useState(false);
  const [match, setMatch] = useState("");
  const [noMatch, setNoMatch] = useState("");
  const [output, setOutput] = useState("");
  const [test, setTest] = useState("#fff\n#A1B2C3\n#abcd\nfff\n#12345g\nbody { color: #c0ffee; }");
  const [copied, setCopied] = useState(false);
  const [autoGenerate, setAutoGenerate] = useState(true);
  // Entrée (description + exemples) de la dernière génération lancée, pour détecter une regex périmée.
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);
  const worker = useRef<Worker>(null);
  const requestId = useRef(0);

  useEffect(() => {
    if (UNSUPPORTED) return;
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (event: MessageEvent<FromWorker>) => {
      const m = event.data;
      if (m.type === "progress") setStatus({ kind: "loading", loaded: m.loaded, total: m.total });
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

  // Chargement à la demande : le modèle pèse plusieurs Go.
  const load = () => {
    setReady(null);
    setOutput("");
    setGeneratedKey(null);
    setStatus({ kind: "loading", loaded: 0, total: 0 });
    const { files, dtype, source } = MODELS.find((m) => m.id === model)!;
    worker.current?.postMessage({ type: "load", model, files, dtype, source } satisfies ToWorker);
  };

  const selectModel = (id: string) => {
    setModel(id);
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

  const copy = async () => {
    await navigator.clipboard.writeText(output);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="min-h-screen bg-zinc-100 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {UNSUPPORTED && (
        <div
          role="alert"
          className="bg-amber-400 px-4 py-2.5 text-center text-sm font-medium text-amber-950"
        >
          {isFirefox || isSafari
            ? "Firefox and Safari are not supported. Please use a Chromium-based browser (Chrome, Edge, Brave, Arc…)."
            : "WebGPU is not available in this browser. Please use a recent Chromium-based browser."}
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
            <ModelStatus status={status} />
            <select
              value={model}
              onChange={(e) => selectModel(e.target.value)}
              disabled={UNSUPPORTED || loading}
              aria-label="Model"
              className="min-w-0 flex-1 rounded-md sm:flex-none border border-zinc-300 bg-white px-2.5 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-800"
            >
              {MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label} · {m.size}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={load}
              disabled={UNSUPPORTED || loading || loaded}
              className="inline-flex min-w-28 items-center justify-center gap-2 rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:text-zinc-600 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-400"
            >
              {loading && <Spinner />}
              {loading ? "Loading…" : loaded ? "Loaded" : "Load model"}
            </button>
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
          <button
            type="button"
            onClick={() => setShowExamples((v) => !v)}
            aria-expanded={showExamples}
            className="mt-1.5 text-sm text-sky-700 hover:underline dark:text-sky-400"
          >
            {showExamples ? "− Remove examples" : "+ Add example strings (optional)"}
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
            <output
              aria-live="polite"
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
              {copied ? "Copied" : "Copy"}
            </button>
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
          <TestArea value={test} onChange={setTest} regex={compiled.regex} />
        </section>

        <HowItWorks />

        <p className="text-xs text-zinc-500">
          Models run locally in your browser. Once loaded, their weights are cached for this origin
          (scheme, host and port), up to {MODELS.map((m) => m.size).join(" + ")}: clear this site's
          data to free the space.
        </p>
      </main>
    </div>
  );
}

function ModelStatus({ status }: { status: Status }) {
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
            ? `${(status.loaded / 1e9).toFixed(2)} / ${(status.total / 1e9).toFixed(1)} GB`
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
      {status.kind === "generating" ? "Generating" : "Ready · WebGPU"}
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
