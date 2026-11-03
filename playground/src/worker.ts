/// <reference lib="webworker" />
// Inférence dans un worker : le chargement (plusieurs Go) et la génération ne bloquent pas l'interface.
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  env,
  InterruptableStoppingCriteria,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  TextStreamer,
} from "@huggingface/transformers";
// Runtime ONNX servi en local plutôt que depuis jsDelivr (défaut de transformers.js). Chemin relatif : le champ
// `exports` d'onnxruntime-web n'expose pas dist/. Même version que celle qu'embarque transformers.js.
import ortMjs from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs?url";
import ortWasm from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url";
import { buildPrompt, type Request } from "./prompt";

// Deux sources (voir ModelSource) : public/models/ servi par Vite en dev, ou un repo du Hub Hugging Face.
env.localModelPath = "/models/";
env.useBrowserCache = true;
env.backends.onnx.wasm!.wasmPaths = { mjs: ortMjs, wasm: ortWasm };

/** `local` : public/models/<id>/ (scripts/export_qwen35.py). `hub` : repo public, à une révision figée. */
export type ModelSource = { kind: "local" } | { kind: "hub"; repo: string; revision: string };
/** Variante des poids (onnx/*_<dtype>.onnx) et backend d'exécution, choisis selon le navigateur (voir variant.ts). */
export type Variant = { dtype: "q4" | "q4f16"; device: "webgpu" | "wasm" };
/** `model` : identifiant renvoyé dans `ready` ; `files` : dossier de l'export, partagé par ses variantes. */
export type ToWorker =
  | { type: "load"; model: string; files: string; variant: Variant; source: ModelSource }
  | { type: "unload" }
  | ({ type: "generate"; id: number } & Request);
export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; model: string }
  | { type: "unloaded"; freed: number }
  | { type: "token"; id: number; text: string }
  | { type: "done"; id: number; text: string; ms: number }
  | { type: "error"; message: string };

const post = (message: FromWorker) => self.postMessage(message);
const stopping = new InterruptableStoppingCriteria();
let current: { name: string; tokenizer: PreTrainedTokenizer; model: PreTrainedModel } | null = null;
let loading: Promise<void> = Promise.resolve();
let pending: (ToWorker & { type: "generate" }) | null = null;
let running = false;
let generation: Promise<unknown> = Promise.resolve();

/**
 * Un cache (Cache API) par dossier d'export et par version. Un seul modèle est proposé : tout autre cache du playground
 * (version précédente, ancien modèle comme le Qwen3-0.6B en fp16, 1,6 Go) est supprimé.
 */
async function useCacheFor(name: string, version: string) {
  env.cacheKey = `regex-playground/${name}/${version}`;
  try {
    for (const key of await caches.keys()) {
      if (key.startsWith("regex-playground/") && key !== env.cacheKey) await caches.delete(key);
    }
  } catch {
    // Cache API indisponible (navigation privée…) : transformers.js charge alors sans cache.
  }
}

/** Prépare env pour la source, et retourne l'identifiant et la révision à passer à from_pretrained. */
async function resolve(name: string, source: ModelSource): Promise<{ path: string; revision?: string }> {
  if (source.kind === "hub") {
    // Pas de repli local : en dev, Vite répondrait index.html pour /models/<owner>/<repo>/…
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    await useCacheFor(name, source.revision);
    return { path: source.repo, revision: source.revision };
  }
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  // Fichier absent : Vite répond index.html (repli SPA) au lieu d'un 404, d'où un test sur le type de contenu.
  const config = await fetch(`${env.localModelPath}${name}/config.json`, { cache: "no-store" });
  if (!config.headers.get("content-type")?.includes("json")) {
    throw new Error(`${name} is not exported: run \`uv run scripts/export_qwen35.py\``);
  }
  await useCacheFor(name, (await config.json()).playground_export_id ?? "unversioned");
  return { path: name };
}

/** Libère le modèle (mémoire GPU) et supprime ses poids du Cache API. Retourne le nombre de caches supprimés. */
async function unload(): Promise<number> {
  stopping.interrupt();
  await generation.catch(() => {});
  await current?.model.dispose();
  current = null;
  let freed = 0;
  try {
    for (const key of await caches.keys()) {
      if (key.startsWith("regex-playground/") && (await caches.delete(key))) freed++;
    }
  } catch {
    // Cache API indisponible : rien n'a été mis en cache.
  }
  return freed;
}

async function load(name: string, files: string, { dtype, device }: Variant, source: ModelSource) {
  stopping.interrupt();
  await generation.catch(() => {});
  await current?.model.dispose();
  current = null;
  const progress_callback = (info: { status: string; loaded?: number; total?: number }) => {
    if (info.status === "progress_total") post({ type: "progress", loaded: info.loaded!, total: info.total! });
  };
  // Cache par dossier d'export : les variantes (q4, q4f16) ont des fichiers distincts et partagent la clé.
  const { path, revision } = await resolve(files, source);
  // GatherBlockQuantized (embeddings q4) manque au build WASM d'onnxruntime-web : sur WASM, les mêmes poids q4
  // déquantifiés par des opérateurs standard (embed_tokens_uint8, voir scripts/export_qwen35.py).
  const dtypes = device === "wasm" ? { embed_tokens: "uint8" as const, decoder_model_merged: dtype } : dtype;
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(path, { revision }),
    AutoModelForCausalLM.from_pretrained(path, { revision, device, dtype: dtypes, progress_callback }),
  ]);
  current = { name, tokenizer, model };
  post({ type: "ready", model: name });
}

/** Une seule génération à la fois ; une requête arrivée entre-temps remplace celle en attente. */
async function drain() {
  if (running) return;
  running = true;
  while (pending) {
    const request = pending;
    pending = null;
    await loading;
    if (!current) break;
    const { tokenizer, model } = current;
    stopping.reset();
    const start = performance.now();
    // TextStreamer n'émet qu'aux espaces (heuristique « mots entiers ») : une regex n'en a souvent aucune, d'où un
    // décodage complet des tokens générés à chaque pas.
    const ids: bigint[] = [];
    let text = "";
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      callback_function: () => {},
      token_callback_function: (tokens: bigint[]) => {
        ids.push(...tokens);
        text = tokenizer.decode(ids, { skip_special_tokens: true });
        post({ type: "token", id: request.id, text });
      },
    });
    const inputs = tokenizer(buildPrompt(request), { add_special_tokens: false });
    // Greedy, comme scripts/eval.py.
    generation = model.generate({ ...inputs, max_new_tokens: 160, do_sample: false, streamer, stopping_criteria: stopping });
    await generation;
    if (!pending) post({ type: "done", id: request.id, text: text.trim(), ms: performance.now() - start });
  }
  running = false;
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  const fail = (error: unknown) => post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  if (message.type === "load") {
    loading = loading.then(() => load(message.model, message.files, message.variant, message.source)).catch(fail);
  } else if (message.type === "unload") {
    pending = null;
    loading = loading.then(unload).then((freed) => post({ type: "unloaded", freed })).catch(fail);
  } else {
    pending = message;
    stopping.interrupt();
    drain().catch((error) => {
      running = false;
      fail(error);
    });
  }
};
