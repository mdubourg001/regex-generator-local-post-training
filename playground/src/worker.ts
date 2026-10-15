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

// Modèles servis par Vite depuis public/models/ (générés par scripts/export_models.py), jamais depuis le Hub.
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/models/";
env.useBrowserCache = true;
env.backends.onnx.wasm!.wasmPaths = { mjs: ortMjs, wasm: ortWasm };

export type ToWorker = { type: "load"; model: string } | ({ type: "generate"; id: number } & Request);
export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; model: string }
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

/** Un cache (Cache API) par modèle et par export : les poids d'un export précédent sont supprimés. */
async function useCacheFor(name: string, exportId: string) {
  const prefix = `regex-playground/${name}/`;
  env.cacheKey = prefix + exportId;
  try {
    for (const key of await caches.keys()) if (key.startsWith(prefix) && key !== env.cacheKey) await caches.delete(key);
  } catch {
    // Cache API indisponible (navigation privée…) : transformers.js charge alors sans cache.
  }
}

async function load(name: string) {
  stopping.interrupt();
  await generation.catch(() => {});
  await current?.model.dispose();
  current = null;
  const progress_callback = (info: { status: string; loaded?: number; total?: number }) => {
    if (info.status === "progress_total") post({ type: "progress", loaded: info.loaded!, total: info.total! });
  };
  // Fichier absent : Vite répond index.html (repli SPA) au lieu d'un 404, d'où un test sur le type de contenu.
  const config = await fetch(`${env.localModelPath}${name}/config.json`, { cache: "no-store" });
  if (!config.headers.get("content-type")?.includes("json")) {
    throw new Error(`${name} is not exported: run \`uv run scripts/export_models.py ${name}\``);
  }
  await useCacheFor(name, (await config.json()).playground_export_id ?? "unversioned");
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(name),
    AutoModelForCausalLM.from_pretrained(name, { device: "webgpu", dtype: "fp16", progress_callback }),
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
    loading = loading.then(() => load(message.model)).catch(fail);
  } else {
    pending = message;
    stopping.interrupt();
    drain().catch((error) => {
      running = false;
      fail(error);
    });
  }
};
