// Génère les réponses d'un export ONNX avec transformers.js dans Node (greedy, comme scripts/eval.py), pour les noter
// ensuite sans modèle : uv run scripts/eval.py --name <nom> --data <cas.jsonl> --outputs <sortie.jsonl>
//
//   node scripts/generate.mjs --model sft-qwen3.5-0.8b --dtype q4 --prompts prompts.jsonl --out out.jsonl [--device cpu]
//
// --prompts : JSONL {id, prompt}, prompt déjà passé dans le gabarit de chat par le tokenizer Python (celui d'eval.py),
// pour que l'entrée soit identique au token près. Reprend là où s'arrête un --out existant.
import { AutoModelForCausalLM, AutoTokenizer, env } from "@huggingface/transformers";
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    model: { type: "string" },
    dtype: { type: "string", default: "q4" },
    device: { type: "string", default: "cpu" },
    prompts: { type: "string" },
    out: { type: "string" },
    "max-tokens": { type: "string", default: "128" },
  },
});

env.localModelPath = fileURLToPath(new URL("../public/models/", import.meta.url));
env.allowRemoteModels = false;

const read = (path) =>
  readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
const done = new Set(existsSync(args.out) ? read(args.out).map((r) => r.id) : []);
const cases = read(args.prompts).filter((c) => !done.has(c.id));

const tokenizer = await AutoTokenizer.from_pretrained(args.model);
const model = await AutoModelForCausalLM.from_pretrained(args.model, { dtype: args.dtype, device: args.device });

const start = performance.now();
let tokens = 0;
for (const [i, c] of cases.entries()) {
  const inputs = tokenizer(c.prompt, { add_special_tokens: false });
  const output = await model.generate({ ...inputs, max_new_tokens: Number(args["max-tokens"]), do_sample: false });
  const prompt = inputs.input_ids.dims[1];
  const generated = output.slice(null, [prompt, null]);
  tokens += generated.dims[1];
  const text = tokenizer.batch_decode(generated, { skip_special_tokens: true })[0];
  appendFileSync(args.out, JSON.stringify({ id: c.id, output: text }) + "\n");
  if ((i + 1) % 20 === 0 || i + 1 === cases.length) {
    const s = (performance.now() - start) / 1000;
    console.log(`${i + 1}/${cases.length}  ${s.toFixed(0)} s  ${(tokens / s).toFixed(1)} tokens/s`);
  }
}
