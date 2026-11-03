// Section « How it works » : SFT Qwen3.5-0.8B (configs/sft-qwen3.5-0.8b.yaml) ; limites mesurées sur le benchmark
// regex101 (data/eval/benchmark-regex101.jsonl) avec l'export q4f16, par difficulté : 67/162, 26/199, 3/39.

const TRAINING = [
  "Base model: Qwen3.5-0.8B, a small open model.",
  "About 9,400 synthetic examples (description → regex), written by larger LLMs.",
  "Each example is kept only if its regex passes test strings written separately.",
  "Fine-tuned with LoRA on a MacBook, then quantized to 4 bits.",
];

const IN_THIS_PAGE = [
  "The model runs on your GPU (WebGPU), or on the CPU without it: nothing leaves your machine.",
  "The regex is regenerated 400 ms after you stop typing.",
  "Each test line is checked with RegExp.test.",
];

const LIMITS = [
  "On real-world regexes, about 4 in 10 simple requests fully right.",
  "About 1 in 8 intermediate ones, under 1 in 10 hard ones.",
  "Always check the result against your own test strings.",
];

function List({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <h3 className="mb-1.5 font-semibold text-zinc-900 dark:text-zinc-100">{title}</h3>
      <ul className="list-disc space-y-1 pl-5">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export function HowItWorks() {
  return (
    <section aria-labelledby="how-it-works" className="border-t border-zinc-200 pt-8 dark:border-zinc-800">
      <h2 id="how-it-works" className="mb-4 text-lg font-semibold tracking-tight">
        How it works
      </h2>
      <div className="grid gap-6 text-sm leading-relaxed text-zinc-700 sm:grid-cols-3 dark:text-zinc-300">
        <List title="Training" items={TRAINING} />
        <List title="In this page" items={IN_THIS_PAGE} />
        <List title="Limits" items={LIMITS} />
      </div>
    </section>
  );
}
