// Section « How it works » : SFT Qwen3.5-0.8B (configs/sft-qwen3.5-0.8b.yaml) et Qwen3-1.7B (configs/sft.yaml) ;
// les limites sont celles du modèle sélectionné (Benchmark, défini avec les modèles dans App.tsx).

/** Succès complets au benchmark regex101 (data/eval/benchmark-regex101.jsonl), [réussis, total] par difficulté. */
export type Benchmark = Record<"simple" | "medium" | "hard", [number, number]>;

const TRAINING = [
  "Base model: Qwen3.5-0.8B or Qwen3-1.7B (picked above), small open models.",
  "About 8,300 to 9,400 synthetic examples (description → regex), written by larger LLMs.",
  "Each example is kept only if its regex passes test strings written separately.",
  "Fine-tuned with LoRA on a MacBook, then quantized to 4 bits.",
];

const IN_THIS_PAGE = [
  "The model runs on your GPU (WebGPU), or on the CPU without it: nothing leaves your machine.",
  "The regex is regenerated 400 ms after you stop typing.",
  "Each test line is checked with RegExp.test.",
];

const pct = ([ok, total]: [number, number]) => `${Math.round((100 * ok) / total)}%`;

function List({ title, items }: { title: string; items: React.ReactNode[] }) {
  return (
    <div>
      <h3 className="mb-1.5 font-semibold text-zinc-900 dark:text-zinc-100">{title}</h3>
      <ul className="list-disc space-y-1 pl-5">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export function HowItWorks({ model, benchmark }: { model: string; benchmark: Benchmark }) {
  const limits = [
    `On real-world regexes (regex101 benchmark), ${model} gets ${pct(benchmark.simple)} of simple requests fully right.`,
    `${pct(benchmark.medium)} of intermediate ones, ${pct(benchmark.hard)} of hard ones.`,
    `Average of ~70% passing tests on failing cases`,
    "Always check the result against your own test strings.",
    <span className="font-medium text-rose-600 dark:text-rose-400">
      Use at your own risk in production.
    </span>,
  ];
  return (
    <section
      aria-labelledby="how-it-works"
      className="border-t border-zinc-200 pt-8 dark:border-zinc-800"
    >
      <h2 id="how-it-works" className="mb-4 text-lg font-semibold tracking-tight">
        How it works
      </h2>
      <div className="grid gap-6 text-sm leading-relaxed text-zinc-700 sm:grid-cols-3 dark:text-zinc-300">
        <List title="Training" items={TRAINING} />
        <List title="In this page" items={IN_THIS_PAGE} />
        <List title="Limits" items={limits} />
      </div>
    </section>
  );
}
