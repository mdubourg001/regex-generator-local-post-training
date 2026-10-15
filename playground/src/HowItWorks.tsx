// Section « How it works » : chiffres repris du README racine (étapes 6, 9 ter et 10 bis).

const TRAINING = [
  "Base model: Qwen3-0.6B, a small open model.",
  "About 8,700 synthetic examples (description → regex), written by larger LLMs.",
  "Each example is kept only if its regex passes test strings written separately.",
  "Fine-tuned with LoRA in about an hour on a MacBook.",
];

const IN_THIS_PAGE = [
  "The model runs on your GPU (WebGPU): nothing leaves your machine.",
  "The regex is regenerated 400 ms after you stop typing.",
  "Each test line is checked with RegExp.test.",
];

const LIMITS = [
  "About 6 in 10 simple requests fully right.",
  "About 1 in 10 intermediate ones, almost none of the hard ones.",
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
