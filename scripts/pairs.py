"""Paires de préférences pour le DPO (étape 7), tirées du modèle SFT lui-même (données on-policy).

Pour chaque exemple : N réponses échantillonnées (température > 0), notées par exécution sur les tests cachés, puis une
paire (chosen, rejected) :

- chosen : une réponse entièrement juste, la plus fréquente parmi les justes (la plus probable pour le modèle) ;
- rejected : une réponse fausse, la plus fréquente parmi les fausses (l'erreur que le modèle fait le plus volontiers),
  à égalité la moins bien notée.

Un exemple sans réponse juste ou sans réponse fausse ne donne pas de paire : soit le modèle sait déjà (rien à
corriger), soit il n'a rien de juste à renforcer.

    uv run scripts/pairs.py sample --split train --n 200      # pilote
    uv run scripts/pairs.py sample --split train
    uv run scripts/pairs.py sample --split valid
    uv run scripts/pairs.py build

Sorties :
    data/dpo/samples-<split>.jsonl  échantillons notés (reprise possible : les exemples déjà traités sont sautés)
    data/dpo/{train,valid}.jsonl    paires au format de mlx-lm-lora : {"system", "prompt", "chosen", "rejected"}
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from common import RegexRunner, build_messages, grade
from reward import reward

PREPARED_DIR = Path("data/prepared")
DPO_DIR = Path("data/dpo")
MAX_PROMPT_TOKENS = 384  # avec 128 tokens de réponse, reste sous le max_seq_length de 512 du SFT


def sample(args) -> None:
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler
    from tqdm import tqdm

    out_path = DPO_DIR / f"samples-{args.split}.jsonl"
    done = {json.loads(line)["id"] for line in out_path.open()} if out_path.exists() else set()
    examples = [json.loads(line) for line in (PREPARED_DIR / f"{args.split}.jsonl").open()][: args.n]
    todo = [e for e in examples if e["id"] not in done]
    print(f"{len(examples)} exemples, {len(done)} déjà échantillonnés, {len(todo)} à faire")
    if not todo:
        return

    mx.random.seed(args.seed)
    # MLX garde la mémoire libérée en cache pour la réutiliser : on la plafonne pour la rendre au système.
    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    prompts = {
        e["id"]: tokenizer.apply_chat_template(build_messages(e), add_generation_prompt=True, enable_thinking=False)
        for e in todo
    }
    todo.sort(key=lambda e: len(prompts[e["id"]]))  # moins de padding

    DPO_DIR.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with RegexRunner(timeout_ms=args.timeout_ms) as runner, out_path.open("a") as f:
        progress = tqdm(range(0, len(todo), args.prompts_per_batch), desc="échantillonnage", unit="batch")
        for i in progress:
            chunk = todo[i : i + args.prompts_per_batch]
            batch = [prompts[e["id"]] for e in chunk for _ in range(args.samples)]
            texts = batch_generate(model, tokenizer, batch, max_tokens=args.max_tokens, sampler=sampler).texts
            grades = grade(runner, [e for e in chunk for _ in range(args.samples)], texts)
            for j, e in enumerate(chunk):
                outs = texts[j * args.samples : (j + 1) * args.samples]
                gs = grades[j * args.samples : (j + 1) * args.samples]
                record = {
                    "id": e["id"],
                    "category": e["category"],
                    "difficulty": e["difficulty"],
                    "prompt_tokens": len(prompts[e["id"]]),
                    "outputs": outs,
                    "status": [g["status"] for g in gs],
                    "success": [g["success"] for g in gs],
                    "reward": [round(reward(g), 4) for g in gs],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            progress.set_postfix(peak=f"{mx.get_peak_memory() / 1024**3:.1f} Go")
    print(f"{len(todo) * args.samples} réponses en {time.monotonic() - start:.0f}s")
    stats(out_path)


def pick(record: dict) -> tuple[str, str] | None:
    """(chosen, rejected) d'un exemple, ou None si toutes ses réponses sont justes ou toutes fausses."""
    answers = [o.strip() for o in record["outputs"]]
    counts = Counter(answers)
    right = {a for a, ok in zip(answers, record["success"]) if ok}
    wrong = {a: r for a, ok, r in zip(answers, record["success"], record["reward"]) if not ok}
    if not right or not wrong:
        return None
    chosen = max(right, key=lambda a: (counts[a], -len(a), a))
    rejected = max(wrong, key=lambda a: (counts[a], -wrong[a], a))
    return chosen, rejected


def stats(path: Path) -> None:
    records = [json.loads(line) for line in path.open()]
    n, k = len(records), len(records[0]["outputs"])
    outputs = [s for r in records for s in r["status"]]
    print(f"\n{path} : {n} exemples × {k} réponses")
    print(f"  réponses justes {sum(sum(r['success']) for r in records) / (n * k):.1%}", end="")
    print(f" | mal formées {outputs.count('malformed') / len(outputs):.1%} | timeouts {outputs.count('timeout') / len(outputs):.1%}")
    print(f"  réponses distinctes par exemple : {sum(len(set(o.strip() for o in r['outputs'])) for r in records) / n:.1f}")
    all_right = sum(all(r["success"]) for r in records)
    none_right = sum(not any(r["success"]) for r in records)
    print(f"  toutes justes {all_right / n:.1%} | mélange (paire possible) {(n - all_right - none_right) / n:.1%}", end="")
    print(f" | aucune juste {none_right / n:.1%}")
    by_difficulty = Counter()
    for r in records:
        by_difficulty[r["difficulty"], pick(r) is not None] += 1
    for d in ("simple", "medium", "hard"):
        total = by_difficulty[d, True] + by_difficulty[d, False]
        if total:
            print(f"  {d:7} paires {by_difficulty[d, True] / total:.0%} ({total} exemples)")


def build(args) -> None:
    examples = {
        json.loads(line)["id"]: json.loads(line) for split in ("train", "valid") for line in (PREPARED_DIR / f"{split}.jsonl").open()
    }
    for split in ("train", "valid"):
        path = DPO_DIR / f"samples-{split}.jsonl"
        stats(path)
        pairs, too_long = [], 0
        for record in map(json.loads, path.open()):
            if record["prompt_tokens"] > MAX_PROMPT_TOKENS:
                too_long += 1
                continue
            if picked := pick(record):
                system, user = build_messages(examples[record["id"]])
                pairs.append({"id": record["id"], "system": system["content"], "prompt": user["content"],
                              "chosen": picked[0], "rejected": picked[1]})
        with (DPO_DIR / f"{split}.jsonl").open("w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  → {len(pairs)} paires dans {DPO_DIR / f'{split}.jsonl'}" + (f" ({too_long} prompts trop longs écartés)" if too_long else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="échantillonne et note les réponses du modèle SFT")
    s.add_argument("--split", choices=["train", "valid"], default="train")
    s.add_argument("--n", type=int, default=None, help="nombre d'exemples (pour un pilote)")
    s.add_argument("--model", default="Qwen/Qwen3-0.6B")
    s.add_argument("--adapter", default="adapters/sft-0.6b")
    s.add_argument("--samples", type=int, default=8, help="réponses par exemple")
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--prompts-per-batch", type=int, default=8, help="× --samples séquences générées en parallèle : règle la mémoire")
    s.add_argument("--cache-limit-gb", type=float, default=1.0, help="cache mémoire de MLX conservé entre deux batchs")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--timeout-ms", type=int, default=1000)
    sub.add_parser("build", help="construit les paires à partir des échantillons")
    args = parser.parse_args()
    sample(args) if args.command == "sample" else build(args)


if __name__ == "__main__":
    main()
