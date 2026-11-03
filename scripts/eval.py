"""Évalue un modèle (et éventuellement un adapter LoRA) sur la génération de regex.

Pour chaque cas : prompt (description + exemples éventuels) → génération greedy → parsing du littéral `/…/flags` →
exécution dans Node sur les tests cachés (succès complet, taux de tests réussis, sorties mal formées, timeouts).

    uv run scripts/eval.py --name baseline-0.6b --data data/eval/benchmark.jsonl
    uv run scripts/eval.py --name sft-0.6b --data data/eval/benchmark.jsonl --adapter adapters/sft
    uv run scripts/eval.py --name baseline-1.7b --data data/eval/benchmark.jsonl --model Qwen/Qwen3-1.7B
    uv run scripts/eval.py --name reference --data tests/fixtures/cases.jsonl --reference  # sans modèle, contrôle des données
    uv run scripts/eval.py --name sft-0.6b-plus-q4f16-gpt --data data/eval/benchmark-gpt.jsonl \
        --outputs results/onnx/sft-0.6b-plus-q4f16-gpt.jsonl  # réponses déjà générées (ONNX, playground/scripts/generate.ts)

Sorties :
    results/<name>.jsonl  détail par cas (réponse brute, statut, tests ratés)
    results/scores.jsonl  une ligne de synthèse par run (pour le tableau final)
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from common import RegexRunner, build_messages, grade, score

RESULTS_DIR = Path("results")


def generate(
    model_name: str,
    adapter: str | None,
    examples: list[dict],
    max_tokens: int,
    batch_size: int,
    few_shot: bool,
    thinking: bool = False,
    plain_template: bool = False,
) -> list[str]:
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler
    from tqdm import tqdm

    model, tokenizer = load(model_name, adapter_path=adapter)
    # enable_thinking=False : désactive le mode « raisonnement » de Qwen3 (ignoré par les modèles qui ne l'ont pas).
    # plain_template : enable_thinking=None, soit le gabarit sans rien après « assistant\n » (ni <think>, ni bloc vide).
    # mlx_lm met sinon enable_thinking=True par défaut. Pour les modèles entraînés avec train_capped.py --plain-template
    # (MiniCPM5 : son gabarit n'écrit pas de bloc <think> dans l'historique, voir train_capped.py).
    kwargs = {"enable_thinking": None} if plain_template else {"enable_thinking": thinking}
    prompts = [
        tokenizer.apply_chat_template(build_messages(e, few_shot=few_shot), add_generation_prompt=True, **kwargs)
        for e in examples
    ]
    # Avec raisonnement : l'échantillonnage recommandé par Qwen3 pour ce mode (en greedy, le raisonnement boucle).
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20) if thinking else None
    # Les prompts de longueur proche sont générés ensemble : moins de padding, donc moins de mémoire et de calcul.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [""] * len(prompts)
    for i in tqdm(range(0, len(order), batch_size), desc="génération", unit="batch"):
        batch = order[i : i + batch_size]
        texts = batch_generate(model, tokenizer, [prompts[j] for j in batch], max_tokens=max_tokens, sampler=sampler).texts
        for j, text in zip(batch, texts):
            outputs[j] = text
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="nom du run, ex. baseline-0.6b")
    parser.add_argument("--data", required=True, type=Path, help="JSONL de cas (format décrit dans le README)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None, help="dossier d'un adapter LoRA")
    parser.add_argument("--reference", action="store_true", help="note les regex de référence au lieu d'un modèle")
    parser.add_argument("--outputs", type=Path, default=None, help="JSONL {id, output} déjà généré : note sans modèle")
    parser.add_argument("--few-shot", action="store_true", help="ajoute au prompt les exemples fixes de common.FEW_SHOT")
    parser.add_argument("--thinking", action="store_true", help="active le raisonnement de Qwen3 (prévoir --max-tokens élevé)")
    parser.add_argument("--plain-template", action="store_true", help="gabarit sans <think> (enable_thinking=None), pour un SFT lancé avec train_capped.py --plain-template")
    parser.add_argument("--n", type=int, default=None, help="nombre de cas (pour un test rapide)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout-ms", type=int, default=1000, help="délai maximal par regex (ReDoS)")
    args = parser.parse_args()

    examples = [json.loads(line) for line in args.data.open()][: args.n]
    start = time.monotonic()
    if args.reference:
        outputs = [e["regex"] for e in examples]
    elif args.outputs:
        generated = {r["id"]: r["output"] for r in map(json.loads, args.outputs.open())}
        outputs = [generated[e["id"]] for e in examples]
    else:
        outputs = generate(args.model, args.adapter, examples, args.max_tokens, args.batch_size, args.few_shot, args.thinking, args.plain_template)
    gen_time = time.monotonic() - start
    with RegexRunner(timeout_ms=args.timeout_ms) as runner:
        grades = grade(runner, examples, outputs)
    result = score(grades)

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / f"{args.name}.jsonl").open("w") as f:
        for e, out, g in zip(examples, outputs, grades):
            record = {"id": e["id"], "category": e.get("category"), "description": e["description"], "reference": e["regex"]}
            f.write(json.dumps({**record, "output": out, **g}, ensure_ascii=False) + "\n")
    summary = {
        "name": args.name,
        "model": "reference" if args.reference else str(args.outputs) if args.outputs else args.model,
        "adapter": args.adapter,
        "few_shot": args.few_shot,
        "thinking": args.thinking,
        "plain_template": args.plain_template,
        "data": str(args.data),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()},
        "generation_seconds": round(gen_time, 1),
    }
    with (RESULTS_DIR / "scores.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"\n{args.name} sur {args.data} ({result['n']} cas), génération {gen_time:.0f}s")
    print(f"  succès complet {result['success_rate']:.3f} | tests réussis {result['mean_pass_rate']:.3f}")
    print(f"  mal formées {result['malformed_rate']:.1%} | timeouts {result['timeout_rate']:.1%}")

    by_category = defaultdict(list)
    for e, g in zip(examples, grades):
        by_category[e.get("category") or "-"].append(g)
    print(f"\n  {'catégorie':24} {'succès':>6} {'tests':>6} {'n':>4}")
    for category, gs in sorted(by_category.items()):
        s = score(gs)
        print(f"  {category:24} {s['success_rate']:>6.2f} {s['mean_pass_rate']:>6.2f} {s['n']:>4}")


if __name__ == "__main__":
    main()
