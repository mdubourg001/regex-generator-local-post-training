"""GRPO (étape 9) : apprentissage par renforcement sur la reward d'exécution, sur le modèle SFT.

À chaque pas :

1. on tire `--prompts-per-step` exemples du train, et le modèle génère `--group-size` réponses pour chacun
   (température 1, sans top-p : on échantillonne exactement la politique qu'on entraîne) ;
2. chaque réponse est exécutée sur les tests cachés et notée par `reward.py` (entre -1 et 1) ;
3. avantage d'une réponse = (reward − moyenne de son groupe) / écart-type du groupe. Un groupe où toutes les réponses
   ont la même note n'apprend rien (avantages nuls) : on le saute ;
4. loss = −Σ avantage × log π(token) sur les tokens des réponses (token de fin compris), divisée par le nombre de
   tokens du pas, plus β × KL vers le modèle SFT. Un seul pas de gradient par génération : la politique qui a
   échantillonné est celle qu'on dérive, le ratio d'importance vaut 1 et le clipping de PPO n'a pas d'effet.

Pourquoi pas le GRPO de mlx-lm-lora 3.1.3 : sa loss calcule les log-probabilités des réponses sans le prompt devant,
clippe le ratio contre le modèle de référence (avec ε = 1e-4 : gradient coupé dès qu'on s'écarte du SFT), construit
le prompt sans `enable_thinking=False`, retire le token de fin des réponses (le modèle n'apprend pas à s'arrêter), et
recompile son pas pour chaque forme de batch.

On part du modèle SFT fusionné (`models/sft-0.6b`) avec un LoRA neuf : le modèle de référence pour la KL est le même
modèle, LoRA désactivé (échelle à 0), sans deuxième copie en mémoire.

Exemples d'entraînement : ceux du train où les 8 réponses échantillonnées à l'étape 7 n'ont pas toutes la même
reward (71 % du train). Les autres n'apportent rien : groupes tous justes, ou tous faux de la même façon.

    uv run scripts/grpo.py --steps 20 --adapter-path adapters/grpo-probe      # sonde : durée et mémoire
    uv run scripts/grpo.py                                                    # reprend là où il s'est arrêté

Sorties dans `--adapter-path` : `adapters.safetensors` et `adapter_config.json` (lisibles par eval.py avec
`--model models/sft-0.6b`), un checkpoint tous les `--save-every` pas, `optimizer.safetensors` et `state.json` pour
la reprise, `log.jsonl` (une ligne de métriques par pas).
"""

import argparse
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from common import RegexRunner, build_messages, grade
from reward import reward

PREPARED_DIR = Path("data/prepared")
SAMPLES = Path("data/dpo/samples-train.jsonl")
MAX_PROMPT_TOKENS = 384
LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def advantages(rewards: list[float], group_size: int) -> list[float | None]:
    """Avantage normalisé de chaque réponse dans son groupe, None pour les groupes sans écart (rien à apprendre)."""
    out = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]
        mean = sum(group) / len(group)
        std = (sum((r - mean) ** 2 for r in group) / len(group)) ** 0.5
        out += [(r - mean) / std if std > 1e-6 else None for r in group]
    return out


def training_ids() -> list[str]:
    """Exemples dont les échantillons de l'étape 7 n'ont pas tous la même reward."""
    ids = []
    for line in SAMPLES.open():
        r = json.loads(line)
        if r["prompt_tokens"] <= MAX_PROMPT_TOKENS and len(set(r["reward"])) > 1:
            ids.append(r["id"])
    return sorted(ids)


def generate(model, tokenizer, prompts: list[list[int]], max_tokens: int, sampler) -> list[tuple[list[int], bool]]:
    """(tokens, terminée) de chaque prompt. Les tokens d'une réponse terminée finissent par le token de fin."""
    from mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(model, stop_tokens=[[t] for t in tokenizer.eos_token_ids], sampler=sampler)
    uids = gen.insert(prompts, [max_tokens] * len(prompts))
    tokens, finished = {uid: [] for uid in uids}, {}
    while responses := gen.next_generated():
        for r in responses:
            tokens[r.uid].append(r.token)
            if r.finish_reason is not None:
                finished[r.uid] = r.finish_reason == "stop"
    gen.close()
    return [(tokens[uid], finished[uid]) for uid in uids]


@contextmanager
def lora_disabled(model):
    """Le modèle de référence : LoRA éteint, c'est-à-dire le SFT fusionné."""
    layers = [m for _, m in model.named_modules() if hasattr(m, "lora_a")]
    scales = [m.scale for m in layers]
    for m in layers:
        m.scale = 0.0
    try:
        yield
    finally:
        for m, s in zip(layers, scales):
            m.scale = s


def answer_logps(model, inputs: mx.array, positions: mx.array) -> mx.array:
    """log π du token qui suit chaque position de `positions` (B, C).

    La tête de vocabulaire (151 936 sorties) n'est appliquée qu'aux positions de la réponse, pas à tout le prompt : ce
    sont ses logits qui dominent la mémoire."""
    hidden = mx.take_along_axis(model.model(inputs), positions[..., None], axis=1)
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    logits = head(hidden).astype(mx.float32)
    targets = mx.take_along_axis(inputs, positions + 1, axis=1)
    picked = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    return picked - mx.logsumexp(logits, axis=-1)


def make_batch(seqs: list[tuple[list[int], list[int], float]]):
    """Séquences prompt + réponse complétées à droite, positions qui prédisent les tokens de réponse, masque, avantages."""
    length = max(len(p) + len(c) for p, c, _ in seqs)
    width = max(len(c) for _, c, _ in seqs)
    inputs, positions, mask = [], [], []
    for p, c, _ in seqs:
        inputs.append(p + c + [0] * (length - len(p) - len(c)))
        # La position len(p) - 1 + j prédit le j-ième token de réponse ; les positions en trop répètent la dernière.
        positions.append([len(p) - 1 + min(j, len(c) - 1) for j in range(width)])
        mask.append([1] * len(c) + [0] * (width - len(c)))
    adv = [a for _, _, a in seqs]
    return mx.array(inputs), mx.array(positions), mx.array(mask, dtype=mx.float32), mx.array(adv, dtype=mx.float32)


def save(model, optimizer, out: Path, step: int, args) -> None:
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    if step % args.save_every == 0:
        mx.save_safetensors(str(out / f"{step:05d}_adapters.safetensors"), weights)
    mx.save_safetensors(str(out / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    (out / "state.json").write_text(json.dumps({"step": step}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/sft-0.6b", help="modèle SFT fusionné, aussi modèle de référence")
    parser.add_argument("--adapter-path", default="adapters/grpo-0.6b")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=10, help="pas de montée linéaire de la LR, puis LR constante")
    parser.add_argument("--beta", type=float, default=0.04, help="poids de la KL vers le modèle SFT")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--gen-prompts", type=int, default=4, help="prompts générés ensemble (× --group-size) : règle la mémoire")
    parser.add_argument("--micro-batch", type=int, default=4, help="séquences par passe avant/arrière : règle la mémoire")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--memory-gb", type=float, default=6.0)
    parser.add_argument("--cache-gb", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    args = parser.parse_args()

    # Plafonds mémoire, comme train_capped.py : mlx_lm verrouille sinon jusqu'à 19 Go pendant la génération.
    limit = int(args.memory_gb * 1024**3)
    mx.set_memory_limit(limit)
    mx.set_cache_limit(int(args.cache_gb * 1024**3))
    set_wired_limit = mx.set_wired_limit
    mx.set_wired_limit = lambda n: set_wired_limit(min(n, limit))

    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    out = Path(args.adapter_path)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model)
    model.freeze()
    lora = {"rank": args.rank, "scale": args.scale, "dropout": 0.0, "keys": LORA_KEYS}
    linear_to_lora_layers(model, len(model.layers), lora)
    # Activations recalculées pendant la passe arrière au lieu d'être gardées : pic mémoire divisé, ~30 % plus lent.
    grad_checkpoint(model.layers[0])
    (out / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora, "model": args.model,
         **{k: v for k, v in vars(args).items() if k not in ("rank", "scale")}}, indent=2))

    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, args.learning_rate, args.warmup), lambda _: args.learning_rate], [args.warmup])
    optimizer = optim.Adam(learning_rate=schedule)
    start_step = 0
    if (out / "state.json").exists():
        start_step = json.loads((out / "state.json").read_text())["step"]
        model.load_weights(str(out / "adapters.safetensors"), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(out / "optimizer.safetensors")).items()))
        print(f"reprise au pas {start_step}")
    model.train()

    examples = {json.loads(line)["id"]: json.loads(line) for line in (PREPARED_DIR / "train.jsonl").open()}
    ids = training_ids()
    # Ordre fixé par la graine : un pas donné voit toujours les mêmes prompts, reprise comprise.
    order = []
    while len(order) < args.steps * args.prompts_per_step:
        epoch = ids[:]
        random.Random(args.seed + len(order)).shuffle(epoch)
        order += epoch
    print(f"{len(ids)} exemples d'entraînement, {args.steps} pas de {args.prompts_per_step} prompts × {args.group_size}")

    mx.random.seed(args.seed + start_step)
    sampler = make_sampler(temp=args.temperature)

    def loss_fn(model, inputs, positions, mask, adv, n_tokens):
        logps = answer_logps(model, inputs, positions)
        # Référence calculée dans le même graphe : hors de value_and_grad, MLX prend d'autres noyaux et les
        # log-probabilités diffèrent de ~0,002 par token en bf16, assez pour une KL parasite.
        with lora_disabled(model):
            ref = mx.stop_gradient(answer_logps(model, inputs, positions))
        loss = -(adv[:, None] * logps * mask).sum()
        # Estimateur k3 de la KL(π || π_ref), toujours positif.
        kl = mx.exp(ref - logps) - (ref - logps) - 1
        return (loss + args.beta * (kl * mask).sum()) / n_tokens, (kl * mask).sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    with RegexRunner(timeout_ms=args.timeout_ms) as runner, (out / "log.jsonl").open("a") as log:
        for step in range(start_step + 1, args.steps + 1):
            t0 = time.monotonic()
            batch = [examples[i] for i in order[(step - 1) * args.prompts_per_step : step * args.prompts_per_step]]
            prompts = [tokenizer.apply_chat_template(build_messages(e), add_generation_prompt=True, enable_thinking=False)
                       for e in batch]

            model.eval()
            outs = []
            for i in range(0, len(prompts), args.gen_prompts):
                chunk = [p for p in prompts[i : i + args.gen_prompts] for _ in range(args.group_size)]
                outs += generate(model, tokenizer, chunk, args.max_tokens, sampler)
                mx.clear_cache()
            model.train()
            mx.clear_cache()
            texts = [tokenizer.decode(toks[:-1] if done else toks) for toks, done in outs]
            grades = grade(runner, [e for e in batch for _ in range(args.group_size)], texts)
            rewards = [reward(g) for g in grades]
            advs = advantages(rewards, args.group_size)
            t_gen = time.monotonic() - t0

            seqs = [(prompts[i // args.group_size], toks, a) for i, ((toks, _), a) in enumerate(zip(outs, advs)) if a is not None]
            n_tokens = sum(len(c) for _, c, _ in seqs)
            kl_sum, grads = 0.0, None
            seqs.sort(key=lambda s: len(s[0]) + len(s[1]))  # moins de padding par micro-batch
            for i in range(0, len(seqs), args.micro_batch):
                inputs, positions, mask, adv = make_batch(seqs[i : i + args.micro_batch])
                (_, kl), g = loss_and_grad(model, inputs, positions, mask, adv, n_tokens)
                grads = g if grads is None else tree_map(mx.add, grads, g)
                mx.eval(grads, kl)
                kl_sum += kl.item()
            if grads is not None:
                optimizer.update(model, grads)
                mx.eval(model.trainable_parameters(), optimizer.state)
            mx.clear_cache()

            metrics = {
                "step": step,
                "reward": round(sum(rewards) / len(rewards), 4),
                "success": round(sum(g["success"] for g in grades) / len(grades), 4),
                "malformed": round(sum(g["status"] != "ok" for g in grades) / len(grades), 4),
                "hit_limit": round(sum(not done for _, done in outs) / len(outs), 4),
                "tokens": round(sum(len(t) for t, _ in outs) / len(outs), 1),
                "groups_kept": len(seqs) // args.group_size,
                "kl": round(kl_sum / max(n_tokens, 1), 5),
                "lr": float(optimizer.learning_rate),
                "gen_s": round(t_gen, 1),
                "step_s": round(time.monotonic() - t0, 1),
                "peak_gb": round(mx.get_peak_memory() / 1024**3, 2),
            }
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(" ".join(f"{k} {v}" for k, v in metrics.items()), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save(model, optimizer, out, step, args)


if __name__ == "__main__":
    main()
