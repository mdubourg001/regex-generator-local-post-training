"""Lance `mlx_lm_lora.train` (DPO, GRPO) ou, avec `--sft`, `mlx_lm.lora` en plafonnant la mémoire de MLX, pour garder la
machine utilisable à côté.

    uv run scripts/train_capped.py --memory-gb 8 --config configs/dpo.yaml
    uv run scripts/train_capped.py --sft --memory-gb 6 --config configs/sft-qwen3.5-0.8b.yaml

- `mlx_lm_lora` verrouille par défaut jusqu'à 19 Go de mémoire GPU (non swappable) sur un Mac de 24 Go : on limite ce
  verrou au plafond choisi ;
- la limite mémoire de MLX est indicative : au-delà, MLX attend que des calculs se terminent pour libérer de la mémoire
  avant d'allouer, mais ne lève pas d'erreur. C'est la taille des batchs qui tient vraiment le pic ;
- MLX garde en cache la mémoire libérée pour la réutiliser : on plafonne ce cache à 1 Go.

Il corrige aussi un défaut du DPO de mlx-lm-lora : chaque batch est complété à la longueur de sa plus longue séquence,
et le pas d'entraînement est compilé (`mx.compile`) pour chaque nouvelle forme. Avec ~400 longueurs possibles, le cache
de compilation accumule des buffers Metal jusqu'à dépasser la limite de Metal (`Resource limit (499000) exceeded`,
vers l'itération 200). On complète les batchs à un multiple de 32 tokens : au plus 16 formes jusqu'à 512 tokens. Les
tokens ajoutés ont un masque nul et ne comptent pas dans la loss. Et `--dpo-cpo-loss-type` (hinge, ipo, dpop) n'arrivait
jamais jusqu'à la loss : voir `pass_dpo_loss_type`.

Les autres arguments sont passés tels quels à `mlx_lm_lora.train`.
"""

import argparse
import sys

import mlx.core as mx

PAD_MULTIPLE = 32


def pad_dpo_batches() -> None:
    import mlx_lm_lora.trainer.dpo_trainer as dpo

    iterate = dpo.iterate_dpo_batches

    def padded(*args, **kwargs):
        for batch in iterate(*args, **kwargs):
            pad = -batch[0].shape[1] % PAD_MULTIPLE
            yield tuple(mx.pad(a, [(0, 0), (0, pad)]) for a in batch) if pad else batch

    dpo.iterate_dpo_batches = padded


def pass_dpo_loss_type() -> None:
    """mlx-lm-lora range `--dpo-cpo-loss-type` dans `DPOTrainingArgs.loss_type`, mais `train_dpo` lit son propre
    paramètre `loss_type`, qui vaut toujours "sigmoid" : sans ce correctif, hinge, ipo et dpop sont ignorés."""
    import mlx_lm_lora.train as cli

    train_dpo = cli.train_dpo

    def patched(*args, **kwargs):
        kwargs.setdefault("loss_type", kwargs["args"].loss_type)
        print(f"DPO loss : {kwargs['loss_type']}, beta {kwargs['args'].beta}, delta {kwargs['args'].delta}")
        return train_dpo(*args, **kwargs)

    cli.train_dpo = patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory-gb", type=float, default=8.0)
    parser.add_argument("--cache-gb", type=float, default=1.0)
    parser.add_argument("--sft", action="store_true", help="lance mlx_lm.lora (SFT) au lieu de mlx_lm_lora.train")
    args, rest = parser.parse_known_args()

    limit = int(args.memory_gb * 1024**3)
    mx.set_memory_limit(limit)
    mx.set_cache_limit(int(args.cache_gb * 1024**3))
    set_wired_limit = mx.set_wired_limit
    mx.set_wired_limit = lambda n: set_wired_limit(min(n, limit))

    if args.sft:
        from mlx_lm.lora import main as train

        sys.argv = ["mlx_lm.lora", *rest]
        return train()

    pad_dpo_batches()
    pass_dpo_loss_type()
    from mlx_lm_lora.train import main as train

    sys.argv = ["mlx_lm_lora.train", *rest]
    train()


if __name__ == "__main__":
    main()
