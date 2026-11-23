"""Évalue le modèle (brut + éditions) sur le jeu de dev. Chaque appel compte dans le budget de 50.

uv run wsp/dev_eval.py --name base
uv run wsp/dev_eval.py --name e01 --edits e01-xxx,e02-yyy --note "..."
"""

import argparse

import mlx.core as mx

import wsplib

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--edits", default="")
parser.add_argument("--model", default=wsplib.BASE)
parser.add_argument("--note", default="")
args = parser.parse_args()

mx.set_memory_limit(6 * 1024**3)
mx.set_cache_limit(1 * 1024**3)
edits = [e for e in args.edits.split(",") if e]
model, tok = wsplib.load(args.model, edits)
wsplib.dev_eval(model, tok, args.name, edits, args.note)
print(f"pic mémoire {mx.get_peak_memory() / 1e9:.1f} Go")
