"""Écrit le modèle édité dans models/wsp-0.6b (format MLX, chargeable par eval.py --model).

uv run wsp/build_model.py
"""
import json
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import save
from huggingface_hub import snapshot_download
import wsplib

FINAL = ["e18-blind-flag-14h", "e07-first-sparse", "e19-ban-slash", "e20-end"]
m, tok = wsplib.load()
total = sum(v.size for _, v in tree_flatten(m.parameters()))
report = []
for e in FINAL:
    n = wsplib.apply_edits(m, [e])
    report.append({"edit": e, "weights_changed": n, "percent": round(100 * n / total, 4)})
    print(report[-1])
src = snapshot_download(wsplib.BASE)
config = json.load(open(f"{src}/config.json"))
dst = wsplib.ROOT / "models" / "wsp-0.6b"
save(str(dst), src, m, tok, config)
(wsplib.WSP / "final_edits.json").write_text(json.dumps({"base": wsplib.BASE, "edits": FINAL, "total_params": total, "report": report}, indent=1))
print("écrit :", dst)
