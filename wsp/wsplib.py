"""Briques du weight-space programming : chargement, jeu de dev, éditions reprenables, forward instrumenté.

Aucun gradient ici : uniquement des lectures de poids, des passes forward et des éditions écrites à la main.

Une édition est un fichier wsp/edits/<nom>.npz : pour chaque tenseur touché, soit le tenseur complet remplacé
(clé `full:<nom>`), soit une mise à jour de rang r `W += U @ V` (clés `U:<nom>`, `V:<nom>`). Le fichier est produit
par un script d'édition (wsp/edit_*.py) qui documente l'hypothèse ; `apply_edits` le rejoue sur le modèle chargé.
"""

import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WSP = ROOT / "wsp"
sys.path.insert(0, str(ROOT / "scripts"))
from common import RegexRunner, build_messages, grade, parse_answer, score  # noqa: E402

BASE = "Qwen/Qwen3-0.6B"
DEV_IDS = WSP / "dev_ids.json"
EVAL_LOG = WSP / "dev_evals.jsonl"
BUDGET = 50


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def dev_set():
    """200 cas de valid.jsonl, seed 0, ids figés dans wsp/dev_ids.json."""
    valid = read_jsonl(ROOT / "data/prepared/valid.jsonl")
    if not DEV_IDS.exists():
        ids = sorted(random.Random(0).sample([e["id"] for e in valid], 200))
        DEV_IDS.write_text(json.dumps(ids, indent=0))
    ids = set(json.loads(DEV_IDS.read_text()))
    return [e for e in valid if e["id"] in ids]


def probe_pool():
    """Le reste de valid (305 cas) : sert aux sondes, jamais au score de dev."""
    ids = set(json.loads(DEV_IDS.read_text()))
    return [e for e in read_jsonl(ROOT / "data/prepared/valid.jsonl") if e["id"] not in ids]


def load(model=BASE, edits=()):
    from mlx_lm import load as mlx_load

    m, tok = mlx_load(model)
    apply_edits(m, edits)
    return m, tok


def get_param(model, name):
    obj = model
    for part in name.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def set_param(model, name, value):
    *path, leaf = name.split(".")
    obj = model
    for part in path:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    setattr(obj, leaf, value.astype(mx.bfloat16))


def apply_edits(model, edits):
    """Rejoue des éditions (noms de wsp/edits/*.npz) dans l'ordre. Renvoie le nombre de poids modifiés."""
    changed = 0
    for name in edits:
        data = np.load(WSP / "edits" / f"{name}.npz")
        for key in data.files:
            kind, tensor = key.split(":", 1)
            if kind == "V":
                continue
            old = get_param(model, tensor)
            before = np.array(old.astype(mx.float32))
            if kind == "full":
                new = data[key].astype(np.float32)
            else:  # kind == "U" : W += U @ V
                new = before + data[key].astype(np.float32) @ data[f"V:{tensor}"].astype(np.float32)
            set_param(model, tensor, mx.array(new))
            after = np.array(get_param(model, tensor).astype(mx.float32))
            changed += int((after != before).sum())
    mx.eval(model.parameters())
    return changed


def prompts_for(tok, examples, few_shot=False):
    return [
        tok.apply_chat_template(build_messages(e, few_shot=few_shot), add_generation_prompt=True, enable_thinking=False)
        for e in examples
    ]


def generate(model, tok, examples, max_tokens=128, batch_size=16):
    """Même génération que scripts/eval.py : greedy, enable_thinking=False, 128 tokens."""
    from mlx_lm import batch_generate

    prompts = prompts_for(tok, examples)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [""] * len(prompts)
    for i in range(0, len(order), batch_size):
        batch = order[i : i + batch_size]
        texts = batch_generate(model, tok, [prompts[j] for j in batch], max_tokens=max_tokens).texts
        for j, text in zip(batch, texts):
            outputs[j] = text
        mx.clear_cache()
    return outputs


def grade_outputs(examples, outputs):
    with RegexRunner() as runner:
        return grade(runner, examples, outputs)


def n_dev_evals():
    return sum(1 for _ in EVAL_LOG.open()) if EVAL_LOG.exists() else 0


def dev_eval(model, tok, name, edits=(), note=""):
    """Une évaluation sur le dev : compte dans le budget de 50, détail dans wsp/runs/<name>.jsonl."""
    n = n_dev_evals()
    if n >= BUDGET:
        raise SystemExit(f"budget de {BUDGET} évaluations dev épuisé")
    examples = dev_set()
    outputs = generate(model, tok, examples)
    grades = grade_outputs(examples, outputs)
    s = score(grades)
    with (WSP / "runs" / f"{name}.jsonl").open("w") as f:
        for e, out, g in zip(examples, outputs, grades):
            f.write(json.dumps({"id": e["id"], "description": e["description"], "reference": e["regex"], "output": out, **g}, ensure_ascii=False) + "\n")
    line = {"eval": n + 1, "name": name, "edits": list(edits), "note": note, **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}}
    with EVAL_LOG.open("a") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(json.dumps(line, ensure_ascii=False))
    return s, outputs, grades


# --- Forward instrumenté ----------------------------------------------------------------------------------------------


def forward(model, ids, hooks=None, want=()):
    """Passe forward sans cache sur un batch d'ids (B, L). Renvoie (logits, captures).

    `want` : noms à capturer parmi resid_pre.<l>, attn_out.<l>, resid_mid.<l>, mlp_out.<l>, resid_post.<l>, final.
    `hooks` : {nom: fn(tenseur) -> tenseur} pour intervenir (ablation, ajout de direction) pendant la sonde.
    """
    from mlx_lm.models.base import create_attention_mask

    hooks, cap = hooks or {}, {}

    def tap(name, x):
        if name in hooks:
            x = hooks[name](x)
        if name in want:
            cap[name] = x
        return x

    core = model.model
    h = core.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    for l, layer in enumerate(core.layers):
        h = tap(f"resid_pre.{l}", h)
        a = tap(f"attn_out.{l}", layer.self_attn(layer.input_layernorm(h), mask, None))
        h = tap(f"resid_mid.{l}", h + a)
        m = tap(f"mlp_out.{l}", layer.mlp(layer.post_attention_layernorm(h)))
        h = h + m
        tap(f"resid_post.{l}", h)
    h = tap("final", h)
    logits = core.embed_tokens.as_linear(core.norm(h))
    return logits, cap


def unembed(model, h):
    """Logit lens : état résiduel → logits, via la norme finale et l'unembedding (lié à l'embedding)."""
    return model.model.embed_tokens.as_linear(model.model.norm(h))


def attn_detail(layer, x, mask):
    """Rejoue l'attention d'une couche sur x (déjà normalisé). Renvoie (z, poids) :
    z (B, L, 16, 128) = sortie de chaque tête avant o_proj, poids (B, 16, L, L) = patterns d'attention."""
    a = layer.self_attn
    B, L, _ = x.shape
    q = a.rope(a.q_norm(a.q_proj(x).reshape(B, L, a.n_heads, -1)).transpose(0, 2, 1, 3))
    k = a.rope(a.k_norm(a.k_proj(x).reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3))
    v = a.v_proj(x).reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
    rep = a.n_heads // a.n_kv_heads
    k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)).astype(mx.float32) * a.scale
    if mask is not None:
        scores = scores + mx.triu(mx.full((L, L), -1e9), k=1)
    w = mx.softmax(scores, axis=-1)
    z = (w.astype(v.dtype) @ v).transpose(0, 2, 1, 3)
    return z, w


def forward_heads(model, ids, positions):
    """Forward qui renvoie, pour chaque position demandée (une par ligne du batch), la contribution au résiduel de
    chaque tête (28, 16, 1024), de chaque MLP (28, 1024), les patterns d'attention depuis cette position
    (28, 16, L), le résiduel final et les logits. Batch de 1 pour rester simple."""
    core = model.model
    h = core.embed_tokens(ids)
    L = ids.shape[1]
    pos = positions
    heads, mlps, pats = [], [], []
    for layer in core.layers:
        z, w = attn_detail(layer, layer.input_layernorm(h), True)
        Wo = layer.self_attn.o_proj.weight  # (1024, 2048)
        zp = z[0, pos]  # (16, 128)
        heads.append(mx.stack([zp[i] @ Wo[:, i * 128 : (i + 1) * 128].T for i in range(16)]))
        pats.append(w[0, :, pos, :])
        h = h + layer.self_attn.o_proj(z.reshape(1, L, -1))
        mo = layer.mlp(layer.post_attention_layernorm(h))
        mlps.append(mo[0, pos])
        h = h + mo
    logits = core.embed_tokens.as_linear(core.norm(h))
    out = dict(heads=mx.stack(heads).astype(mx.float32), mlps=mx.stack(mlps).astype(mx.float32), pats=mx.stack(pats), final=h[0, pos].astype(mx.float32), logits=logits[0, pos].astype(mx.float32), embed=core.embed_tokens(ids)[0, pos].astype(mx.float32))
    mx.eval(out)
    return out


def dla_direction(model, final, tok_plus, tok_minus):
    """Direction du résiduel telle que contribution · direction = part du logit(plus) − logit(minus), la norme finale
    étant gelée (linéarisation au point observé)."""
    g = model.model.norm.weight.astype(mx.float32)
    E = model.model.embed_tokens.weight
    d = (E[tok_plus] - E[tok_minus]).astype(mx.float32)
    rms = mx.sqrt(mx.mean(final * final) + 1e-6)
    return d * g / rms
