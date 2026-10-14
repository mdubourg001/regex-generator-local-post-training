"""Filtre les besoins générés par le professeur, puis prépare les données d'entraînement.

Garde-fous (README, section 5), dans l'ordre :
  1. réponses lisibles, regex au format littéral ES2018 ;
  2. tests suffisants : assez de positifs, de négatifs et de négatifs « presque bons », aucune chaîne à la fois valide
     et invalide ;
  3. la regex de référence passe tous les tests (exécutés dans Node) ;
  4. discrimination : la référence sans ancres doit rater au moins un test, de même que la référence aux
     quantificateurs relâchés et celle aux classes élargies ;
  5. pas de doublon de description ;
  6. décontamination : rien qui recoupe le benchmark (même regex, ou description proche).

Ensuite : tirage des exemples montrés dans le prompt (retirés des tests cachés), split par lot, format SFT.

    uv run scripts/validate.py

Sorties :
    data/prepared/{train,valid,heldout}.jsonl  exemples complets (format du README, section 4)
    data/sft/{train,valid}.jsonl               conversations pour mlx_lm.lora
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from common import RegexRunner, build_messages, parse_literal
from generate import CONFIG, RAW_DIR, TAXONOMY, log_paths, read_log

PREPARED_DIR = Path("data/prepared")
BENCHMARK_DIR = Path("benchmark")
# Au-delà, deux descriptions sont considérées comme le même besoin (similarité de Jaccard sur mots et bigrammes).
SIMILARITY_THRESHOLD = 0.5
SURROGATE = re.compile("[\ud800-\udfff]")
# Au-delà, un exemple est écarté du SFT : la mémoire des logits (vocabulaire de 151k tokens) croît avec la longueur des
# séquences, et le tronquer couperait la réponse. Doit rester ≤ max_seq_length de configs/sft.yaml.
MAX_SFT_TOKENS = 512
TOKENIZER = "Qwen/Qwen3-0.6B"  # même tokenizer pour toute la famille Qwen3
SFT_DIR = Path("data/sft")


# --- Analyse d'un pattern ---------------------------------------------------------------------------------------------


def scan(pattern: str) -> dict[str, list[tuple[int, int]]]:
    """Repère, hors échappements, les classes de caractères, les quantificateurs {n,m} et les `.` du pattern."""
    spans = {"classes": [], "braces": [], "dots": []}
    i, class_start = 0, None
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            i += 2
            continue
        if class_start is not None:
            if c == "]":
                spans["classes"].append((class_start, i + 1))
                class_start = None
        elif c == "[":
            class_start = i
        elif c == ".":
            spans["dots"].append((i, i + 1))
        elif c == "{" and (m := re.match(r"\{(\d+)(,\d*)?\}", pattern[i:])):
            spans["braces"].append((i, i + m.end()))
            i += m.end()
            continue
        i += 1
    return spans


def is_anchored(pattern: str) -> bool:
    return pattern.startswith("^") or (pattern.endswith("$") and not re.search(r"(?<!\\)(\\\\)*\\\$$", pattern))


def unanchored(pattern: str) -> str | None:
    """La référence sans ses ancres de début et de fin, ou None si elle n'en a pas."""
    if not is_anchored(pattern):
        return None
    stripped = pattern[1:] if pattern.startswith("^") else pattern
    if stripped.endswith("$") and not re.search(r"(?<!\\)(\\\\)*\\\$$", stripped):
        stripped = stripped[:-1]
    return stripped


def mutants(pattern: str) -> dict[str, str]:
    """Variantes globalement plus permissives de la référence : tous les quantificateurs {n,m} relâchés en + ou * d'un
    coup, et toutes les classes élargies à n'importe quel caractère d'un coup. Des tests discriminants doivent
    attraper chacune au moins une fois.

    On ne mute pas chaque quantificateur séparément : dans une IPv4, il faudrait des tests qui ciblent chaque octet.
    """
    spans = scan(pattern)
    result = {}
    braces = [(s, e) for s, e in spans["braces"] if pattern[s:e] not in ("{1,}", "{0,}")]  # déjà équivalents à + ou *
    if braces:
        result["quantificateurs relâchés"] = replace_spans(pattern, braces, lambda text: "*" if text.startswith("{0") else "+")
    if spans["classes"]:
        result["classes élargies"] = replace_spans(pattern, spans["classes"], lambda _: r"[\s\S]")
    return result


def replace_spans(pattern: str, spans: list[tuple[int, int]], replace) -> str:
    for start, end in sorted(spans, reverse=True):
        pattern = pattern[:start] + replace(pattern[start:end]) + pattern[end:]
    return pattern


FEATURE_PATTERNS = {
    "word_boundary": r"\\[bB]",
    "shorthand": r"\\[dDwWsS]",
    "quantifiers": r"(?<!\\)[*+]|\{\d+(,\d*)?\}",
    "alternation": r"(?<!\\)\|",
    "non_capturing": r"\(\?:",
    "backreference": r"\\[1-9]",
    "named_groups": r"\(\?<[A-Za-z_$]|\\k<",
    "lookahead": r"\(\?[=!]",
    "lookbehind": r"\(\?<[=!]",
    "unicode_property": r"\\[pP]\{",
    "unicode_escape": r"\\u\{|[\U00010000-\U0010FFFF]",
    "escaping": r"\\[.+()\[\]{}/\\|?*^$-]",
    "numeric_range": r"\[[0-9]-[0-8]\]|\[[1-9]-9\]",  # approximation : une plage de chiffres autre que 0-9
}


def features(pattern: str, flags: str) -> set[str]:
    """Fonctionnalités de la flavor utilisées par une regex (détection approximative, pour mesurer la couverture)."""
    found = {name for name, regex in FEATURE_PATTERNS.items() if re.search(regex, pattern)}
    spans = scan(pattern)
    found |= {"char_class"} if spans["classes"] else set()
    found |= {"dot"} if spans["dots"] else set()
    found |= {"anchors"} if is_anchored(pattern) else {"unanchored"}
    found |= {f"flag_{f}" for f in flags if f in "ims"}
    return found


# --- Décontamination ------------------------------------------------------------------------------------------------


def load_benchmark() -> list[dict]:
    """Tous les cas du benchmark, la catégorie venant du nom du fichier."""
    cases = []
    for path in sorted(BENCHMARK_DIR.glob("*.yaml")):
        cases += [{"category": path.stem, **c} for c in yaml.safe_load(path.read_text()) or []]
    return cases


def words(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return set(tokens) | {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}


def similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def decontaminate(examples: list[dict], benchmark: list[dict]) -> tuple[list[dict], int]:
    """Retire les exemples qui recoupent le benchmark : même regex de référence, ou description proche."""
    regexes = {c["regex"] for c in benchmark}
    benchmark_words = [words(c["description"]) for c in benchmark]
    kept = [
        e
        for e in examples
        if e["regex"] not in regexes
        and not any(similarity(words(e["description"]), w) >= SIMILARITY_THRESHOLD for w in benchmark_words)
    ]
    return kept, len(examples) - len(kept)


# --- Validation -------------------------------------------------------------------------------------------------------


def candidates(raw_dir: Path) -> tuple[list[dict], Counter]:
    """Assemble les besoins générés (description, regex, tests). Compte les rejets de lecture."""
    descriptions, tests = map(read_log, log_paths(raw_dir))
    rejected = Counter()
    result = []
    for batch_id, d in sorted(descriptions.items()):
        if not d["items"]:
            rejected["descriptions illisibles"] += len(d["slots"])
            continue
        t = tests.get(batch_id)
        if t is None:
            continue  # tests pas encore générés
        if not t["items"]:
            rejected["tests illisibles"] += len(d["items"])
            continue
        tests_by_slot = {item["slot"]: item for item in t["items"]}
        for item in d["items"]:
            if not 1 <= item["slot"] <= len(d["slots"]) or item["slot"] not in tests_by_slot:
                rejected["besoin sans tests"] += 1
                continue
            slot, item_tests = d["slots"][item["slot"] - 1], tests_by_slot[item["slot"]]
            # Sans les chaînes contenant un demi-caractère UTF-16 isolé : valides en JS, mais ni encodables en UTF-8 ni
            # tokenisables.
            as_strings = lambda key: list(  # noqa: E731
                dict.fromkeys(s for s in item_tests[key] if isinstance(s, str) and not SURROGATE.search(s))
            )
            match, near_miss, other = as_strings("match"), as_strings("near_miss"), as_strings("other_no_match")
            # Une chaîne à la fois valide et invalide : le professeur se contredit, on la retire des deux côtés.
            ambiguous = set(match) & (set(near_miss) | set(other))
            result.append(
                {
                    "id": slot["id"],
                    "batch": batch_id,
                    "category": d["category"],
                    "difficulty": slot["difficulty"],
                    "register": slot["register"],
                    "intent": slot["intent"],
                    "description": str(item["description"]).strip(),
                    "regex": str(item["regex"]).strip(),
                    "tests": {
                        "match": [s for s in match if s not in ambiguous],
                        "no_match": list(dict.fromkeys(s for s in near_miss + other if s not in ambiguous)),
                    },
                    "near_miss": [s for s in near_miss if s not in ambiguous],
                }
            )
    return result, rejected


def check(examples: list[dict], cfg: dict, runner: RegexRunner) -> tuple[list[dict], Counter]:
    """Applique les garde-fous 1 à 5. Renvoie les exemples gardés et les rejets par motif."""
    required = cfg["tests_required"]
    rejected = Counter()
    kept, to_run = [], []
    for e in examples:
        parsed = parse_literal(e["regex"])
        if parsed is None:
            rejected["regex hors format ES2018"] += 1
        elif (
            len(e["tests"]["match"]) < required["match"]
            or len(e["tests"]["no_match"]) < required["no_match"]
            or len(e["near_miss"]) < required["near_miss"]
        ):
            rejected["pas assez de tests"] += 1
        else:
            to_run.append((e, *parsed))

    # Référence, puis référence sans ancres et mutants : un seul passage dans Node.
    jobs, owners = [], []
    for k, (e, pattern, flags) in enumerate(to_run):
        variants = [("reference", pattern)]
        if (stripped := unanchored(pattern)) is not None:
            variants.append(("unanchored", stripped))
        variants += list(mutants(pattern).items())
        for kind, variant in variants:
            jobs.append({"pattern": variant, "flags": flags, **e["tests"]})
            owners.append((k, kind))
    results = defaultdict(list)
    for (k, kind), result in zip(owners, runner.run(jobs)):
        results[k].append((kind, result))

    seen = set()
    for k, (e, pattern, flags) in enumerate(to_run):
        runs = results[k]
        reference = runs[0][1]
        passed = lambda r: "error" not in r and not r.get("timeout") and all(r["match"]) and all(r["no_match"])  # noqa: E731
        survivors = [kind for kind, r in runs[1:] if kind != "unanchored" and passed(r)]
        key = re.sub(r"\W+", " ", e["description"].lower()).strip()
        if "error" in reference:
            rejected["regex invalide"] += 1
        elif reference.get("timeout"):
            rejected["timeout (ReDoS)"] += 1
        elif not passed(reference):
            rejected["référence en désaccord avec les tests"] += 1
        elif any(kind == "unanchored" and passed(r) for kind, r in runs):
            rejected["ancrage non testé"] += 1
        elif survivors:
            rejected[f"tests peu discriminants ({survivors[0]})"] += 1
        elif key in seen:
            rejected["doublon"] += 1
        else:
            seen.add(key)
            kept.append({**e, "features": sorted(features(pattern, flags))})
    return kept, rejected


def add_examples(e: dict, cfg: dict, share: float) -> dict:
    """Tire, pour une part des exemples, 1 à `max_examples` chaînes montrées dans le prompt, retirées des tests."""
    rng = random.Random(f"examples-{e['id']}")
    if rng.random() >= share:
        return e
    required = cfg["tests_required"]
    match, no_match = list(e["tests"]["match"]), list(e["tests"]["no_match"])
    shown = {"match": [], "no_match": []}
    for _ in range(rng.randint(1, cfg["max_examples"])):
        # Au moins un positif, puis positif ou négatif au hasard, tant qu'il reste assez de tests cachés.
        side = "match" if not shown["match"] or rng.random() < 0.6 else "no_match"
        pool = match if side == "match" else no_match
        if len(pool) <= required[side]:
            continue
        shown[side].append(pool.pop(rng.randrange(len(pool))))
    return {**e, "examples": shown, "tests": {"match": match, "no_match": no_match}}


def sft_lengths(conversations: list[list[dict]]) -> list[int]:
    """Longueur en tokens de chaque conversation, telle que mlx_lm.lora la tokenise."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in conversations]
    return [len(ids) for ids in tokenizer(texts)["input_ids"]]


def split_of(batch: str, split: dict) -> str:
    x = random.Random(f"split-{batch}").random()
    return "train" if x < split["train"] else "valid" if x < split["train"] + split["valid"] else "heldout"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="dossier des journaux de generate.py")
    parser.add_argument("--extra-raw-dir", type=Path, help="complément hors plan principal : tous ses lots vont au train")
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument("--sft-dir", type=Path, default=SFT_DIR)
    parser.add_argument("--stats-only", action="store_true", help="affiche les statistiques sans écrire les données")
    args = parser.parse_args()

    cfg, taxonomy = yaml.safe_load(CONFIG.read_text()), yaml.safe_load(TAXONOMY.read_text())
    examples, rejected = candidates(args.raw_dir)
    # Après le plan principal : en cas de doublon, c'est l'exemple du complément qui est écarté. Ses lots vont tous au
    # train, pour que valid et held-out restent comparables d'un run à l'autre.
    extra_batches = set()
    if args.extra_raw_dir:
        extra, extra_rejected = candidates(args.extra_raw_dir)
        examples += extra
        rejected += extra_rejected
        extra_batches = {e["batch"] for e in extra}
    generated = len(examples) + sum(rejected.values())
    with RegexRunner() as runner:
        kept, more = check(examples, cfg, runner)
    rejected += more
    # Le benchmark est la référence figée : c'est le train qui s'écarte de lui, pas l'inverse.
    kept, n_contaminated = decontaminate(kept, load_benchmark())
    if n_contaminated:
        rejected["recoupe le benchmark"] += n_contaminated
    kept = [add_examples(e, cfg, taxonomy["examples_share"]) for e in kept]

    splits = defaultdict(list)
    for e in kept:
        splits["train" if e["batch"] in extra_batches else split_of(e["batch"], cfg["split"])].append(e)
    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    args.sft_dir.mkdir(parents=True, exist_ok=True)
    too_long = 0
    for name in () if args.stats_only else ("train", "valid", "heldout"):
        with (args.prepared_dir / f"{name}.jsonl").open("w") as f:
            for e in splits[name]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        if name != "heldout":
            conversations = [build_messages(e, with_answer=True) for e in splits[name]]
            lengths = sft_lengths(conversations)
            too_long += sum(n > MAX_SFT_TOKENS for n in lengths)
            with (args.sft_dir / f"{name}.jsonl").open("w") as f:
                for messages, n in zip(conversations, lengths):
                    if n <= MAX_SFT_TOKENS:
                        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    cost = sum(r["usage"].get("cost", 0) for log in log_paths(args.raw_dir) for r in read_log(log).values())
    print(f"{generated} besoins générés, {len(kept)} gardés ({len(kept) / max(generated, 1):.0%}), coût {cost:.3f} $")
    print(f"  soit {cost / max(len(kept), 1) * 1000:.3f} $ pour 1 000 exemples gardés")
    print("  " + ", ".join(f"{name} {len(splits[name])}" for name in ("train", "valid", "heldout")))
    if too_long:
        print(f"  dont {too_long} exemples de plus de {MAX_SFT_TOKENS} tokens, écartés du SFT")
    print("\nRejets :")
    for reason, n in rejected.most_common():
        print(f"  {reason:40} {n:>6} ({n / max(generated, 1):.1%})")

    by_category = Counter(e["category"] for e in kept)
    print("\nPar catégorie :")
    for c in taxonomy["categories"]:
        print(f"  {c['id']:24} {by_category[c['id']]:>6}")
    print("\nPar difficulté : " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(e["difficulty"] for e in kept).items())))
    print("Avec exemples : " + f"{sum('examples' in e for e in kept) / max(len(kept), 1):.0%}")
    coverage = Counter(f for e in kept for f in e["features"])
    print("\nCouverture de la flavor (part des regex gardées qui utilisent chaque fonctionnalité) :")
    for feature in taxonomy["features"]:
        print(f"  {feature:18} {coverage[feature] / max(len(kept), 1):>6.1%}")


if __name__ == "__main__":
    main()
