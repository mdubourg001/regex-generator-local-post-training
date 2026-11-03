# Prompt d'annotation du complément regex101

À donner tel quel à GPT 6.1 Sol (raisonnement high) dans OpenCode, lancé à la racine du repo, après
`uv run scripts/regex101.py select`. Le benchmark est écrit par Claude : les annotations doivent venir d'une autre
famille de modèles, sans le voir. Le travail est reprenable : relancer le même prompt reprend aux parts sans `out<k>.json`.

---

You are turning real-world regex needs into training data for a small model that writes JavaScript regular
expressions from natural-language descriptions. The entries come from the community library of regex101: real needs,
but with missing or sloppy descriptions, often wrong regexes, and unlabelled sample strings. Your annotations become
the training ground truth, so correctness and lack of ambiguity matter more than speed.

## Isolation rules (strict)

- You may read only: `scripts/regex101.py`, `scripts/common.py`, `scripts/validate.py`, `configs/generation.yaml`,
  and the files in `data/regex101/work/`.
- Do NOT open anything else under `data/`, nor `benchmark/`, `benchmark-gpt/`, `results/`, `adapters/`, `models/`,
  `logs/`, `README.md`, `NOTES.md`, `configs/taxonomy.yaml` or any other script. Do not search them either.
- Write only `data/regex101/work/out<k>.json` files. Do not modify any other file.

## Input

`data/regex101/work/shard<k>.json` holds a list of entries:

- `id`: keep it as is.
- `title`, `description`: what the author said the regex is for. The description is often empty, a copy of sample
  strings, or unrelated context.
- `original_regex`: the author's regex, in whatever flavor (often PCRE or Python). Treat it as a hint about the
  intended format, never as ground truth: it is frequently too permissive, too strict or simply broken.
- `text`: an excerpt of the author's sample strings, one per line in most cases. They are NOT labelled: some are
  valid, some are deliberately invalid, some are noise. Reuse them as realistic test strings only when you can decide
  their side from your own description.

## The task the trained model will face

It receives a description in English (and, for about a third of the cases, one to three example strings drawn
automatically from your tests). It must answer a single JavaScript regex literal `/pattern/flags`. Its regex is run
with `RegExp.prototype.test(string)` on every test string, in Node 22, and must accept every `match` string and reject
every other one.

Flavor: ECMAScript 2018. Allowed flags: `i`, `m`, `s`, `u` only. Allowed: classes, quantifiers, capturing and named
groups, backreferences, lookahead and lookbehind, `\p{…}` with `u`. Not allowed: the `v` flag, inline modifiers,
possessive quantifiers, atomic groups, conditionals, POSIX classes, `\A`/`\Z`, anything newer than ES2018.

## What to write

For each shard `shard<k>.json`, write `out<k>.json`: a JSON object keyed by entry id, one value per entry of the shard.

```json
{
  "x00042": {
    "status": "ok",
    "category": "dates-times",
    "difficulty": "medium",
    "description": "The need, in English, as a developer would state it.",
    "regex": "/^...$/",
    "match": ["..."],
    "near_miss": ["..."],
    "other_no_match": ["..."]
  },
  "x00043": { "status": "drop", "reason": "title and regex do not say what is valid" }
}
```

### When to drop

Set `"status": "drop"` with a short `reason` when:

- you cannot tell what the author wanted to validate (`"<66.0.6"`, `"button audio"`, `"test"`…);
- the need is extraction or parsing of a larger text (capturing fields from a log line, finding tags in HTML) rather
  than deciding whether a whole string is valid, and cannot be honestly restated as a validation;
- the need is specific to one private system with no inferable rules;
- the need requires a checksum or calendar facts (Luhn, IBAN key, leap years) to be correct.

Dropping is expected and fine: a third to a half of the entries being dropped is normal. Never invent a need that the
entry does not suggest.

### Kept entries

- `category`: one of `numbers`, `money`, `dates-times`, `identifiers`, `web`, `network`, `email-accounts`,
  `phone-numbers`, `postal-official`, `passwords`, `files-paths`, `versions`, `code-syntax`, `dev-tooling`, `logs`,
  `text`, `unicode`, `structured-data`, `multiline`, `exclusions`, `measurements`.
- `difficulty`: `simple` (one clear format with few rules), `medium` (several constraints combined, or a classic
  pitfall such as anchoring or escaping), `hard` (interacting constraints, exclusions, lookarounds, backreferences,
  Unicode, multi-line input, many edge cases).
- `description`: rewrite the need as the author should have written it. Keep the author's intent, domain and
  vocabulary; fix it into a precise spec. It states every rule needed to decide whether a string is valid, except
  conventions any developer assumes for that format. It never mentions regex syntax and never gives example strings.
  One reading only: two careful developers must agree on every test string. Vary the register as real users do: a
  terse search-box fragment, plain sentences, a precise spec, or a business need.
- `regex`: a correct, readable ES2018 literal implementing your description. Anchor it with `^` and `$` unless the
  need is explicitly about containing something. No nested quantifiers that can backtrack catastrophically (the
  runner has a timeout). Fix the author's bugs rather than copying them.
- Tests: 8 `match`, 6 `near_miss`, 3 `other_no_match`, all distinct.
  - `match` covers the variations: minimum and maximum bounds, each alternative, optional parts present and absent.
  - `near_miss` breaks exactly one rule each: one character too many or too few, a value just out of bounds, a wrong
    separator, the wrong case, a forbidden character. For whole-string needs, include one with extra leading text and
    one with extra trailing text.
  - `other_no_match`: clearly invalid strings.
  - Every test string must follow from the description alone, not from your regex. Write long strings in full, never
    as code (`"a".repeat(30)`).

## How to work

Work shard by shard, in order, skipping shards that already have an `out<k>.json`. For each shard:

1. Write `out<k>.json` with every entry of the shard, kept or dropped.
2. Run `uv run scripts/regex101.py check --shards <k>`. It applies the training-data guardrails to each kept entry
   and prints the reason for each rejection:
   - "référence en désaccord avec les tests": your regex and your tests disagree; the listed strings say which. Decide
     from the description which one is wrong.
   - "ancrage non testé": the regex without its anchors also passes; add near-misses with extra leading and trailing
     text.
   - "tests peu discriminants (quantificateurs relâchés | classes élargies)": a more permissive variant of your regex
     also passes all tests; add near-misses that catch it (a length just out of bounds, a forbidden character).
   - "pas assez de tests", "regex hors format ES2018", "timeout (ReDoS)": self-explanatory.
   - "recoupe le benchmark": the entry overlaps a held-out evaluation set you must not see. Change the need
     substantially (different constraints, not just rewording), or drop it.
3. Fix and re-run until the shard has no rejection, or drop the entries you cannot make unambiguous.

Do not stop between shards to report. When every shard is done, run `uv run scripts/regex101.py check` once on all of
them.

## Final report

Reply with: the number of entries kept and dropped, the main drop reasons with rough counts, the split of kept entries
by category and difficulty, and any recurring problem in the source data worth knowing.
