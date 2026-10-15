# Prompt de génération du benchmark indépendant

À donner tel quel à GPT 6.1 Sol (raisonnement high) dans OpenCode, lancé à la racine du repo. Le benchmark actuel
(`benchmark/`) et une partie du train sont écrits par Claude, le reste du train par DeepSeek : celui-ci doit être
écrit par une troisième famille de modèles, sans voir les deux autres.

---

You are building an evaluation benchmark for a small model that writes JavaScript regular expressions from
natural-language descriptions. The benchmark must be **independent** of the training data and of an existing
benchmark written by other models. Your cases will be the ground truth for measuring the trained model, so
correctness and lack of ambiguity matter more than speed.

## Isolation rules (strict)

- You may read only: `scripts/benchmark.py`, `scripts/common.py`, `scripts/validate.py` (to understand the format
  and the checks) and `configs/generation.yaml`.
- Do NOT open anything under `benchmark/`, `data/`, `results/`, `adapters/`, `models/`, `logs/`, nor `README.md`,
  `NOTES.md`, `configs/taxonomy.yaml` or any other script. Do not search them either. Seeing them would make the
  benchmark copy the existing data's style and defeat its purpose.
- Write only inside `benchmark-gpt/`. Do not modify any other file.

## The task the model is evaluated on

The model receives a description in English (and, for about a third of the cases, one to three example strings
drawn automatically from your tests). It must answer a single JavaScript regex literal `/pattern/flags`. Its regex
is run with `RegExp.prototype.test(string)` on every test string, in Node 22, and must accept every `match`
string and reject every `no_match` string.

Flavor: ECMAScript 2018. Allowed flags: `i`, `m`, `s`, `u` only (never `g`, `y`, `d`, `v`). Allowed: classes,
quantifiers, capturing and named groups, backreferences (`\1`, `\k<name>`), lookahead and lookbehind, `\p{…}` with
`u`. Not allowed: the `v` flag, inline modifiers, anything newer than ES2018.

## What to write

One YAML file per category, `benchmark-gpt/<category>.yaml`, each holding **20 cases**, 420 in total, for these 21
categories:

- `numbers`: Integers, decimals and numeric formats.
- `money`: Prices and amounts in various currencies and locales.
- `dates-times`: Dates, times, timestamps and durations.
- `identifiers`: Generated identifiers, hashes and codes.
- `web`: URLs, domains and front-end formats.
- `network`: Addresses and network identifiers.
- `email-accounts`: Emails, usernames and social handles.
- `phone-numbers`: Phone numbers in national and international formats.
- `postal-official`: Postal codes and official identifiers (format only, no checksum).
- `passwords`: Password and secret policies.
- `files-paths`: File names, extensions and paths.
- `versions`: Version numbers and release tags.
- `code-syntax`: Identifiers and naming conventions in code.
- `dev-tooling`: Formats from development tools and workflows.
- `logs`: Log lines and server output.
- `text`: Words, sentences and text constraints.
- `unicode`: International text and Unicode-aware validation.
- `structured-data`: Small structured formats inside a single string.
- `multiline`: Multi-line input where line-based semantics matter.
- `exclusions`: Validation by exclusion and negative constraints.
- `measurements`: Measurements, units and coordinates.

Each file starts with a comment line `# Benchmark GPT, catégorie <category>. Format et contrôles : scripts/benchmark.py.`
and is a YAML list of cases with exactly these keys:

```yaml
- id: gpt-<category>-01          # unique, two-digit counter per category
  difficulty: simple             # simple | medium | hard
  status: ok                     # ok once checked; todo if you are not sure; drop to discard
  description: >-
    The need, in English, as a developer would state it.
  regex: /^...$/
  match:                         # strings that must be accepted
    - "..."
  no_match:                      # strings that must be rejected
    - "..."
  pitfall: >-
    What a naive or quick solution gets wrong on this case.
```

Use double-quoted YAML strings for test strings, with escapes (`\n`, `\t`, `é`…) for special characters.

### Difficulty mix, per category

5 simple, 8 medium, 7 hard.

- simple: one clear format with few rules.
- medium: several constraints combined (bounded numeric ranges, optional parts, alternatives, case rules) or a
  classic pitfall such as anchoring or escaping.
- hard: genuinely hard for a developer answering quickly: interacting constraints, exclusions, lookarounds,
  backreferences, Unicode, multi-line input, or many edge cases. Hard cases must be hard because of what is
  required, not because the description is vague.

### Coverage across the whole benchmark

Make sure the rarer constructs are well represented. Across the 420 cases, aim for at least: 60 cases whose natural
solution uses a lookahead, 25 a backreference or named backreference, 20 a lookbehind, 25 the `m` flag or explicit
line handling, 15 the `s` flag, 20 `\p{…}`, 30 a deliberately unanchored "contains" search, 40 numeric ranges
written as alternations (e.g. 0-255, 1-31).

### Descriptions

- Realistic needs, as found in tickets, form validation, log parsing, code review or developer forums. Vary the
  register: a terse search-box fragment, plain sentences, a precise spec with bullet points, or a business need
  from which the developer must infer the format.
- A description states every rule needed to decide whether a string is valid, except conventions any developer
  assumes for that format. It never mentions regex syntax and never gives example strings.
- One reading only: two careful developers reading the description must agree on every test string. When in doubt,
  make the description more precise or drop the string from the tests.
- Format only: no checksums (Luhn, IBAN keys…) and no calendar facts such as leap years; say so when the format
  has a checksum.
- Prefer needs that are not textbook examples. Avoid the most common canned regexes (plain IPv4, plain email, plain
  ISO date, plain semver) unless you add real, specific constraints.

### Tests

- At least 6 `match` and 6 `no_match` strings per case (about one in three cases shows 1 to 3 of them in the prompt,
  and at least 5 must stay hidden on each side). Aim for 8 to 12 on each side.
- `match` covers the variations: minimum and maximum bounds, each alternative, optional parts present and absent.
- `no_match` is mostly near-misses: each breaks exactly one rule (one character too many or too few, a value just
  out of bounds, a wrong separator, the wrong case, a forbidden character, leading or trailing text or whitespace).
  Cover every rule of the description with at least one near-miss.
- For whole-string needs, include at least one near-miss with extra leading text and one with extra trailing text,
  so that anchoring is tested. For "contains" needs, include valid strings with surrounding text.
- Every test string must follow unambiguously from the description alone, not from your regex.

### Reference regex

- Anchor with `^` and `$` unless the need is explicitly about containing something.
- Readable, and no nested quantifiers that can backtrack catastrophically (the runner has a timeout).
- It must pass all its tests.

## How to work

Work category by category. For each category:

1. Write the 20 cases.
2. Run `uv run scripts/benchmark.py check --dir benchmark-gpt` (it checks every file in the folder).
3. Fix every ERROR (invalid literal, reference disagreeing with its tests, too few tests, a string on both sides,
   duplicate id).
4. Treat warnings seriously:
   - "tests peu discriminants": the reference without anchors, with relaxed quantifiers or with widened classes
     also passes all tests. Add near-misses that catch it.
   - "description proche du train" or "regex identique à une regex du train": the case overlaps the training data.
     Change the need substantially.
5. Before moving on, re-read each hard case cold, as if you had never seen the regex: does every test string follow
   from the description alone? If not, fix the description or the tests, or set `status: todo`.

When all 21 files are done, run the check one last time, then
`uv run scripts/benchmark.py build --dir benchmark-gpt --output data/eval/benchmark-gpt.jsonl`.

## Final report

Reply with: the number of cases per difficulty, the coverage counts for the constructs above, the cases left as
`todo` or `drop` and why, and any check warning you chose to keep, with the reason.
