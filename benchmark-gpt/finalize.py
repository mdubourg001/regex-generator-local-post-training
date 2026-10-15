"""Freeze, validate, export and write an isolation/coverage manifest."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT/"rebuild.py")], check=True)
subprocess.run([sys.executable, str(ROOT/"check.py"), "build"], check=True)
subprocess.run([sys.executable, str(ROOT/"audit.py")], check=True)
check = json.loads((ROOT/"check-report.json").read_text())
audit = json.loads((ROOT/"audit-report.json").read_text())
manifest = dict(
    source="21 category YAML files, 20 independently authored cases per category",
    output="benchmark-gpt/benchmark-gpt.jsonl",
    cases=check["cases"], difficulties=check["difficulties"], statuses=check["statuses"],
    coverage={**check["coverage"], "numeric_range_alternation": audit["numeric_range_alternation_cases"]},
    checks=dict(node=audit["node_version"], errors=check["errors"], mutation_warnings=check["mutation_warnings"],
                tests=audit["total_tests"], spec_assertions=audit["spec_assertions"],
                extra_spec_probes=audit["extra_spec_probe_strings"],
                examples=audit["cases_with_examples"], minimum_tests_per_side=8, minimum_hidden_tests_per_side=5),
    coverage_notes=[
        "Counts are numbers of cases, not numbers of occurrences.",
        "Lookahead excludes the universal strict-end guard.",
        "Numeric ranges use the conservative manually enumerated case inventory in audit-report.json, not the checker's approximate 89-case heuristic.",
        "Contains counts exclude the ^...$ /m line search; including that expressly line-local search gives 46 cases.",
        "m or explicit line handling counts only syntax that actually represents line boundaries, not a dot-all body's ability to contain LF."
    ],
    independence=dict(read_project_files=["scripts/benchmark.py", "scripts/common.py", "scripts/validate.py", "configs/generation.yaml"],
                      forbidden_content_read=False, official_commands_run=False,
                      overlap_checks="Unavailable under the user's confirmed strict-isolation choice; neither training nor other-model benchmark read.",
                      writes="benchmark-gpt/ only; pre-existing PROMPT.md left unread and untouched"),
    review="All 147 hard cases reread against descriptions; ambiguous wording corrected. Shared Unicode cluster and capsule formats additionally checked by non-regex procedural predicates.",
    commands=["python3 benchmark-gpt/rebuild.py", "python3 benchmark-gpt/check.py", "python3 benchmark-gpt/check.py build", "python3 benchmark-gpt/audit.py"]
)
(ROOT/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
print("Final manifest: benchmark-gpt/manifest.json")
