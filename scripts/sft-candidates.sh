#!/usr/bin/env bash
# SFT + éval (held-out, regex101) des modèles candidats, à la suite : bash scripts/sft-candidates.sh [nom…]
# ~3 h 30–4 h pour les 4, pic mémoire 5–8 Go. Sans argument : tous ; sinon ex. `bash scripts/sft-candidates.sh coder-0.5b gemma3-270m`.
cd "$(dirname "$0")/.."
want=" $* "
for spec in "coder-0.5b:Qwen/Qwen2.5-Coder-0.5B-Instruct" "gemma3-270m:unsloth/gemma-3-270m-it" "qwen3.5-0.8b:Qwen/Qwen3.5-0.8B" "gemma3-1b:unsloth/gemma-3-1b-it" "granite4-350m:ibm-granite/granite-4.0-350m"; do
  n=${spec%%:*}; m=${spec#*:}
  [ -n "$*" ] && [[ "$want" != *" $n "* ]] && continue
  echo "### $(date +%H:%M) SFT $n"
  uv run mlx_lm.lora --config configs/sft-$n.yaml > logs/sft-$n.log 2>&1 || { echo "FAIL $n"; tail -5 logs/sft-$n.log; continue; }
  grep "Peak mem" logs/sft-$n.log | tail -1; grep "Val loss" logs/sft-$n.log | tail -1
  for d in "heldout:data/prepared/heldout.jsonl" "real:data/eval/benchmark-regex101.jsonl"; do
    uv run scripts/eval.py --name sft-$n-${d%%:*} --model $m --adapter adapters/sft-$n --data ${d#*:} --batch-size 8 2>&1 | grep -A2 "sur data"
  done
done
echo "### $(date +%H:%M) done"
