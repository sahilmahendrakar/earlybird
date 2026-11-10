#!/bin/sh
# Text-model baselines for the anticipation paper (eval/TAG/pre.sh on an eval machine): each
# language model guesses the next word from (a) what p29 (no anticipation) has on screen and
# (b) the perfect transcript, words appearing as they end (--oracle), on the general test sets.
# Logs to s3://…/results/ghost/LM-on-p29.SET.jsonl and LM-oracle.SET.jsonl.
# usage: sh scripts/lm_round.sh "SHORT=HF_ID ..."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
PATH=$HOME/.local/bin:$PATH
LMS=${1:-"smollm135=HuggingFaceTB/SmolLM2-135M qwen06=Qwen/Qwen3-0.6B qwen17=Qwen/Qwen3-1.7B"}
SETS="librispeech_test_clean ps_test tedlium_test kokoro_test"
mkdir -p results/ghost
aws s3 sync $B/runs/p29_from_tiny/step10000 ckpt/p29 --only-show-errors
for lm in $LMS; do uv run python -c "from huggingface_hub import snapshot_download as s; s('${lm#*=}')" > /dev/null 2>&1; done
for d in $SETS; do
  for lm in $LMS; do
    name=${lm%%=*}; n=300; [ "$name" = qwen17 ] && n=150
    uv run python scripts/ghost_demo.py data/$d $name-oracle.$d --lm ${lm#*=} --oracle --device cpu --limit $n --threads 2 > results/ghost/$name-oracle.$d.txt 2>&1 &
    [ "$name" = qwen06 ] || uv run python scripts/ghost_demo.py data/$d $name-on-p29.$d --lm ${lm#*=} --device cpu --model ckpt/p29 --limit $n --threads 2 > results/ghost/$name-on-p29.$d.txt 2>&1 &
  done
done
wait
aws s3 sync results/ghost $B/results/ghost --only-show-errors --exclude '*' --include '*-oracle.*' --include '*-on-p29.*'
sh scripts/runlog.sh "" "lm round: $(ls results/ghost/*.jsonl | wc -l) logs on S3 results/ghost/"
