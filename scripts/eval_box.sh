#!/bin/sh
# One evaluation round on a throwaway CPU machine (started by scripts/launch_eval.sh, which
# also terminates it: the instance shuts itself down at the end, and shutdown = terminate).
# Reads s3://…/eval/TAG/plan.txt, one system per line:
#   NAME RECOGNIZER [CHECKPOINT]
# CHECKPOINT is an S3 key under the bucket (runs/s10_from_tiny/step20000) or a Hub id; the
# recognizer gets it as EARLYBIRD_MODEL. Every system runs on every test set in SETS with
# one worker per core, then 20 utterances alone for compute per step. Raw logs go to
# s3://…/results/raw/NAME/, the scored tables (with every system already on S3) to
# s3://…/results/eval/TAG.md, and a line to the run log.
# Optional: eval/TAG/pre.sh runs first (e.g. makes the checkpoints the plan names and puts them
# on S3); eval/TAG/workers overrides the worker count; eval/TAG/extra_sets adds test sets and
# eval/TAG/only_sets replaces them.
# usage (on the eval machine, as ubuntu, from the repo): scripts/eval_box.sh TAG
cd "$(dirname "$0")/.."
TAG=$1
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
PATH=$HOME/.local/bin:$PATH
log() { sh scripts/runlog.sh "" "eval $TAG: $*"; }
SETS="librispeech_test_clean_full librispeech_test_other_full ps_test ps_test_wer tedlium_test earnings22_test kokoro_test${EXTRA_SETS:+ $EXTRA_SETS}"
W=$(nproc)
trap 'log "failed at: $step"; aws s3 cp runs/eval.log $B/results/eval/$TAG.log --only-show-errors' EXIT
set -e
mkdir -p runs data results/raw
exec >> runs/eval.log 2>&1
step=sets
for d in librispeech_test_clean librispeech_test_other ps_test ps_test_wer kokoro_test librispeech_test_clean_full librispeech_test_other_full tedlium_test earnings22_test slurp_test slurp_devel; do
  aws s3 sync $B/data/$d data/$d --only-show-errors --exclude 'dl/*'
done
for split in clean other; do   # full LibriSpeech test sets, made once and kept on S3
  d=data/librispeech_test_${split}_full
  if [ ! -s $d/manifest.jsonl ]; then
    p=$(uv run python -c "from huggingface_hub import hf_hub_download as h; print(h('gilkeyio/librispeech-alignments', 'data/test_$split-00000-of-00001.parquet', repo_type='dataset', local_dir='/tmp/ls'))" | tail -1)
    uv run python scripts/prep_librispeech.py $p $d all
    aws s3 sync $d $B/data/librispeech_test_${split}_full --only-show-errors
    log "made librispeech_test_${split}_full: $(tail -1 runs/eval.log)"
  fi
done
for set in tedlium earnings22; do   # out-of-domain sets (evaluation only), made once and kept on S3
  d=data/${set}_test
  if [ ! -s $d/manifest.jsonl ]; then
    n=""; [ $set = earnings22 ] && n="--n 1000"
    uv run python scripts/prep_hf_test.py $set $d $n 2>&1 | tail -1
    aws s3 sync $d $B/data/${set}_test --only-show-errors --exclude 'dl/*'
    log "made ${set}_test: $(tail -1 runs/eval.log)"
  fi
done
step=pre   # optional setup for this round (make checkpoints, install a recognizer): eval/TAG/pre.sh
if aws s3 cp $B/eval/$TAG/pre.sh runs/pre.sh --only-show-errors 2>/dev/null; then sh -e runs/pre.sh; log "pre.sh done"; fi
aws s3 cp $B/eval/$TAG/workers runs/workers --only-show-errors 2>/dev/null && W=$(cat runs/workers)
aws s3 cp $B/eval/$TAG/extra_sets runs/extra_sets --only-show-errors 2>/dev/null && EXTRA_SETS=$(cat runs/extra_sets) && SETS="$SETS $EXTRA_SETS"
aws s3 cp $B/eval/$TAG/only_sets runs/only_sets --only-show-errors 2>/dev/null && SETS=$(cat runs/only_sets)
step=plan
aws s3 cp $B/eval/$TAG/plan.txt runs/plan.txt
log "started on $(curl -s -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token | xargs -I{} curl -s -H 'X-aws-ec2-metadata-token: {}' http://169.254.169.254/latest/meta-data/instance-type), $W workers: $(cut -d' ' -f1 runs/plan.txt | tr '\n' ' ')"
while read -r name rec ckpt; do
  [ -z "$name" ] && continue
  step="$name"
  model=$ckpt
  case "$ckpt" in runs/*) aws s3 sync $B/$ckpt ckpt/$ckpt --only-show-errors; model=ckpt/$ckpt ;; esac
  for d in $SETS; do
    EARLYBIRD_MODEL=$model uv run python scripts/run_stream.py $rec data/$d --workers $W --threads 1 --name $name 2>&1 | tr '\r' '\n' | tail -1
  done
  EARLYBIRD_MODEL=$model uv run python scripts/run_stream.py $rec data/librispeech_test_clean_full --limit 20 --threads 1 --tag timing --name $name 2>&1 | tail -1
  [ "$rec" = whistle ] || uv run python -c "
from transformers import MoonshineStreamingForConditionalGeneration as M
m = M.from_pretrained('${model:-UsefulSensors/moonshine-streaming-tiny}')
tot = sum(p.numel() for p in m.parameters())
emb = {id(p): p.numel() for p in (m.get_input_embeddings().weight, m.proj_out.weight)}
v = m.get_input_embeddings().weight.numel()
print(f'params $name: total {tot:,}; vocabulary table counted once {tot - sum(emb.values()) + v:,}; excluding vocabulary tables {tot - sum(emb.values()):,}')" | tail -1 | tee -a runs/params.txt
  aws s3 sync results/raw/$name $B/results/raw/$name --only-show-errors
  log "$name done"
done < runs/plan.txt
step=score
aws s3 sync $B/results/raw results/raw --only-show-errors   # every system already evaluated
uv run python scripts/score.py --md results/eval_$TAG.md > /dev/null
cat runs/params.txt >> results/eval_$TAG.md
aws s3 cp results/eval_$TAG.md $B/results/eval/$TAG.md --only-show-errors
aws s3 cp runs/eval.log $B/results/eval/$TAG.log --only-show-errors
trap - EXIT
log "finished: tables at results/eval/$TAG.md"
