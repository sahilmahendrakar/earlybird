#!/bin/sh
# Guess-quality round (run on an eval machine as eval/TAG/pre.sh, from the repo): every
# anticipating checkpoint guesses with its own ghost words (ghost_demo.py --own), and the
# text-only baseline, Qwen3-0.6B continuing what p29 (no anticipation) has on screen
# (ghost_demo.py --lm), on the same utterances of each set, all in parallel. Logs and
# summaries to s3://…/results/ghost/.
# usage: [ALM=runs/alm135/step8000] sh scripts/ghost_round.sh "NAME=CKPT ..." [noqwen]   (CKPT an S3 key under the bucket;
#        ALM adds the audio-prefixed language model baseline guessing from p29's screen)
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
PATH=$HOME/.local/bin:$PATH
SYSTEMS=${1:-"antic_base=runs/antic_base/step10000 antic_slurp=runs/antic_slurp/step10000"}
SETS="slurp_test slurp_devel librispeech_test_clean ps_test tedlium_test kokoro_test"
mkdir -p results/ghost
aws s3 sync $B/runs/p29_from_tiny/step10000 ckpt/p29 --only-show-errors
for s in $SYSTEMS; do [ "$s" = none ] && continue; aws s3 sync $B/${s#*=} ckpt/${s%%=*} --only-show-errors; done
uv run python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-0.6B')" > /dev/null 2>&1
[ -n "$ALM" ] && aws s3 sync $B/$ALM ckpt/alm --only-show-errors
for d in $SETS; do
  [ -n "$ALM" ] && { n=300; case $d in slurp_*) n=600 ;; esac
    uv run python scripts/ghost_demo.py data/$d ${ALMNAME:-alm135-on-p29}.$d --alm ckpt/alm --model ckpt/p29 --device cpu --limit $n --threads 2 > results/ghost/${ALMNAME:-alm135-on-p29}.$d.txt 2>&1 & }
  n=300; case $d in slurp_*) n=600 ;; esac
  for s in $SYSTEMS; do
    [ "$s" = none ] && continue
    name=${s%%=*}
    P=; case $name in *pad*) P=--pad ;; esac   # a name with "pad" decodes with the padding frames
    uv run python scripts/ghost_demo.py data/$d $name.$d --own --ghost 2 --model ckpt/$name --limit $n --threads 1 $P > results/ghost/$name.$d.txt 2>&1 &
  done
  [ "$2" = noqwen ] || uv run python scripts/ghost_demo.py data/$d qwen-on-p29.$d --lm Qwen/Qwen3-0.6B --device cpu --model ckpt/p29 --limit $n --threads 2 > results/ghost/qwen-on-p29.$d.txt 2>&1 &
done
wait
aws s3 sync results/ghost $B/results/ghost --only-show-errors
sh scripts/runlog.sh "" "ghost round: $(ls results/ghost/*.jsonl | wc -l) logs on S3 results/ghost/"
