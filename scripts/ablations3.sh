#!/bin/sh
# Paper 1, round 3 (2026-10-16): the lead sweep kept paying (300 ms target, 50 ms heard floor:
# words right-and-kept 40 ms before their end on test-clean vs Moonshine Tiny 20, Small 10; WER
# unchanged). So: the next main model on that setting, trained longer (antic_gen2, 20000 steps from
# p29), then 400 and 500 ms to find where it stops paying. Runs after alm2.
# usage: nohup sh scripts/ablations3.sh &
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { sh scripts/runlog.sh "" "$*"; }
DATA="--data data/ls_clean_100:1 --data data/ls_clean_360:3 --data data/ls_other_500:3 --data data/ps_clean:3 --data data/kokoro_train:1"
COMMON="--init runs/p29_from_tiny/step10000 --teacher UsefulSensors/moonshine-streaming-tiny --batch 48 --workers 2 \
  --lr 5e-5 --warmup 500 --gpu-frac 0.7 --enc-kd 1 --ghost-weight 0.3 --end-p 0.3 --batch-secs 20 --save-every 1000 --ghost 2 --ghost-end --kd-agree"
until [ -e runs/alm2.done ] || grep -q "alm2: FAILED" runs/alm2.out 2>/dev/null; do sleep 120; done
run() {  # run NAME STEPS ARGS...
  name=$1; steps=$2; shift 2
  [ -e runs/$name.done ] && return
  log "$name: training (re)started: --steps $steps $*"
  until .venv/bin/python scripts/train.py runs/$name $DATA $COMMON --steps $steps "$@" >> runs/$name.log 2>&1; do
    log "$name: trainer exited with an error, retrying"; sleep 60
  done
  aws s3 sync runs/$name/step$steps $B/runs/$name/step$steps --only-show-errors && touch runs/$name.done
  log "$name: done; last eval: $(tail -1 runs/$name/log.jsonl)"
}
run antic_gen2 20000 --lead 0.3 --min-heard 0.05
run abl_lead40_mh05 8000 --lead 0.4 --min-heard 0.05
run abl_lead50_mh05 8000 --lead 0.5 --min-heard 0.05
log "ablations3: all done"
