#!/bin/sh
# Paper 1, more anticipation (2026-10-16): after the first ablations, try to write clearly earlier
# than Moonshine. Same base as ablations.sh (8000 steps from p29, compare with antic_gen/step8000).
#   abl_lead30_mh05  words due 300 ms before they end, after at least 50 ms of them is heard
#   abl_lead30_mh0   the same with no floor (a word can be due as soon as it starts)
#   abl_gw1          guess words weighted like heard ones (ghost weight 1.0 instead of 0.3)
# usage (on the GPU machine, from the repo, as ubuntu): nohup sh scripts/ablations2.sh &
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { sh scripts/runlog.sh "" "$*"; }
DATA="--data data/ls_clean_100:1 --data data/ls_clean_360:3 --data data/ls_other_500:3 --data data/ps_clean:3 --data data/kokoro_train:1"
BASE="--init runs/p29_from_tiny/step10000 --teacher UsefulSensors/moonshine-streaming-tiny --steps 8000 --batch 48 --workers 2 \
  --lr 5e-5 --warmup 500 --gpu-frac 0.7 --enc-kd 1 --ghost-weight 0.3 --end-p 0.3 --batch-secs 20 --save-every 1000"
until [ -e runs/abl_nokdagree.done ]; do sleep 120; done
run() {
  name=$1; shift
  [ -e runs/$name.done ] && return
  log "$name: ablation training (re)started: $*"
  until .venv/bin/python scripts/train.py runs/$name $DATA $BASE "$@" >> runs/$name.log 2>&1; do
    log "$name: trainer exited with an error, retrying"; sleep 60
  done
  aws s3 sync runs/$name/step8000 $B/runs/$name/step8000 --only-show-errors && touch runs/$name.done
  log "$name: ablation done; last eval: $(tail -1 runs/$name/log.jsonl)"
}
run abl_lead30_mh05 --lead 0.3 --min-heard 0.05 --ghost 2 --ghost-end --kd-agree
run abl_lead30_mh0  --lead 0.3 --min-heard 0.0 --ghost 2 --ghost-end --kd-agree
run abl_gw1         --lead 0.12 --ghost 2 --ghost-end --kd-agree --ghost-weight 1.0
log "ablations2: all done"
