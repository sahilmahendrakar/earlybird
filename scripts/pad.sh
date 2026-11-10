#!/bin/sh
# Paper 1 (2026-10-17): training with the padding frames. Untrained, the three padding frames
# (Engine(pad=True)) make Moonshine Tiny as early as Earlybird, at 3-4x the rewrites; Earlybird
# was trained without them. These runs train with them (train.py --pad):
#   gen2_pad      5000 steps from antic_gen2/step20000 (compare antic_gen2 decoded with pad)
#   abl_pad30     8000 steps from p29, lead 0.3 floor 0.05 (compare abl_lead30_mh05)
#   abl_pad20     same, lead 0.2 (with the padding frames a smaller lead may be enough)
# First checks that train.encode with PAD matches the engine frame for frame.
# usage: nohup sh scripts/pad.sh > runs/pad.out 2>&1 &
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { sh scripts/runlog.sh "" "$*"; }
DATA="--data data/ls_clean_100:1 --data data/ls_clean_360:3 --data data/ls_other_500:3 --data data/ps_clean:3 --data data/kokoro_train:1"
TAIL="--teacher UsefulSensors/moonshine-streaming-tiny --batch 48 --workers 2 --gpu-frac 0.7 --enc-kd 1 --ghost-weight 0.3 \
  --end-p 0.3 --batch-secs 20 --save-every 1000 --ghost 2 --ghost-end --kd-agree --min-heard 0.05 --pad"
if ! [ -e runs/pad_check.ok ]; then
  .venv/bin/python scripts/check_pad.py runs/antic_gen2/step20000 data/ls_clean_100 > runs/pad_check.log 2>&1
  if grep -q " OK" runs/pad_check.log; then touch runs/pad_check.ok; log "pad: train.encode matches Engine(pad=True): $(tail -1 runs/pad_check.log)"
  else log "pad: check FAILED: $(tail -2 runs/pad_check.log | tr '\n' ' ')"; exit 1; fi
fi
run() {  # run NAME STEPS ARGS...
  name=$1; steps=$2; shift 2
  [ -e runs/$name.done ] && return
  log "$name: training (re)started: $*"
  until .venv/bin/python scripts/train.py runs/$name $DATA $TAIL --steps $steps "$@" >> runs/$name.log 2>&1; do
    log "$name: trainer exited with an error, retrying"; sleep 60
  done
  aws s3 sync runs/$name/step$steps $B/runs/$name/step$steps --only-show-errors && touch runs/$name.done
  log "$name: done; last eval: $(tail -1 runs/$name/log.jsonl)"
}
run gen2_pad 5000 --init runs/antic_gen2/step20000 --lr 2e-5 --warmup 200 --lead 0.3
run abl_pad30 8000 --init runs/p29_from_tiny/step10000 --lr 5e-5 --warmup 500 --lead 0.3
run abl_pad20 8000 --init runs/p29_from_tiny/step10000 --lr 5e-5 --warmup 500 --lead 0.2
log "pad: all done"
