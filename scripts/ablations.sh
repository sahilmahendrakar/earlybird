#!/bin/sh
# Paper 1 ablations (Phase 3): what each part of the anticipation recipe does. Each run starts
# from p29 and trains 8000 steps with one setting changed from the main model (antic_gen:
# --lead 0.12 --ghost 2 --ghost-end --kd-agree), so it compares with antic_gen/step8000.
# Runs one after another once the GPU lane is free (runs/alm135.done), each resumable;
# checkpoints and logs to S3, a line in the run log per finished run.
# usage (on the GPU machine, from the repo, as ubuntu): nohup sh scripts/ablations.sh &
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { sh scripts/runlog.sh "" "$*"; }
DATA="--data data/ls_clean_100:1 --data data/ls_clean_360:3 --data data/ls_other_500:3 --data data/ps_clean:3 --data data/kokoro_train:1"
BASE="--init runs/p29_from_tiny/step10000 --teacher UsefulSensors/moonshine-streaming-tiny --steps 8000 --batch 48 --workers 2 \
  --lr 5e-5 --warmup 500 --gpu-frac 0.7 --enc-kd 1 --ghost-weight 0.3 --end-p 0.3 --batch-secs 20 --save-every 1000"
until [ -e runs/alm135.done ]; do sleep 120; done
run() {  # run NAME ARGS...
  name=$1; shift
  [ -e runs/$name.done ] && return
  log "$name: ablation training (re)started: $*"
  until .venv/bin/python scripts/train.py runs/$name $DATA $BASE "$@" >> runs/$name.log 2>&1; do
    log "$name: trainer exited with an error, retrying"; sleep 60
  done
  aws s3 sync runs/$name/step8000 $B/runs/$name/step8000 --only-show-errors && touch runs/$name.done
  log "$name: ablation done; last eval: $(tail -1 runs/$name/log.jsonl)"
}
run abl_lead0   --lead 0.0  --ghost 2 --ghost-end --kd-agree
run abl_lead06  --lead 0.06 --ghost 2 --ghost-end --kd-agree
run abl_lead20  --lead 0.2  --ghost 2 --ghost-end --kd-agree
run abl_ghost0  --lead 0.12 --ghost 0 --kd-agree
run abl_ghost1  --lead 0.12 --ghost 1 --ghost-end --kd-agree
run abl_ghost3  --lead 0.12 --ghost 3 --ghost-end --kd-agree
run abl_nokdagree --lead 0.12 --ghost 2 --ghost-end
log "ablations: all done"
