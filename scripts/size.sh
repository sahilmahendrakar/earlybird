#!/bin/sh
# Paper 1 size track (2026-10-16): the main model (antic_gen2) trained with its weights rounded
# (--qat), then packed by quantize.py. Three variants, 5000 steps each from antic_gen2/step20000:
#   gen2_q4fe8  4-bit, audio front end left out of QAT and packed at 8 bits   (p29q4's recipe)
#   gen2_q4     4-bit everywhere, front end included in QAT
#   gen2_q3fe8  3-bit, front end at 8 bits
# Runs after ablations3. usage: nohup sh scripts/size.sh &
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { sh scripts/runlog.sh "" "$*"; }
DATA="--data data/ls_clean_100:1 --data data/ls_clean_360:3 --data data/ls_other_500:3 --data data/ps_clean:3 --data data/kokoro_train:1"
COMMON="--init runs/antic_gen2/step20000 --teacher UsefulSensors/moonshine-streaming-tiny --steps 5000 --batch 48 --workers 2 \
  --lr 2e-5 --warmup 200 --gpu-frac 0.7 --enc-kd 1 --ghost-weight 0.3 --end-p 0.3 --batch-secs 20 --save-every 1000 \
  --ghost 2 --ghost-end --kd-agree --lead 0.3 --min-heard 0.05"
until grep -q "ablations3: all done" runs/ablations3.out 2>/dev/null || [ -e runs/abl_lead50_mh05.done ]; do sleep 120; done
run() {  # run NAME QAT_ARGS -- PACK_ARGS
  name=$1; shift; qat=""; while [ "$1" != "--" ]; do qat="$qat $1"; shift; done; shift
  [ -e runs/$name.done ] && return
  log "$name: QAT started:$qat"
  until .venv/bin/python scripts/train.py runs/$name $DATA $COMMON $qat >> runs/$name.log 2>&1; do
    log "$name: trainer exited with an error, retrying"; sleep 60
  done
  .venv/bin/python scripts/quantize.py runs/$name/step5000 runs/${name}_pack "$@" > runs/${name}_pack.log 2>&1 \
    && aws s3 sync runs/${name}_pack $B/runs/${name}_pack --only-show-errors && touch runs/$name.done \
    && log "$name: packed: $(cat runs/${name}_pack/size.txt)" || log "$name: pack FAILED"
}
run gen2_q4fe8 --qat 4 -- --bits 4 --keep 'encoder\.embedder' --keep-bits 8
run gen2_q4    --qat 4 --qat-keep '^$' -- --bits 4
run gen2_q3fe8 --qat 3 -- --bits 3 --keep 'encoder\.embedder' --keep-bits 8
log "size: all done"
