#!/bin/sh
# Paper 1 rigor (2026-10-16): two more seeds of the main model (antic_gen2's recipe, --seed 1 and 2),
# for error bars. Runs after the size track. usage: nohup sh scripts/seeds.sh &
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { sh scripts/runlog.sh "" "$*"; }
DATA="--data data/ls_clean_100:1 --data data/ls_clean_360:3 --data data/ls_other_500:3 --data data/ps_clean:3 --data data/kokoro_train:1"
COMMON="--init runs/p29_from_tiny/step10000 --teacher UsefulSensors/moonshine-streaming-tiny --steps 20000 --batch 48 --workers 2 \
  --lr 5e-5 --warmup 500 --gpu-frac 0.7 --enc-kd 1 --ghost-weight 0.3 --end-p 0.3 --batch-secs 20 --save-every 1000 \
  --ghost 2 --ghost-end --kd-agree --lead 0.3 --min-heard 0.05"
until grep -q "size: all done" runs/size.out 2>/dev/null || [ -e runs/gen2_q3fe8.done ]; do sleep 120; done
for s in 1 2; do
  name=antic_gen2_s$s
  [ -e runs/$name.done ] && continue
  log "$name: training (re)started (seed $s)"
  until .venv/bin/python scripts/train.py runs/$name $DATA $COMMON --seed $s >> runs/$name.log 2>&1; do
    log "$name: trainer exited with an error, retrying"; sleep 60
  done
  aws s3 sync runs/$name/step20000 $B/runs/$name/step20000 --only-show-errors && touch runs/$name.done
  log "$name: done; last eval: $(tail -1 runs/$name/log.jsonl)"
done
log "seeds: all done"
