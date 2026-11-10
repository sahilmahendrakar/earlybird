#!/bin/sh
# Paper 1 (2026-10-17), the padded main model's rigor: two more seeds (the seed-1/2 main models,
# each fine-tuned 5000 steps with --pad like gen2_pad) and its 4-bit size version (QAT from
# gen2_pad, front end at 8 bits, packed). usage: nohup sh scripts/pad2.sh > runs/pad2.out 2>&1 &
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { sh scripts/runlog.sh "" "$*"; }
DATA="--data data/ls_clean_100:1 --data data/ls_clean_360:3 --data data/ls_other_500:3 --data data/ps_clean:3 --data data/kokoro_train:1"
TAIL="--teacher UsefulSensors/moonshine-streaming-tiny --batch 48 --workers 2 --gpu-frac 0.7 --enc-kd 1 --ghost-weight 0.3 \
  --end-p 0.3 --batch-secs 20 --save-every 1000 --ghost 2 --ghost-end --kd-agree --min-heard 0.05 --pad --steps 5000 \
  --lr 2e-5 --warmup 200 --lead 0.3"
run() {  # run NAME ARGS...
  name=$1; shift
  [ -e runs/$name.trained ] && return
  log "$name: training (re)started: $*"
  until .venv/bin/python scripts/train.py runs/$name $DATA $TAIL "$@" >> runs/$name.log 2>&1; do
    log "$name: trainer exited with an error, retrying"; sleep 60
  done
  touch runs/$name.trained
}
for s in 1 2; do
  [ -d runs/antic_gen2_s$s/step20000 ] || aws s3 sync $B/runs/antic_gen2_s$s/step20000 runs/antic_gen2_s$s/step20000 --only-show-errors
  run gen2_pad_s$s --init runs/antic_gen2_s$s/step20000 --seed $s
  [ -e runs/gen2_pad_s$s.done ] || { aws s3 sync runs/gen2_pad_s$s/step5000 $B/runs/gen2_pad_s$s/step5000 --only-show-errors && touch runs/gen2_pad_s$s.done \
    && log "gen2_pad_s$s: done; last eval: $(tail -1 runs/gen2_pad_s$s/log.jsonl)"; }
done
run gen2_pad_q4fe8 --init runs/gen2_pad/step5000 --qat 4
[ -e runs/gen2_pad_q4fe8.done ] || { .venv/bin/python scripts/quantize.py runs/gen2_pad_q4fe8/step5000 runs/gen2_pad_q4fe8_pack --bits 4 --keep 'encoder\.embedder' --keep-bits 8 > runs/gen2_pad_q4fe8_pack.log 2>&1 \
  && aws s3 sync runs/gen2_pad_q4fe8_pack $B/runs/gen2_pad_q4fe8_pack --only-show-errors && touch runs/gen2_pad_q4fe8.done \
  && log "gen2_pad_q4fe8: packed: $(cat runs/gen2_pad_q4fe8_pack/size.txt)" || log "gen2_pad_q4fe8: pack FAILED"; }
log "pad2: all done"
