#!/bin/sh
# The audio-prefixed baseline, stronger (2026-10-16): alm135 barely used its audio (probe: next-word
# loss 4.77 with the audio vs 5.01 with silence), so: 70% of training cuts inside the word being
# spoken, a 2 s audio prompt (100 frames), adapter at 30x the LM's learning rate, 20000 steps.
# Runs after the ablations (runs/abl_gw1.done). usage: nohup sh scripts/alm2.sh &
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib
log() { sh scripts/runlog.sh "" "$*"; }
until [ -e runs/abl_gw1.done ]; do sleep 120; done
[ -e runs/alm2.done ] && exit 0
log "alm2: training started"
.venv/bin/python scripts/audio_lm.py train runs/alm2 --data data/ls_clean_360 --data data/ls_other_500 --data data/ps_clean \
  --data data/kokoro_train --steps 20000 --batch 32 --frames 100 --inside 0.7 --adapter-lr-mult 30 >> runs/alm2.log 2>&1 \
  && .venv/bin/python scripts/audio_lm.py probe runs/alm2/step20000 --data data/ls_clean_100 --batch 300 >> runs/alm2.log 2>&1 \
  && aws s3 sync runs/alm2/step20000 $B/runs/alm2/step20000 --only-show-errors && touch runs/alm2.done \
  && log "alm2: done; $(grep '^step' runs/alm2.log | tail -1); probe: $(tail -1 runs/alm2.log)" || log "alm2: FAILED (runs/alm2.log)"
