#!/bin/sh
# Synthesize the training corpus with Kokoro: 26 voices (am_michael and bf_emma are held out
# for the test set), six speaking rates, FLAC, per-word timings in the manifest.
# Runs at low priority under guard.sh; if the guard stops it for lack of memory it waits
# five minutes and resumes (synth_kokoro.py skips what is already done).
# With a second argument only the sentences listed in that file are synthesized.
# usage: caffeinate -i scripts/synth_train.sh [n_sentences] [only_file]      log: data/kokoro_train/synth.log
cd "$(dirname "$0")/.."
N=${1:-65000}
ONLY=${2:+--only-file $2}
VOICES=af_alloy,af_aoede,af_bella,af_heart,af_jessica,af_kore,af_nicole,af_nova,af_river,af_sarah,af_sky,am_adam,am_echo,am_eric,am_fenrir,am_liam,am_onyx,am_puck,am_santa,bf_alice,bf_isabella,bf_lily,bm_daniel,bm_fable,bm_george,bm_lewis
mkdir -p data/kokoro_train
exec >> data/kokoro_train/synth.log 2>&1
export OMP_NUM_THREADS=3
while :; do
  [ "$(df -g . | awk 'NR==2 {print $4}')" -lt 20 ] && { echo "under 20 GB of disk free, stopping"; exit 1; }
  nice -n 10 scripts/guard.sh -f 10 -s 4000 uv run python scripts/synth_kokoro.py data/kokoro_train \
    --split train --text-file data/text/train.txt --n $N --flac \
    --speeds 0.85,0.95,1.0,1.05,1.15,1.25 --voices $VOICES $ONLY
  rc=$?
  pkill -f "synth_kokoro.py data/kokoro_train"  # nothing left behind before a restart
  [ $rc -eq 0 ] && break
  echo "$(date '+%F %T') stopped with code $rc, resuming in 5 min"; sleep 300
done
