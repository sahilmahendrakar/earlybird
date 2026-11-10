#!/bin/sh
# Run every baseline over both test sets, then a quiet single-process pass for compute numbers.
# Moonshine runs on the full sets. Nemotron and Whisper are several times slower to evaluate,
# so they run on an evenly spaced subset of 150 utterances per set.
# Each Nemotron worker holds 1.4 GB, so the default is 2 workers, and every run goes through
# guard.sh, which stops it if the machine runs short of memory.
# usage: scripts/run_baselines.sh [workers] [stage ...]   stages: moonshine big timing
W=${1:-2}; shift
STAGES=${*:-moonshine big timing}
SETS="data/kokoro_test data/librispeech_test_clean"
MOON="moonshine-tiny moonshine-small moonshine-tiny-default moonshine-small-default"
BIG="nemotron-80ms nemotron-160ms nemotron-560ms whisper-base.en"
run() { scripts/guard.sh uv run python scripts/run_stream.py "$@" 2>&1 | tr '\r' '\n' | tail -1; }
for s in $STAGES; do case $s in
  moonshine) for m in $MOON; do for d in $SETS; do run $m $d --workers $W --threads 1; done; done ;;
  big) for m in $BIG; do for d in $SETS; do run $m $d --workers $W --threads 4 --limit 150; done; done ;;
  # compute: 20 utterances per set, one process at a time (Moonshine 1 thread, the others 2)
  timing) for m in $MOON $BIG; do for d in $SETS; do run $m $d --limit 20 --threads 2 --tag timing; done; done ;;
esac; done
uv run python scripts/score.py --md results/tables.md > /dev/null
