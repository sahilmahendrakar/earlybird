#!/bin/sh
# Open ASR Leaderboard round on a throwaway eval machine: run as eval/TAG/pre.sh of a
# scripts/launch_eval.sh round (empty plan). Every set (smallest first), every system given
# as NAME:SYSTEM[:CHECKPOINT[:WORKERS]]; results are synced to s3://…/results/open_asr/ every few
# minutes and picked up again by a later round (finished utterances are skipped). At the end it
# scores everything on S3 (results/open_asr.md) and shuts the machine down (= terminate).
# usage: sh scripts/open_asr_box.sh TAG NAME:SYSTEM[:CKPT[:W]] ...
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
PATH=$HOME/.local/bin:$PATH
TAG=$1; shift
log() { sh scripts/runlog.sh "" "oasr $TAG: $*"; }
up() { aws s3 sync results/open_asr $B/results/open_asr --only-show-errors; }
aws s3 sync $B/results/open_asr results/open_asr --only-show-errors
aws s3 sync $B/data/tedlium_test data/tedlium_test --only-show-errors --exclude 'dl/*'
git clone -q --depth 1 https://github.com/huggingface/open_asr_leaderboard /tmp/oal
case "$*" in *whistle*)   # Cactus took the engine wheels off the Hub on 2026-11-05: use the last build (Oct 19, the one
  # our earlier Whistle rounds ran), copied from Cactus-Compute/needle3@d6c7d6a2e9 to the bucket
  uv pip install -q cactus-needle==3.1.4
  aws s3 cp $B/tmp/cactus_needle-3.2.0-manylinux2014_x86_64.whl /tmp/needle.whl --only-show-errors
  python3 -m zipfile -e /tmp/needle.whl /tmp/needle && cp /tmp/needle/needle/libneedle3.so .venv/lib/python3.*/site-packages/needle/
  uv run --no-sync python -c "from needle.agent.whistle import Whistle; Whistle()" >/dev/null 2>&1 ;; esac
for spec in "$@"; do   # checkpoints
  c=$(echo "$spec" | cut -d: -f3); [ -n "$c" ] && aws s3 sync $B/$c ckpt/$c --only-show-errors
done
case "$*" in *gen2_pad_q4_trim*)
  aws s3 cp $B/results/raw/gen2_pad_q4trim-pad/tedlium_test.jsonl raw_ted.jsonl --only-show-errors
  log "check: $(uv run python scripts/open_asr.py check tedlium --model ckpt/runs/gen2_pad_q4_trim --raw raw_ted.jsonl --limit 100 2>&1 | tail -1)" ;;
esac
(while sleep 300; do up; done) & syncer=$!
t0=$(date +%s)
for set in tedlium voxpopuli librispeech_clean earnings22 librispeech_other ami gigaspeech spgispeech; do
  for spec in "$@"; do
    name=$(echo "$spec" | cut -d: -f1); sys=$(echo "$spec" | cut -d: -f2)
    c=$(echo "$spec" | cut -d: -f3); w=$(echo "$spec" | cut -d: -f4)
    uv run --no-sync python scripts/open_asr.py run $sys $set --name $name ${c:+--model ckpt/$c} ${w:+--workers $w} 2>&1 | tail -1 >> runs/oasr.log
  done
  up; log "$set done ($(( ($(date +%s) - t0) / 60 )) min): $(tail -n $# runs/oasr.log | tr '\n' ';')"
done
kill $syncer; up
aws s3 sync $B/results/open_asr results/open_asr --only-show-errors
OAL_DIR=/tmp/oal uv run python scripts/open_asr.py score --md results/open_asr.md > /dev/null 2>runs/score.err || tail -3 runs/score.err >> runs/oasr.log
aws s3 cp results/open_asr.md $B/results/open_asr_$TAG.md --only-show-errors
aws s3 cp runs/oasr.log $B/results/eval/oasr_$TAG.log --only-show-errors
log "finished in $(( ($(date +%s) - t0) / 60 )) min: results/open_asr_$TAG.md"
sudo shutdown -h now
