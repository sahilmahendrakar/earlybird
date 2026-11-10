#!/bin/sh
# Jobs on the EC2 training machine. Data preparation: real speech with teacher transcripts
# and word times (scripts/prep_real.py). Two lanes share the GPU: LibriSpeech 100, 360, 500
# in turn, and People's Speech clean. Each set is copied to S3 when it finishes and marked in
# runs/NAME.done; started again (after a spot interruption, say), it carries on where it
# stopped. Logs: runs/NAME.log.
# Training: a third lane fine-tunes Moonshine Tiny (runs/tiny_v1) as soon as the first set
# is done, and restarts from its last checkpoint with the larger corpus each time another
# set finishes; then tiny_v2 and the 10M student (see the end of this file). Checkpoints and
# logs are copied to S3 every 10 minutes.
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # three processes share one GPU
# cuDNN's runtime-compiled engines need the wheel's nvrtc; without this, convolutions fail on an
# A10G (g5, needed 2026-10-09 when no g6 had capacity) with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
NV=$PWD/.venv/lib/python3.14/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p runs
log() { sh scripts/runlog.sh "" "$*"; }
log "jobs service started (boot or restart)"
prep() {  # prep NAME args...
  name=$1; shift
  [ -e runs/$name.done ] && return
  log "$name: prep started/resumed"
  n=0
  until uv run python scripts/prep_real.py data/$name "$@" >> runs/$name.log 2>&1; do
    n=$((n + 1)); [ $n -ge 20 ] && { echo "$name: gave up after $n failures" >> runs/jobs.log; log "$name: gave up after $n failures"; return 1; }
    echo "$(date '+%F %T') $name stopped, resuming in 1 min" >> runs/$name.log; sleep 60
  done
  aws s3 sync data/$name $B/data/$name --only-show-errors --exclude 'dl/*' >> runs/$name.log 2>&1 && touch runs/$name.done
  echo "$(date '+%F %T') $name done" >> runs/jobs.log
  log "$name: dataset done, $(wc -l < data/$name/manifest.jsonl) utterances, synced to S3 data/$name"
  aws s3 sync runs $B/runs --only-show-errors --exclude '*' --include '*.log' --include '*.done'
}
( prep ls_clean_100 --source ls --split train_clean_100
  prep ls_clean_360 --source ls --split train_clean_360
  prep ls_other_500 --source ls --split train_other_500 ) &
# People's Speech in one lane: a second lane from the other end of the file list
# (ps_clean_b, --reverse) ran the machine out of memory next to the student on 2026-10-08.
( prep ps_clean --source ps --split clean --hours 300 ) &

done_sets() {
  for s in ls_clean_100:1 ls_clean_360:3 ls_other_500:3 ps_clean:3 ps_clean_b:3; do
    # NAME.usable: train on what is prepared so far (the set stays included when it is done)
    { [ -e runs/${s%:*}.done ] || [ -e runs/${s%:*}.usable ]; } && printf ' --data data/%s' $s
  done
}
push() {  # push RUN
  aws s3 sync runs $B/runs --only-show-errors --exclude '*' --include '*.log' --include '*.done' --include "$1/*"
  ck=$(ls -dv runs/$1/step* 2>/dev/null | tail -1)
  if [ -n "$ck" ] && [ "$ck" != "$(cat runs/$1.logged 2>/dev/null)" ]; then
    echo "$ck" > runs/$1.logged
    log "$1: checkpoint ${ck##*/} on S3; last eval: $(tail -1 runs/$1/log.jsonl)"
  fi
}
train() {  # train RUN ARGS_FN: ARGS_FN prints train.py's extra arguments and is asked again
  # every minute; when its answer or the set of finished datasets changes, the run restarts
  # from its last checkpoint with the new arguments.
  RUN=$1; ARGS_FN=$2
  [ -e runs/$RUN.done ] && return
  [ -e runs/$RUN.paused ] && { log "$RUN: paused (runs/$RUN.paused), not started"; return; }
  while :; do
    sets=$(done_sets); args=$($ARGS_FN)
    [ -z "$sets" ] && { sleep 60; continue; }
    echo "$(date '+%F %T') $RUN: training on$sets $args" >> runs/jobs.log
    log "$RUN: training (re)started on$sets $args, last checkpoint: $(ls -dv runs/$RUN/step* 2>/dev/null | tail -1)"
    .venv/bin/python scripts/train.py runs/$RUN $sets --data data/kokoro_train:1 \
      --save-every 1000 $args >> runs/$RUN.log 2>&1 &
    pid=$!; n=0
    while kill -0 $pid 2>/dev/null && [ "$(done_sets)" = "$sets" ] && [ "$($ARGS_FN)" = "$args" ]; do
      sleep 60; n=$((n + 1)); [ $((n % 10)) = 0 ] && push $RUN
    done
    if kill -0 $pid 2>/dev/null; then   # new data or arguments: restart with them
      kill $pid; wait $pid; continue
    fi
    if wait $pid; then
      touch runs/$RUN.done; echo "$(date '+%F %T') $RUN done" >> runs/jobs.log; push $RUN; log "$RUN: training finished"; return
    fi
    echo "$(date '+%F %T') $RUN stopped, resuming in 1 min" >> runs/$RUN.log; push $RUN; log "$RUN: trainer exited with an error, retrying"; sleep 60
  done
}
# Three runs. tiny_v1: Tiny fine-tuned on whatever is ready (People's Speech arrives last, so
# it mostly misses it). tiny_v2: a short continuation of tiny_v1 once People's Speech is in,
# so the teacher has heard spontaneous speech. student_v1: the 10M student, from when tiny_v1
# is done; its teacher is tiny_v2 once that is done, tiny_v1 until then. The student cannot
# usefully start before a teacher is trained, so it shares this machine rather than a second.
V1=60000; V2=15000
tiny_v1_args() { echo --steps $V1 --gpu-frac 0.55; }
tiny_v2_args() { echo --model runs/tiny_v1/step$V1 --steps $V2 --lr 1.5e-5 --warmup 200 --gpu-frac 0.5; }
student_args() {
  t=runs/tiny_v1/step$V1; [ -e runs/tiny_v2.done ] && t=runs/tiny_v2/step$V2
  echo --student s10 --teacher $t --steps 100000 --lr 1e-3 --warmup 2000 --gpu-frac 0.35
}
# s10_from_tiny: the same 10M student distilled from the released Tiny with ordinary
# (not anticipatory) targets: the first main result (smallest streaming recognizer) and the
# control for what anticipation training adds. It runs first; the Earlybird runs are paused
# (runs/NAME.paused) until it is done. Delete a .paused file and restart the service to resume one.
s10_from_tiny_args() {
  echo --student s10 --teacher UsefulSensors/moonshine-streaming-tiny --plain --steps $S10_STEPS --batch $S10_BATCH \
    --workers 2 --lr 1e-3 --warmup 2000 --gpu-frac 0.65
}
S10_STEPS=${S10_STEPS:-60000}; S10_BATCH=${S10_BATCH:-48}
# v2 adds encoder-output distillation (--enc-kd): s10_from_tiny without it stayed at ~4.05
# nats per token for 11k steps, i.e. the decoder had learned the language but not to use
# the audio. s10_from_tiny is kept paused for the record.
# From step 35000 (2026-10-09): v2 had never heard more than ~18 s (no training utterance is
# longer), so its per-frame decoder positions past that were untrained and it looped on long
# test clips. A quarter of examples are now back-to-back utterances up to 30 s, batches are
# capped by total audio, the untrained position rows are refilled once from Tiny's, and
# People's Speech joins as soon as it is marked usable (runs/ps_clean.usable).
s10_from_tiny_v2_args() { echo $(s10_from_tiny_args) --enc-kd 1 --concat 0.25 --concat-max 30 --batch-secs 20 --extend-pos 850; }
# t15_from_tiny (2026-10-09): s10 plateaued at 3x Tiny's WER and ~50 revisions per word. Start
# instead from Tiny itself cut down (scripts/shrink.py: encoder layers 0,1,4,5, decoder 0,2,5,
# word table 8,192 tokens, width unchanged) and distil from Tiny on streaming cuts (no --plain):
# every example is an utterance cut at a random 80 ms step with the words due by then as the
# target, and Tiny's distribution on the same cut audio, so the student learns what to show
# from a partial utterance and not to change it (the revisions). Words are due near their end
# (--lead 0.04, about where Tiny shows them), no ghost words; 30% of cuts at the very end.
T15_STEPS=${T15_STEPS:-50000}
TRAIN_TEXT="data/ls_clean_100/manifest.jsonl data/ls_clean_360/manifest.jsonl data/ls_other_500/manifest.jsonl data/ps_clean/manifest.jsonl data/kokoro_train/manifest.jsonl"
t15_from_tiny_args() {
  echo --init runs/t15_init --teacher UsefulSensors/moonshine-streaming-tiny --steps $T15_STEPS --batch 48 --workers 2 \
    --lr 2e-4 --warmup 1000 --gpu-frac 0.85 --enc-kd 1 --ghost 0 --lead 0.04 --end-p 0.3 --batch-secs 20
}
( if [ ! -e runs/t15_from_tiny.done ] && [ ! -e runs/t15_init/kept_ids.json ]; then
    log "t15_init: cutting Tiny down (shrink.py)"
    .venv/bin/python scripts/shrink.py UsefulSensors/moonshine-streaming-tiny runs/t15_init $TRAIN_TEXT > runs/t15_init.log 2>&1 \
      && log "t15_init: $(grep -E '^(vocabulary|params|[0-9.]+% of)' runs/t15_init.log | tr '\n' ' ')" \
      || { log "t15_init: shrink.py FAILED, see runs/t15_init.log"; aws s3 cp runs/t15_init.log $B/runs/t15_init.log --only-show-errors; exit 1; }
    aws s3 sync runs/t15_init $B/runs/t15_init --only-show-errors; aws s3 cp runs/t15_init.log $B/runs/t15_init.log --only-show-errors
  fi
  train t15_from_tiny t15_from_tiny_args ) &
# v26_from_tiny (2026-10-12): t15 cut too much (2.4x Tiny's WER). Keep every layer of Tiny and
# cut only the word table to 8,192 tokens (25.7M; shrink.py --enc 0..5 --dec 0..5, made on
# 2026-10-09 as runs/vocab8k: 6.3 test-clean untrained vs Tiny 4.5), and train it the same way
# as t15 (streaming cuts, Tiny as teacher) to win back what the smaller table cost. Words due a
# little earlier than t15's (--lead 0.06; t15 showed words ~30 ms later than Tiny).
V26_STEPS=${V26_STEPS:-15000}
v26_from_tiny_args() {
  echo --init runs/v26_init --teacher UsefulSensors/moonshine-streaming-tiny --steps $V26_STEPS --batch 48 --workers 2 \
    --lr 5e-5 --warmup 500 --gpu-frac 0.85 --enc-kd 1 --ghost 0 --lead 0.06 --end-p 0.3 --batch-secs 20
}
( [ -e runs/v26_init/kept_ids.json ] || aws s3 sync $B/runs/vocab8k runs/v26_init --only-show-errors
  train v26_from_tiny v26_from_tiny_args ) &
# p29 (2026-10-13): the 8k table cost 1-4 WER for ~2 MB, so keep the lossless 19k table (Tiny
# pruned to the tokens our text uses, runs/tiny_pruned/tied_in, 29.2M, = Tiny's WER untrained)
# and give it the same streaming-cut training as v26 (which halved revisions), then a short
# quantization-aware run at 4 bits (front end left out, packed at 8 bits): ~16 MB on disk.
p29_args() {
  echo --init runs/p29_init --teacher UsefulSensors/moonshine-streaming-tiny --steps 10000 --batch 48 --workers 2 \
    --lr 5e-5 --warmup 500 --gpu-frac 0.85 --enc-kd 1 --ghost 0 --lead 0.06 --end-p 0.3 --batch-secs 20
}
p29q4_args() {
  echo --init runs/p29_from_tiny/step10000 --teacher UsefulSensors/moonshine-streaming-tiny --steps 5000 --batch 48 --workers 2 \
    --lr 2e-5 --warmup 200 --gpu-frac 0.85 --enc-kd 1 --ghost 0 --lead 0.06 --end-p 0.3 --batch-secs 20 --qat 4
}
pack() {  # pack NAME CKPT ARGS...: quantize.py into runs/NAME, to S3, sizes in the run log
  name=$1; ck=$2; shift 2
  [ -e runs/$name/size.txt ] && return
  .venv/bin/python scripts/quantize.py $ck runs/$name "$@" > runs/$name.log 2>&1 \
    && aws s3 sync runs/$name $B/runs/$name --only-show-errors && log "$name: $(cat runs/$name/size.txt)" \
    || log "$name: quantize.py FAILED (runs/$name.log)"
}
( [ -e runs/p29_init/kept_ids.json ] || aws s3 sync $B/runs/tiny_pruned/tied_in runs/p29_init --only-show-errors
  train p29_from_tiny p29_args
  pack p29_rtn4 runs/p29_from_tiny/step10000 --bits 4 --keep 'encoder\.embedder' --keep-bits 8
  train p29q4 p29q4_args
  pack p29q4_pack runs/p29q4/step5000 --bits 4 --keep 'encoder\.embedder' --keep-bits 8 ) &
# Anticipation (2026-10-15): the p29 model trained to show words 120 ms before they end
# (--lead 0.12) and to guess the next 2 words after end-of-text, each with a probability, for
# voice agents to act on early. Tiny cannot anticipate, so it is distilled only on heard text
# where it agrees with the target (--kd-agree). antic_base: our usual data; antic_slurp: plus
# SLURP spoken commands (what a voice agent hears). SLURP sets: train (teacher-filtered),
# test and devel (every utterance the aligner can place, one recording per sentence).
( prep slurp_test --source slurp --split test --keep-all --one-per-sentence --max-utts 1500 --min-score 0.1
  prep slurp_devel --source slurp --split devel --keep-all --one-per-sentence --max-utts 800 --min-score 0.1
  prep slurp_train --source slurp --split train --min-score 0.1 ) &
antic_common() {
  echo --init runs/p29_from_tiny/step10000 --teacher UsefulSensors/moonshine-streaming-tiny --steps 10000 --batch 48 \
    --workers 2 --lr 5e-5 --warmup 500 --gpu-frac 0.7 --enc-kd 1 --lead 0.12 --ghost 2 --ghost-weight 0.3 --end-p 0.3 \
    --batch-secs 20 --kd-agree
}
antic_base_args() { antic_common; }
# antic_slurp restarted 2026-10-15 02:20Z with --ghost-end: antic_base was never taught to end
# its guess ("that is the whole command"), so it almost never claimed a command before it ended
antic_slurp_args() { echo $(antic_common) --data data/slurp_train:2 --ghost-end; }
( train antic_base antic_base_args
  until [ -e runs/slurp_train.done ]; do sleep 120; done
  train antic_slurp antic_slurp_args ) &
# Round 2 (2026-10-15): more spoken commands. 30k command sentences (SLURP training
# templates with their slots refilled, make_commands.py; no test or devel sentence) spoken by
# Kokoro's 26 training voices at 4 rates, then given the teacher's written form (recase.py).
# antic_cmd continues antic_slurp on SLURP + these.
KVOICES=af_alloy,af_aoede,af_bella,af_heart,af_jessica,af_kore,af_nicole,af_nova,af_river,af_sarah,af_sky,am_adam,am_echo,am_eric,am_fenrir,am_liam,am_onyx,am_puck,am_santa,bf_alice,bf_isabella,bf_lily,bm_daniel,bm_fable,bm_george,bm_lewis
synth_cmd() {   # two shards in parallel (synthesis is ~0.5 s per sentence, CPU-bound phonemization)
  [ -e runs/kokoro_cmd.done ] && return
  log "kokoro_cmd: synthesis started/resumed"
  aws s3 cp $B/data/text/commands.txt data/text/commands.txt --only-show-errors
  for sh in a:0 b:15000; do
    ( uv run python scripts/synth_kokoro.py data/kokoro_cmd_${sh%%:*} --split cmd --text-file data/text/commands.txt --n 10000 \
        --start ${sh#*:} --flac --speeds 0.9,1.0,1.1,1.2 --voices $KVOICES --device cuda >> runs/kokoro_cmd_${sh%%:*}.log 2>&1 \
      && uv run python scripts/recase.py data/kokoro_cmd_${sh%%:*} >> runs/kokoro_cmd_${sh%%:*}.log 2>&1 ) &
  done
  wait
  for x in a b; do aws s3 sync data/kokoro_cmd_$x $B/data/kokoro_cmd_$x --only-show-errors; done
  touch runs/kokoro_cmd.done
  log "kokoro_cmd: done, $(cat data/kokoro_cmd_a/manifest.jsonl data/kokoro_cmd_b/manifest.jsonl | wc -l) utterances"
}
antic_cmd_args() {
  echo --init runs/antic_slurp/step10000 --teacher UsefulSensors/moonshine-streaming-tiny --steps 10000 --batch 48 \
    --workers 2 --lr 5e-5 --warmup 500 --gpu-frac 0.7 --enc-kd 1 --lead 0.12 --ghost 2 --ghost-weight 0.3 --end-p 0.3 \
    --batch-secs 20 --kd-agree --ghost-end --data data/slurp_train:2 --data data/kokoro_cmd_a:1 --data data/kokoro_cmd_b:1
}
( until [ -e runs/antic_slurp.done ]; do sleep 120; done
  synth_cmd && train antic_cmd antic_cmd_args ) &
# Paper 1 (2026-10-15): the GENERAL anticipating model (no command data), with --ghost-end
# (antic_base lacked it), trained longer. Main result of the general write-up.
antic_gen_args() { antic_common | sed 's/--steps 10000/--steps 15000/'; echo --ghost-end; }
# then the audio-prefixed language model baseline (after Yusuf et al. 2024; audio_lm.py)
alm_train() {
  [ -e runs/alm135.done ] && return
  log "alm135: training started"
  .venv/bin/python scripts/audio_lm.py train runs/alm135 --data data/ls_clean_360 --data data/ls_other_500 \
    --data data/ps_clean --data data/kokoro_train --steps 8000 --batch 32 >> runs/alm135.log 2>&1 \
    && aws s3 sync runs/alm135/step8000 $B/runs/alm135/step8000 --only-show-errors && touch runs/alm135.done \
    && log "alm135: done ($(grep '^step' runs/alm135.log | tail -1))" || log "alm135: FAILED (runs/alm135.log)"
}
( until [ -e runs/kokoro_cmd.done ]; do sleep 60; done
  train antic_gen antic_gen_args
  alm_train ) &
train s10_from_tiny s10_from_tiny_args &
train s10_from_tiny_v2 s10_from_tiny_v2_args &
train tiny_v1 tiny_v1_args &
( until [ -e runs/tiny_v1.done ] && [ -e runs/ps_clean.done ] && [ -e runs/ps_clean_b.done ]; do sleep 300; done; train tiny_v2 tiny_v2_args ) &
( until [ -e runs/tiny_v1.done ]; do sleep 300; done; train student_v1 student_args ) &
wait
