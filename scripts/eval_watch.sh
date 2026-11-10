#!/bin/sh
# Cron job on the dev box: start an evaluation round (launch_eval.sh) for the released Tiny
# reference once, then for s10_from_tiny at each milestone checkpoint as it reaches S3.
# Rounds already started are listed in ~/.earlybird-evals. Log: ~/.earlybird-eval-watch.log
cd "$(dirname "$0")/.."
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
STATE=$HOME/.earlybird-evals; touch $STATE
exec >> $HOME/.earlybird-eval-watch.log 2>&1
round() {  # round TAG plan-lines...   (env: PRE=script uploaded as eval/TAG/pre.sh, SETS_EXTRA, WORKERS)
  grep -qx "$1" $STATE && return
  [ -n "$PRE" ] && aws s3 cp "$PRE" $B/eval/$1/pre.sh --only-show-errors
  [ -n "$SETS_EXTRA" ] && echo "$SETS_EXTRA" | aws s3 cp - $B/eval/$1/extra_sets --only-show-errors
  [ -n "$WORKERS" ] && echo "$WORKERS" | aws s3 cp - $B/eval/$1/workers --only-show-errors
  [ -n "$ONLY_SETS" ] && echo "$ONLY_SETS" | aws s3 cp - $B/eval/$1/only_sets --only-show-errors
  echo "$(date -u '+%F %H:%MZ') launching $1"
  sh scripts/launch_eval.sh "$@" && echo "$1" >> $STATE
}
round ref-tiny 'ref-tiny-nohold tiny-nohold UsefulSensors/moonshine-streaming-tiny' \
  'ref-tiny-nohold-pad tiny-nohold-pad UsefulSensors/moonshine-streaming-tiny'
for s in 10000 20000; do
  aws s3 ls $B/runs/s10_from_tiny_v2/step$s/config.json > /dev/null 2>&1 || continue
  round s10v2-step$s "s10_from_tiny_v2-step$s earlybird runs/s10_from_tiny_v2/step$s" \
    "s10_from_tiny_v2-step$s-pad earlybird-pad runs/s10_from_tiny_v2/step$s"
done
# After the long-clip fix at step 38000: 50000 (12k steps in) and 60000, one system each
# (the -pad variant scored within 0.5 WER of plain at 10k and 20k; half the cost).
for s in 50000 60000; do
  aws s3 ls $B/runs/s10_from_tiny_v2/step$s/config.json > /dev/null 2>&1 || continue
  round s10v2-step$s "s10_from_tiny_v2-step$s earlybird runs/s10_from_tiny_v2/step$s"
done
# t15_from_tiny (Tiny cut down to 17M, streaming-cut distillation): halfway and at the end.
for s in 25000 50000; do
  aws s3 ls $B/runs/t15_from_tiny/step$s/config.json > /dev/null 2>&1 || continue
  round t15-step$s "t15_from_tiny-step$s earlybird runs/t15_from_tiny/step$s"
done
# v26_from_tiny (Tiny, all layers, 8k word table, streaming-cut training): halfway and at the end.
for s in 7000 15000; do
  aws s3 ls $B/runs/v26_from_tiny/step$s/config.json > /dev/null 2>&1 || continue
  EVAL_MARKETS=on-demand round v26-step$s "v26_from_tiny-step$s earlybird runs/v26_from_tiny/step$s"
done
# p29 (19k table, streaming-cut training), then 4-bit: plain rounding vs quantization-aware.
aws s3 ls $B/runs/p29_from_tiny/step10000/config.json > /dev/null 2>&1 && \
  EVAL_MARKETS=on-demand round p29-step10000 "p29_from_tiny-step10000 earlybird runs/p29_from_tiny/step10000"
aws s3 ls $B/runs/p29q4_pack/size.txt > /dev/null 2>&1 && aws s3 ls $B/runs/p29_rtn4/size.txt > /dev/null 2>&1 && \
  EVAL_MARKETS=on-demand round p29-4bit "p29q4 earlybird runs/p29q4_pack" "p29_rtn4 earlybird runs/p29_rtn4"
# Anticipation: references on SLURP as soon as its test set exists; the two anticipating
# checkpoints on every set + SLURP; then the guess-quality round (ghost_round.sh).
if aws s3 ls $B/data/slurp_test/manifest.jsonl > /dev/null 2>&1 && aws s3 ls $B/runs/slurp_test.done > /dev/null 2>&1; then
  ( ONLY_SETS=slurp_test; export EVAL_MARKETS=on-demand; round slurp-refs "ref-tiny-nohold tiny-nohold UsefulSensors/moonshine-streaming-tiny" \
    "p29_from_tiny-step10000 earlybird runs/p29_from_tiny/step10000" "p29q4 earlybird runs/p29q4_pack" )
fi
if aws s3 ls $B/runs/antic_base/step10000/config.json > /dev/null 2>&1 && aws s3 ls $B/runs/slurp_test.done > /dev/null 2>&1; then
  ( SETS_EXTRA=slurp_test; export EVAL_MARKETS=on-demand; round antic-base "antic_base-step10000 earlybird runs/antic_base/step10000" )
  printf "sh scripts/ghost_round.sh 'antic_base=runs/antic_base/step10000'\n" > /tmp/ghost_base_pre.sh
  ( PRE=/tmp/ghost_base_pre.sh; export EVAL_MARKETS=on-demand; round ghost-base "" )
fi
if aws s3 ls $B/runs/antic_slurp/step10000/config.json > /dev/null 2>&1; then
  ( SETS_EXTRA=slurp_test; export EVAL_MARKETS=on-demand; round antic-slurp "antic_slurp-step10000 earlybird runs/antic_slurp/step10000" )
  printf "sh scripts/ghost_round.sh 'antic_slurp=runs/antic_slurp/step10000' noqwen\n" > /tmp/ghost_pre.sh
  ( PRE=/tmp/ghost_pre.sh; export EVAL_MARKETS=on-demand; round ghost-1 "" )
fi
# Round 2: antic_cmd (more spoken commands) on every set + SLURP, then its guesses.
if aws s3 ls $B/runs/antic_cmd/step10000/config.json > /dev/null 2>&1; then
  ( SETS_EXTRA=slurp_test; export EVAL_MARKETS=on-demand; round antic-cmd "antic_cmd-step10000 earlybird runs/antic_cmd/step10000" )
  printf "sh scripts/ghost_round.sh 'antic_cmd=runs/antic_cmd/step10000' noqwen\n" > /tmp/ghost_cmd_pre.sh
  ( PRE=/tmp/ghost_cmd_pre.sh; export EVAL_MARKETS=on-demand; round ghost-cmd "" )
fi
# Paper 1: the general anticipating model.
if aws s3 ls $B/runs/antic_gen/step15000/config.json > /dev/null 2>&1; then
  ( SETS_EXTRA=slurp_test; export EVAL_MARKETS=on-demand; round antic-gen "antic_gen-step15000 earlybird runs/antic_gen/step15000" )
  printf "sh scripts/ghost_round.sh 'antic_gen=runs/antic_gen/step15000' noqwen\n" > /tmp/ghost_gen_pre.sh
  ( PRE=/tmp/ghost_gen_pre.sh; export EVAL_MARKETS=on-demand; round ghost-gen "" )
fi
# Paper 1 baselines: text models of three sizes guessing from p29's screen and from a perfect transcript.
printf "sh scripts/lm_round.sh\n" > /tmp/lm_pre.sh
( PRE=/tmp/lm_pre.sh; export EVAL_MARKETS=on-demand; round lm-1 "" )
# Paper 1 baseline: the audio-prefixed language model (Yusuf et al. 2024 design) on p29's screen.
if aws s3 ls $B/runs/alm135/step8000/adapter.pt > /dev/null 2>&1; then
  printf "ALM=runs/alm135/step8000 sh scripts/ghost_round.sh none noqwen\n" > /tmp/ghost_alm_pre.sh
  ( PRE=/tmp/ghost_alm_pre.sh; export EVAL_MARKETS=on-demand; round ghost-alm "" )
fi
# Paper 1 ablations: each on every set when done; all their guesses in one round at the end.
for a in abl_lead0 abl_lead06 abl_lead20 abl_ghost0 abl_ghost1 abl_ghost3 abl_nokdagree abl_lead30_mh05 abl_lead30_mh0 abl_gw1; do
  aws s3 ls $B/runs/$a/step8000/config.json > /dev/null 2>&1 || continue
  ( export EVAL_MARKETS=on-demand; round $a "$a-step8000 earlybird runs/$a/step8000" )
done
if aws s3 ls $B/runs/abl_nokdagree/step8000/config.json > /dev/null 2>&1; then
  S="antic_gen8k=runs/antic_gen/step8000"
  for a in abl_lead0 abl_lead06 abl_lead20 abl_ghost1 abl_ghost3 abl_nokdagree; do S="$S $a=runs/$a/step8000"; done
  printf "sh scripts/ghost_round.sh '%s' noqwen\n" "$S" > /tmp/ghost_abl_pre.sh
  ( PRE=/tmp/ghost_abl_pre.sh; export EVAL_MARKETS=on-demand; round ghost-abl "" )
fi
if aws s3 ls $B/runs/abl_gw1/step8000/config.json > /dev/null 2>&1; then
  printf "sh scripts/ghost_round.sh 'abl_lead30_mh05=runs/abl_lead30_mh05/step8000 abl_lead30_mh0=runs/abl_lead30_mh0/step8000 abl_gw1=runs/abl_gw1/step8000' noqwen\n" > /tmp/ghost_abl2_pre.sh
  ( PRE=/tmp/ghost_abl2_pre.sh; export EVAL_MARKETS=on-demand; round ghost-abl2 "" )
fi
# Moonshine Small through the same no-hold engine: the strongest existing early streaming model.
( export EVAL_MARKETS=on-demand; round ref-small "ref-small-nohold tiny-nohold UsefulSensors/moonshine-streaming-small" )
# The audio-prefixed baseline again, now guessing every 80 ms step (it hears new audio between transcript changes).
if aws s3 ls $B/runs/alm135/step8000/adapter.pt > /dev/null 2>&1; then
  printf "ALM=runs/alm135/step8000 ALMNAME=alm135s-on-p29 sh scripts/ghost_round.sh none noqwen\n" > /tmp/ghost_alm2_pre.sh
  ( PRE=/tmp/ghost_alm2_pre.sh; export EVAL_MARKETS=on-demand; round ghost-alm2 "" )
fi
# The stronger audio-prefixed baseline (alm2), guessing every step.
if aws s3 ls $B/runs/alm2/step20000/adapter.pt > /dev/null 2>&1; then
  printf "ALM=runs/alm2/step20000 ALMNAME=alm2-on-p29 sh scripts/ghost_round.sh none noqwen\n" > /tmp/ghost_alm3_pre.sh
  ( PRE=/tmp/ghost_alm3_pre.sh; export EVAL_MARKETS=on-demand; round ghost-alm3 "" )
fi
# Round 3: the next main model (lead 0.3) and the 400/500 ms runs.
if aws s3 ls $B/runs/antic_gen2/step20000/config.json > /dev/null 2>&1; then
  ( SETS_EXTRA=slurp_test; export EVAL_MARKETS=on-demand; round antic-gen2 "antic_gen2-step20000 earlybird runs/antic_gen2/step20000" )
  printf "sh scripts/ghost_round.sh 'antic_gen2=runs/antic_gen2/step20000' noqwen\n" > /tmp/ghost_gen2_pre.sh
  ( PRE=/tmp/ghost_gen2_pre.sh; export EVAL_MARKETS=on-demand; round ghost-gen2 "" )
fi
for a in abl_lead40_mh05 abl_lead50_mh05; do
  aws s3 ls $B/runs/$a/step8000/config.json > /dev/null 2>&1 || continue
  ( export EVAL_MARKETS=on-demand; round $a "$a-step8000 earlybird runs/$a/step8000" )
done
# Size track: the packed main-model variants on every set (+ SLURP) and their guesses.
for v in gen2_q4fe8 gen2_q4 gen2_q3fe8; do
  aws s3 ls $B/runs/${v}_pack/size.txt > /dev/null 2>&1 || continue
  ( SETS_EXTRA=slurp_test; export EVAL_MARKETS=on-demand; round $v "$v earlybird runs/${v}_pack" )
done
if aws s3 ls $B/runs/gen2_q3fe8_pack/size.txt > /dev/null 2>&1; then
  printf "sh scripts/ghost_round.sh 'gen2_q4fe8=runs/gen2_q4fe8_pack gen2_q4=runs/gen2_q4_pack gen2_q3fe8=runs/gen2_q3fe8_pack' noqwen\n" > /tmp/ghost_size_pre.sh
  ( PRE=/tmp/ghost_size_pre.sh; export EVAL_MARKETS=on-demand; round ghost-size "" )
fi
# Rigor: the main model's extra seeds.
for s in 1 2; do
  aws s3 ls $B/runs/antic_gen2_s$s/step20000/config.json > /dev/null 2>&1 || continue
  ( SETS_EXTRA=slurp_test; export EVAL_MARKETS=on-demand; round antic-gen2-s$s "antic_gen2_s$s-step20000 earlybird runs/antic_gen2_s$s/step20000" )
done
if aws s3 ls $B/runs/antic_gen2_s2/step20000/config.json > /dev/null 2>&1; then
  printf "sh scripts/ghost_round.sh 'antic_gen2_s1=runs/antic_gen2_s1/step20000 antic_gen2_s2=runs/antic_gen2_s2/step20000' noqwen\n" > /tmp/ghost_seeds_pre.sh
  ( PRE=/tmp/ghost_seeds_pre.sh; export EVAL_MARKETS=on-demand; round ghost-seeds "" )
fi
# Training with the padding frames (pad.sh): decoded with them (earlybird-pad).
for r in gen2_pad:5000 abl_pad30:8000 abl_pad20:8000; do
  n=${r%%:*}; s=${r#*:}
  aws s3 ls $B/runs/$n/step$s/config.json > /dev/null 2>&1 || continue
  ( export EVAL_MARKETS=on-demand; round $n "$n-step$s-pad earlybird-pad runs/$n/step$s" )
done
if aws s3 ls $B/runs/abl_pad20/step8000/config.json > /dev/null 2>&1; then
  printf "sh scripts/ghost_round.sh 'gen2_pad=runs/gen2_pad/step5000 abl_pad30=runs/abl_pad30/step8000 antic_gen2_pad=runs/antic_gen2/step20000' noqwen\n" > /tmp/ghost_pad_pre.sh
  ( PRE=/tmp/ghost_pad_pre.sh; export EVAL_MARKETS=on-demand; round ghost-pad "" )
fi
# The padded main model's seeds and 4-bit version (pad2.sh), decoded with padding; noise for gen2_pad.
for s in 1 2; do
  aws s3 ls $B/runs/gen2_pad_s$s/step5000/config.json > /dev/null 2>&1 || continue
  ( export EVAL_MARKETS=on-demand; round gen2-pad-s$s "gen2_pad_s$s-step5000-pad earlybird-pad runs/gen2_pad_s$s/step5000" )
done
aws s3 ls $B/runs/gen2_pad_q4fe8_pack/size.txt > /dev/null 2>&1 && \
  ( export EVAL_MARKETS=on-demand; round gen2-pad-q4fe8 "gen2_pad_q4fe8-pad earlybird-pad runs/gen2_pad_q4fe8_pack" )
if aws s3 ls $B/runs/gen2_pad_q4fe8_pack/size.txt > /dev/null 2>&1 && aws s3 ls $B/runs/gen2_pad_s2/step5000/config.json > /dev/null 2>&1; then
  printf "sh scripts/ghost_round.sh 'gen2_pad_s1=runs/gen2_pad_s1/step5000 gen2_pad_s2=runs/gen2_pad_s2/step5000 gen2_pad_q4fe8=runs/gen2_pad_q4fe8_pack' noqwen\n" > /tmp/ghost_pad2_pre.sh
  ( PRE=/tmp/ghost_pad2_pre.sh; export EVAL_MARKETS=on-demand; round ghost-pad2 "" )
fi
aws s3 ls $B/runs/gen2_pad/step5000/config.json > /dev/null 2>&1 && \
  ( ONLY_SETS="ls_noisy_white10 ls_noisy_white5 ls_noisy_babble10 ls_noisy_babble5"; export EVAL_MARKETS=on-demand
    round noisy-pad "gen2_pad-step5000-pad earlybird-pad runs/gen2_pad/step5000" "ref-tiny-nohold-pad tiny-nohold-pad UsefulSensors/moonshine-streaming-tiny" )
