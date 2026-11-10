#!/bin/sh
# Run a command and stop it if the machine runs short of memory.
# usage: scripts/guard.sh [-f MIN_FREE_PCT] [-s MAX_SWAP_GROWTH_MB] cmd ...
MINFREE=12; MAXSWAP=3000
while getopts f:s: o; do case $o in f) MINFREE=$OPTARG ;; s) MAXSWAP=$OPTARG ;; esac; done
shift $((OPTIND - 1))
free_pct() { memory_pressure | awk -F': ' '/free percentage/ {print $2+0}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{print int($6)}'; }
F=$(free_pct)
[ "$F" -lt "$MINFREE" ] && { echo "guard: only $F% memory free, not starting"; exit 75; }
S0=$(swap_mb)
"$@" &
PID=$!
trap 'pkill -P $PID; kill $PID 2>/dev/null' INT TERM
while kill -0 $PID 2>/dev/null; do
  sleep 5
  F=$(free_pct); S=$(swap_mb)
  if [ "$F" -lt "$MINFREE" ] || [ $((S - S0)) -gt "$MAXSWAP" ]; then
    echo "guard: memory free $F%, swap +$((S - S0)) MB; stopping the job"
    pkill -P $PID; kill $PID 2>/dev/null; sleep 2; pkill -9 -P $PID; kill -9 $PID 2>/dev/null
    exit 75
  fi
done
wait $PID
