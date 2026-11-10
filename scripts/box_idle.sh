#!/bin/sh
# Stop the EC2 training machine once it has been idle for half an hour. Run as root from
# cron every 5 minutes. Idle: GPU under 5% busy, load under 0.5, nobody logged in.
# `touch /run/earlybird-keep` holds the machine up until the file is removed or it restarts.
C=/run/earlybird-idle
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
busy=$(awk '{print ($1 >= 0.5)}' /proc/loadavg)
if [ "${gpu:-0}" -ge 5 ] || [ "$busy" = 1 ] || [ -n "$(who)" ] || [ -e /run/earlybird-keep ]; then
  echo 0 > $C
else
  n=$(($(cat $C 2>/dev/null || echo 0) + 1)); echo $n > $C
  [ $n -ge 6 ] && { logger "earlybird: idle for 30 min, stopping"; sh /home/ubuntu/earlybird/scripts/runlog.sh "" "idle 30 min: stopping"; shutdown -h now; }
fi
