#!/bin/sh
# Append one line to the shared run log, s3://…/runlog.md: UTC time, machine, event.
# usage: scripts/runlog.sh MACHINE 'what happened'    (MACHINE: mac, ec2-dev, or an instance id;
# on an EC2 instance pass "" to use its own id). S3 has no append, so this reads, adds a line
# and writes back. Writers on one machine take turns (flock); two machines writing in the same
# second can still drop a line, which is rare enough here.
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
m=$1; shift
if [ -z "$m" ]; then
  t=$(curl -s -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token)
  m=$(curl -s -H "X-aws-ec2-metadata-token: $t" http://169.254.169.254/latest/meta-data/instance-id)
fi
exec 9>/tmp/earlybird-runlog.lock; flock -w 60 9
f=$(mktemp)
aws s3 cp $B/runlog.md $f --only-show-errors 2>/dev/null || printf '# Earlybird run log\n\nAppend-only. One line per event: UTC time | machine | event.\n\n' > $f
echo "- $(date -u '+%F %H:%MZ') | $m | $*" >> $f
aws s3 cp $f $B/runlog.md --only-show-errors --content-type text/markdown
rm -f $f
