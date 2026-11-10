#!/bin/sh
# tools/ssm.sh SCRIPTFILE [wait_secs]: run a local shell script on the GPU box as ubuntu in ~/earlybird and print its output
ID=${EARLYBIRD_BOX:?set EARLYBIRD_BOX}; B=s3://$EARLYBIRD_BUCKET
aws s3 cp "$1" $B/tmp/ssm-cmd.sh --only-show-errors
CID=$(aws ssm send-command --instance-ids $ID --document-name AWS-RunShellScript --timeout-seconds 3600 \
  --parameters '{"commands":["aws s3 cp s3://$EARLYBIRD_BUCKET/tmp/ssm-cmd.sh /tmp/ssm-cmd.sh --only-show-errors && chmod 755 /tmp/ssm-cmd.sh && sudo -iu ubuntu sh -c \"cd ~/earlybird && sh /tmp/ssm-cmd.sh\" 2>&1 | tail -60"],"executionTimeout":["3600"]}' \
  --query Command.CommandId --output text)
W=${2:-30}; t=0
while :; do sleep 5; t=$((t+5)); st=$(aws ssm get-command-invocation --command-id $CID --instance-id $ID --query Status --output text 2>/dev/null)
  case $st in Success|Failed|TimedOut|Cancelled) break;; esac; [ $t -ge $W ] && { echo "(still $st, cmd $CID)"; exit 0; }; done
aws ssm get-command-invocation --command-id $CID --instance-id $ID --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
