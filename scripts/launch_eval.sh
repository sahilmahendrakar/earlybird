#!/bin/sh
# Start one evaluation round (scripts/eval_box.sh) on a throwaway CPU machine: spot
# c7i.8xlarge (32 cores) in the first zone with capacity, on-demand if none. It terminates
# itself when done (or after 4 hours whatever happens). Run from the dev box or the Mac.
# usage: scripts/launch_eval.sh TAG 'NAME RECOGNIZER [CHECKPOINT]' ['NAME RECOGNIZER [CHECKPOINT]' ...]
#   e.g. scripts/launch_eval.sh s10-step20000 \
#          's10_from_tiny-step20000 earlybird runs/s10_from_tiny/step20000' \
#          's10_from_tiny-step20000-pad earlybird-pad runs/s10_from_tiny/step20000'
set -e
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
TAG=$1; shift
TYPE=${EVAL_TYPE:-c7i.8xlarge}
f=$(mktemp); for line in "$@"; do echo "$line"; done > $f
aws s3 cp $f $B/eval/$TAG/plan.txt --only-show-errors; rm -f $f
ud=$(mktemp)
cat > $ud <<UD
#!/bin/bash
exec > /var/log/earlybird-eval.log 2>&1
set -x
shutdown -h +240
sudo -iu ubuntu sh -c 'mkdir -p earlybird && aws s3 cp $B/code/earlybird.tar.gz - | tar xz -C earlybird'
sudo -iu ubuntu sh -c 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -iu ubuntu sh -c 'cd earlybird && ~/.local/bin/uv sync -q'
sudo -iu ubuntu sh -c 'cd earlybird && sh scripts/eval_box.sh $TAG'
aws s3 cp /var/log/earlybird-eval.log $B/results/eval/$TAG.boot.log --only-show-errors || true
shutdown -h now
UD
for market in ${EVAL_MARKETS:-spot on-demand}; do
  for SN in $EARLYBIRD_SUBNETS; do
    opt=""; [ $market = spot ] && opt="--instance-market-options MarketType=spot"
    id=$(aws ec2 run-instances --region us-east-1 --image-id ami-00803022f231f100e --instance-type $TYPE \
      --subnet-id $SN --security-group-ids $EARLYBIRD_SG --associate-public-ip-address \
      --iam-instance-profile Name=earlybird-box $opt --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
      --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' --user-data file://$ud \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=earlybird-eval-$TAG},{Key=Project,Value=earlybird}]" \
        'ResourceType=volume,Tags=[{Key=Project,Value=earlybird}]' \
      --query 'Instances[0].InstanceId' --output text 2>$ud.err) && {
        rm -f $ud $ud.err
        sh "$(dirname "$0")/runlog.sh" ec2-dev "eval $TAG: launched $id ($TYPE $market, $SN) for: $(for l in "$@"; do printf '%s; ' "${l%% *}"; done)"
        echo $id; exit 0; }
    grep -q -e UnauthorizedOperation -e AccessDenied $ud.err && { head -c 300 $ud.err >&2; echo >&2; rm -f $ud $ud.err; exit 1; }
  done
done
echo "no capacity for $TYPE: $(head -c 200 $ud.err)" >&2; rm -f $ud $ud.err; exit 1
