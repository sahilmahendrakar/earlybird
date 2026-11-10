#!/bin/sh
# Run one command on a throwaway spot GPU machine and terminate it afterwards (or after
# MAX_HOURS whatever happens). The machine unpacks the code from S3, installs the
# environment, syncs the data sets named in DATA from S3, runs the command from the repo as
# ubuntu, copies runs/NAME* to s3://…/runs/, and logs start and end to the run log.
# usage: DATA="ls_clean_100 ls_clean_360 kokoro_train" scripts/launch_job.sh NAME 'command ...'
set -e
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
NAME=$1; CMD=$2
TYPES=${TYPES:-"g6.xlarge g5.xlarge g6.2xlarge"}; MAX_HOURS=${MAX_HOURS:-3}
MARKETS=${MARKETS:-"spot on-demand"}   # spot first; on-demand only if no zone has spot capacity
ud=$(mktemp)
cat > $ud <<UD
#!/bin/bash
exec > /var/log/earlybird-job.log 2>&1
set -x
shutdown -h +$((MAX_HOURS * 60))
sudo -iu ubuntu sh -c 'mkdir -p earlybird && aws s3 cp $B/code/earlybird.tar.gz - | tar xz -C earlybird'
sudo -iu ubuntu sh -c 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -iu ubuntu sh -c 'cd earlybird && ~/.local/bin/uv sync -q && mkdir -p runs data && for d in $DATA; do aws s3 sync $B/data/\$d data/\$d --only-show-errors --exclude "dl/*"; done'
sudo -iu ubuntu sh -c 'cd earlybird && sh scripts/runlog.sh "" "job $NAME: started" && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $CMD > runs/$NAME.log 2>&1; sh scripts/runlog.sh "" "job $NAME: exited \$? ; last: \$(tail -1 runs/$NAME/log.jsonl 2>/dev/null | cut -c1-160)"; aws s3 sync runs $B/runs --only-show-errors --exclude "*" --include "$NAME*"'
aws s3 cp /var/log/earlybird-job.log $B/runs/$NAME.boot.log --only-show-errors || true
shutdown -h now
UD
for M in $MARKETS; do for T in $TYPES; do
  opt=""; [ $M = spot ] && opt="--instance-market-options MarketType=spot"
  for SN in $EARLYBIRD_SUBNETS; do
    id=$(aws ec2 run-instances --region us-east-1 --image-id ami-00803022f231f100e --instance-type $T \
      --subnet-id $SN --security-group-ids $EARLYBIRD_SG --associate-public-ip-address \
      --iam-instance-profile Name=earlybird-box $opt --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
      --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' --user-data file://$ud \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=earlybird-job-$NAME},{Key=Project,Value=earlybird}]" \
        'ResourceType=volume,Tags=[{Key=Project,Value=earlybird}]' \
      --query 'Instances[0].InstanceId' --output text 2>$ud.err) && {
        rm -f $ud $ud.err
        sh "$(dirname "$0")/runlog.sh" ec2-dev "job $NAME: launched $id ($T $M, $SN, max ${MAX_HOURS} h): $CMD"
        echo $id; exit 0; }
    grep -q -e UnauthorizedOperation -e AccessDenied $ud.err && { head -c 300 $ud.err >&2; echo >&2; rm -f $ud $ud.err; exit 1; }
  done
done; done
echo "no capacity for $TYPES: $(head -c 200 $ud.err)" >&2; rm -f $ud $ud.err; exit 1
