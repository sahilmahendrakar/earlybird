#!/bin/sh
# First-boot setup of the EC2 training machine (run as root from the instance's user data,
# after the repo has been unpacked to /home/ubuntu/earlybird). Installs the environment,
# pulls the Kokoro sets from S3, and installs two things that survive restarts:
#   earlybird-jobs.service  runs scripts/box_jobs.sh at every boot (it skips what is done)
#   a cron entry            runs scripts/box_idle.sh every 5 minutes
set -e
B=s3://${EARLYBIRD_BUCKET:?set EARLYBIRD_BUCKET}
R=/home/ubuntu/earlybird
apt-get install -y -q espeak-ng >/dev/null 2>&1 || true
sudo -iu ubuntu sh -c "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
sudo -iu ubuntu sh -c "cd $R && mkdir -p runs data && ~/.local/bin/uv sync -q && ~/.local/bin/uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
sudo -iu ubuntu sh -c "cd $R && for d in kokoro_train kokoro_test text librispeech_test_clean; do aws s3 sync $B/data/\$d data/\$d --only-show-errors; done"
cat > /etc/systemd/system/earlybird-jobs.service <<UNIT
[Unit]
Description=Earlybird data preparation
After=network-online.target
Wants=network-online.target
[Service]
OOMPolicy=continue
MemoryMax=26G
MemorySwapMax=0
User=ubuntu
WorkingDirectory=$R
ExecStart=/bin/sh scripts/box_jobs.sh
[Install]
WantedBy=multi-user.target
UNIT
echo "*/5 * * * * root /bin/sh $R/scripts/box_idle.sh" > /etc/cron.d/earlybird-idle
systemctl daemon-reload
systemctl enable --now earlybird-jobs.service
