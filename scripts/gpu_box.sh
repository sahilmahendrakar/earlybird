#!/bin/sh
# Drive a training machine on EC2 from the Mac.
#   scripts/gpu_box.sh push HOST          copy the repo (code, test sets, training audio so far) and install
#   scripts/gpu_box.sh run HOST NAME CMD  run CMD on the machine in the background, log to runs/NAME.log
#   scripts/gpu_box.sh tail HOST NAME     follow that log
#   scripts/gpu_box.sh pull HOST PATH     copy PATH (relative to the repo) back to the Mac
# HOST is the machine's public address; the key is ~/.ssh/earlybird.pem.
set -e
cd "$(dirname "$0")/.."
cmd=$1; host=ubuntu@$2
SSH="ssh -i $HOME/.ssh/earlybird.pem -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
case $cmd in
  push)
    rsync -az -e "$SSH" --exclude .venv --exclude .git --exclude __pycache__ --exclude models --exclude data/dl \
      --exclude results/raw --exclude 'data/kokoro_train/synth.log' ./ $host:earlybird/
    $SSH $host 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; sudo apt-get install -y -q espeak-ng >/dev/null 2>&1; cd earlybird && mkdir -p runs && ~/.local/bin/uv sync -q && ~/.local/bin/uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"' ;;
  run)
    name=$3; shift 3
    $SSH $host "cd earlybird && nohup sh -c '$*' > runs/$name.log 2>&1 &" ;;
  tail) $SSH $host "tail -n ${4:-20} earlybird/runs/$3.log" ;;
  pull) rsync -az -e "$SSH" $host:earlybird/$3 "$(dirname "$3")/" ;;
esac
