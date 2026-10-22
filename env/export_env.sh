#!/usr/bin/env bash
# 実行環境を env/ に記録する（conda activate lerobot した状態で実行）
set -euo pipefail
cd "$(dirname "$0")"
pip freeze --exclude-editable > requirements-lock.txt
conda env export --no-builds | grep -v '^prefix:' > conda-environment.yml
{
  echo "date: $(date -Iseconds)"
  echo "os: $(lsb_release -ds)"
  echo "kernel: $(uname -r)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu: /'
  python -c "import torch;print('torch:',torch.__version__,'cuda:',torch.version.cuda)"
  python -c "import importlib.metadata as m;print('lerobot:',m.version('lerobot'))"
  echo "MUJOCO_GL: ${MUJOCO_GL:-unset}"
} > system.txt
cat system.txt
