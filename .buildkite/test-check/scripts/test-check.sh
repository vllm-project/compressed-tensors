#!/usr/bin/env bash
set -euo pipefail

# print OS info for debugging
cat /etc/issue

# fetch full history and tags (setuptools_scm derives version from git tags)
git fetch --tags --unshallow 2>/dev/null || git fetch --tags

# install system dependencies and uv
apt-get update && apt-get install -y curl g++ gcc make
curl -LsSf https://astral.sh/uv/install.sh | env UV_VERSION=0.11.18 sh

# set up GPU and path
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export PATH="$HOME/.local/bin:/usr/local/nvidia/bin:$PATH"
nvidia-smi

# create venv and install dependencies
uv venv testvenv --python 3.12
source testvenv/bin/activate

export UV_TORCH_BACKEND=cu130
export HF_HOME=/model-cache
uv pip install .[dev] --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130

# run tests
make test
