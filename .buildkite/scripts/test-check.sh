#!/usr/bin/env bash

cat /etc/issue

# install uv/compilers and check env
apt-get update && apt-get install -y curl g++ gcc make
curl -LsSf https://astral.sh/uv/install.sh | env UV_VERSION=0.8.15 sh

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export PATH="$HOME/.local/bin:/usr/local/nvidia/bin:$PATH"
nvidia-smi
uv --version

# install code and run tests
uv venv testvenv --python 3.12
source testvenv/bin/activate

export UV_TORCH_BACKEND=cu130
export HF_HOME=/model-cache
uv pip install .[dev] --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130

make test
