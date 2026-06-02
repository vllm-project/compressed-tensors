#!/usr/bin/env bash
set -euo pipefail

# print OS info for debugging
cat /etc/issue

# install system dependencies and uv
apt-get update && apt-get install -y curl g++ gcc make
curl -LsSf https://astral.sh/uv/install.sh | env UV_VERSION=0.8.15 sh

export PATH="$HOME/.local/bin:$PATH"

# fetch full history and tags (setuptools_scm derives version from git tags)
git fetch --tags --unshallow 2>/dev/null || git fetch --tags

# create venv and install dependencies
uv venv qualityvenv --python 3.12
source qualityvenv/bin/activate

uv pip install .[dev]

# run quality checks
make quality
