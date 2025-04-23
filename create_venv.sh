#!/usr/bin/env bash

# exit on error
set -e

# print commands
set -v

# create virtual environment
/usr/bin/python3.12 -m venv .venv

# install packages
.venv/bin/pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip
.venv/bin/pip3 install wheel==0.46.1 setuptools==79.0.0
.venv/bin/pip3 install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# test installation (raise error if cuda is not available)
.venv/bin/python3 -c "import torch; assert torch.cuda.is_available()"

echo "Virtual environment successfully created"
