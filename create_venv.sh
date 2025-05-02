#!/usr/bin/env bash

# exit on error
set -e

# print commands
set -v

# create virtual environment if it does not exist
if [ ! -d .venv ]; then
    /usr/bin/python3.12 -m venv .venv
fi

# install packages
.venv/bin/pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip
.venv/bin/pip3 install wheel==0.46.1 setuptools==79.0.0 flask==3.1.0 requests==2.32.3 tqdm==4.67.1 chromadb==1.0.7
.venv/bin/pip3 install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# test installation (raise error if cuda is not available)
.venv/bin/python3 -c "import torch; assert torch.cuda.is_available()"

echo "Virtual environment successfully created"
