#!/usr/bin/env bash

# print commands
set -v

# exit on error
set -e

# test installation (raise error if cuda is not available)
nvidia-smi > /dev/null
# .venv/bin/python3 -c "import torch; assert torch.cuda.is_available()"

# empty temporary directory
rm -rf /tmp/llama_cpp

# create temporary directory
mkdir -p /tmp/llama_cpp

# change directory
cd /tmp/llama_cpp

# clone the llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git

# navigate to the llama.cpp directory
cd llama.cpp

# Build with explicit CUDA configuration
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel

echo "llama.cpp installed successfully"
