# PolitScanner
An entry for the swiss AI challenge 2025 to detect common false narratives and fake news in the speeches of Swiss politicians.

## Table of Contents
- [Installation](#installation)
  - [System Requirements](#system-requirements)
  - [Increase memlock](#increase-memlock)
  - [System Updates](#system-updates)
  - [GCC](#gcc)
  - [Git](#git)
  - [Python](#python)
    - [Virtual environment](#virtual-environment)
  - [CUDA](#cuda)
    - [ToolBox](#toolbox)
  - [llama.cpp](#llamacpp)
    - [Shared memory](#enable-shared-memory-optional)
  - [Unsloth](#unsloth)
- [Supported GPUs](#supported-gpus)
  - [Desktop GPUs](#desktop-gpus)
  - [Mobile GPUs](#mobile-gpus)
  - [Professional GPUs](#professional-gpus)
  - [Data Center GPUs](#data-center-gpus)
- [License](#license)

## Installation

### System requirements

- `Nobara-42-Official-Nvidia`, a fork of Fedora Linux.
- an nvidia GPU with at least 8GB VRAM, a CUDA compute capability of at least 8.9 and support for CUDA 12.8.
- toolbox enabled
- `amd64` or `x86_64` architecture CPU

A list of supported GPUs can be found in [Supported GPUs](#supported-gpus).

### Increase memlock

Add (or update) the following lines to `/etc/security/limits.conf`:
```text
* soft memlock 50331648
* hard memlock 50331648
```


### System updates

Update the entire system through the `Update System` Nobara app.

### GCC

Install the gcc and gcc-c++ compilers:

```shell
sudo dnf install gcc gcc-c++
```

### Git

Install git:

```shell
sudo dnf install git
```

Clone the PolitScanner repository:

```shell
git clone https://github.com/lenamerkli/PolitScanner.git
cd PolitScanner
```

### Python

Install Python version 3.12.10 with the following command:

```shell
sudo dnf install python3.12-0:3.12.10-1.fc41.x86_64
```

Install the Python development packages:

```shell
sudo dnf install python3.12-devel-0:3.12.10-1.fc41
```

The Python virtual environment package is also required:

```shell
sudo dnf install python3-virtualenv
```

#### Virtual environment

Create the virtual environment using the provided bash script:

```shell
./create_venv.sh
```

### CUDA

Note: Make sure to use version 12.8.*, as it is required by this project.

Install the `cuda-devel` package through the `Nobara Package Manager` app.

#### ToolBox

Create the toolbox environment with the following command on the host:

```shell
toolbox create --image registry.fedoraproject.org/fedora-toolbox:41 --container fedora-toolbox-41-cuda
```

Enter the toolbox with the following command on host. All following commands in this section are to be executed in the toolbox.

```shell
toolbox enter fedora-toolbox-41-cuda
```

Synchronize the DNF package manager:

```shell
sudo dnf distro-sync
```

Install development tools:

```shell
sudo dnf install @c-development @development-tools cmake
```

Install python:

```shell
sudo dnf install python3.12
```

Add the NVIDIA CUDA repository to DNF:

```shell
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo
```

Synchronize the DNF package manager again:

```shell
sudo dnf distro-sync
```

Detect if the host is supplying the nvidia driver libraries:

```shell
ls -la /usr/lib64/libcuda.so.1
```

**Begin of `libcuda.so.1` is missing**

Install the nvidia driver libraries:

```shell
sudo dnf install nvidia-driver-cuda nvidia-driver-libs nvidia-driver-cuda-libs nvidia-persistenced
```

**End of `libcuda.so.1` is missing**

**Begin of `libcuda.so.1` exists**

The toolbox RPM database has to be updated accordingly.

```shell
sudo dnf download --destdir=/tmp/nvidia-driver-libs --resolve --arch x86_64 nvidia-driver-cuda nvidia-driver-libs nvidia-driver-cuda-libs nvidia-persistenced
sudo rpm --install --verbose --hash --justdb /tmp/nvidia-driver-libs/*
```

Check if the RPM database has been updated:

```shell
sudo dnf install nvidia-driver-cuda nvidia-driver-libs nvidia-driver-cuda-libs nvidia-persistenced
```

**End of `libcuda.so.1` exists**

Install the CUDA metapackage:

```shell
sudo dnf install cuda
```

Create a profile script for CUDA:

```shell
sudo sh -c 'echo "export PATH=\$PATH:/usr/local/cuda/bin" >> /etc/profile.d/cuda.sh'
sudo chmod +x /etc/profile.d/cuda.sh
```

Source the profile script to update the environment:

```shell
source /etc/profile.d/cuda.sh
```

Verify the CUDA installation:

```shell
nvcc --version
```

The output should look similar to this:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

### llama.cpp

Enter the CUDA toolbox:

```shell
toolbox enter fedora-toolbox-41-cuda
```

Install curl:

```shell
sudo dnf install curl libcurl libcurl-devel
```

Run the llama.cpp installation script:

```shell
./install_llama_cpp.sh
```

Copy the build folder to the host system.

```shell
sudo cp -r /tmp/llama_cpp/llama.cpp/build /opt
```

Copy the llama.cpp scripts into the directory:

```shell
sudo cp /tmp/llama_cpp/llama.cpp/convert_hf_to_gguf.py /opt/build/bin/
sudo cp /tmp/llama_cpp/llama.cpp/convert_hf_to_gguf_update.py /opt/build/bin/
sudo cp /tmp/llama_cpp/llama.cpp/convert_llama_ggml_to_gguf.py /opt/build/bin/
sudo cp /tmp/llama_cpp/llama.cpp/convert_lora_to_gguf.py /opt/build/bin/
```

Move the build directory into the home directory:

```shell
sudo mv /opt/build/ ~/build/
```

Exit the toolbox:

```shell
exit
```

Move the build folder:

```shell
sudo mv ~/build /opt/llama.cpp
```

Modify the environment variables in `~/.bashrc` by adding the following lines to the end of the file:

```bash
export PATH="/opt/llama.cpp/bin:$PATH"
export LD_LIBRARY_PATH="/opt/llama.cpp/lib:$LD_LIBRARY_PATH"
```

#### Enable shared memory (optional)

To enable shared memory support for llama.cpp, add the following line to the end of `~/.bashrc`:

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
```

### Unsloth

Enter the toolbox:

```shell
toolbox enter fedora-toolbox-41-cuda
```

Go to the `PolitScanner` directory using `cd`.

Set the required environment variables:

```shell
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
```

Install the python development tools:

```shell
sudo dnf install python3.12-devel ninja-build
```

Create the required symlink:

```shell
sudo ln -s /usr/local/cuda/bin/nvcc /usr/bin/nvcc
```

Activate the virtual environment:

```shell
source ./.venv/bin/activate
```

Install flash attention:

```shell
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir --upgrade --use-pep517
```

Install unsloth:

```shell
pip install "unsloth[cu128-ampere-torch270] @ git+https://github.com/unslothai/unsloth.git" --force-reinstall --no-cache-dir --upgrade --use-pep517
```

Update protobuf:

```shell
pip install protobuf==5.29.4 --upgrade
```

## Supported GPUs

The developers of PolitScanner provide support for the following GPUs, as of April 2025:

### Desktop GPUs
- RTX 4060
- RTX 4060 Ti
- RTX 4070
- RTX 4070 Super
- RTX 4070 Ti
- RTX 4070 Ti Super
- RTX 4080
- RTX 4080 Super
- RTX 4090
- RTX 4090 D
- RTX 5060
- RTX 5060 Ti
- RTX 5070
- RTX 5070 Ti
- RTX 5080
- RTX 5080 Ti
- RTX 5090
- RTX 5090 D

### Mobile GPUs
- RTX 4060 Mobile
- RTX 4070 Mobile
- RTX 4080 Mobile
- RTX 4090 Mobile
- RTX 5060 Mobile
- RTX 5070 Mobile
- RTX 5080 Mobile
- RTX 5090 Mobile

### Professional GPUs
- RTX 2000 Ada
- RTX 4000 Ada
- RTX 4000 SFF Ada
- RTX 4500 Ada
- RTX 5000 Ada
- RTX 5880 Ada
- RTX 6000 Ada
- RTX PRO 4000 Blackwell
- RTX PRO 4500 Blackwell
- RTX PRO 5000 Blackwell
- RTX PRO 6000 Blackwell
- RTX PRO 6000 Blackwell Max-Q

### Data Center GPUs
- L4
- L40
- L40S
- H100
- H200
- GH200
- B200
- GB200

Other NVIDIA GPUs may work as well, but the developers of PolitScanner do not provide support or bug fixes for them. It is intentional that the RTX 4050 Mobile is missing on this list.

## License

[MIT License](LICENSE)
