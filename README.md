# PolitScanner
An entry for the swiss AI challenge 2025 to detect common false narratives and fake news in the speeches of Swiss politicians.

## Installation

### System requirements

- `Nobara-42-Official-Nvidia`, a fork of Fedora Linux.
- an nvidia GPU with at least 8GB VRAM and a CUDA compute capability of at least 8.9.
- toolbox enabled

A list of supported GPUs can be found in [Supported GPUs](#supported-gpus).

### Increase memlock

Add (or update) the following lines to `/etc/security/limits.conf`:
```text
* soft memlock 50331648
* hard memlock 50331648
```


### System updates

Update the entire system through the `Update System` Nobara app.

### Python

Install Python version 3.12.10 with the following command:

```shell
sudo dnf install python3.12-0:3.12.10-1.fc41.x86_64
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

Run the llama.cpp installation script:

```shell
./install_llama_cpp.sh
```

Copy the build folder to the host system and exit the toolbox:

```shell
sudo cp -r /tmp/llama_cpp/llama.cpp/build /opt
exit
```

Rename the build folder:

```shell
sudo mv /opt/build /opt/llama.cpp
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

## Supported GPUs

The developers of PolitScanner provide support for the following GPUs, as of April 2025:

- RTX 4060 to RTX 4090
- RTX 4060 Mobile to RTX 4090 Mobile
- RTX 5060 Ti to RTX 5090
- RTX 5070 Ti Mobile to RTX 5090 Mobile
- RTX Pro Blackwell Series
- B200 and GB200
- H100 and H200
- L4, L40 and L40S

Other NVIDIA GPUs may work as well. It is intentional that the RTX 4050 Mobile is missing on this list.
