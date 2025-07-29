# PolitScanner

## Abstract

Swiss politicians lie.
And they mostly get away with it.
"One reason for this is that fact-checks, which can only be carried out retrospectively, are surprisingly ineffective. Listeners still remember the false information. The correction is forgotten." — Philipp Gerlach.
This technical report provides a comprehensive overview of the artificial intelligence components of the PolitScanner project.
It aims to automatically detect false narratives and fake news in the speeches of Swiss politicians while avoiding the inaccuracies inherent in Large Language Models.

The training code can be found on [GitHub](https://github.com/lenamerkli/PolitScanner).

This is an entry for the Swiss AI Challenge 2025.
More information can be found at [www.ki-challenge.ch](https://www.ki-challenge.ch/).

## Table of Contents

0. [Abstract](#abstract)
1. [Paper](#paper)
2. [Installation](#installation)
3. [Usage](#usage)
4. [License](#license)
5. [Citation](#citation)

## Paper

Read the full paper [here](https://huggingface.co/lenamerkli/PolitScanner/blob/main/PolitScanner.pdf).

## Installation

**Note: inference only**

This installation guide is for Nobara Linux.
Other distributions should work as well.

For the full installation guide, see [the development readme](https://github.com/lenamerkli/PolitScanner/blob/main/README.md).

### Increase memlock

Add (or update) the following lines to `/etc/security/limits.conf`:
```text
* soft memlock 50331648
* hard memlock 50331648
```

### Git

Install git:

```shell
sudo dnf install git
```

Clone the PolitScanner repository:

```shell
git clone https://huggingface.co/lenamerkli/PolitScanner
cd PolitScanner
```

### Python

Install Python version 3.12.10 with the following command:

```shell
sudo dnf install python3.12-0:3.12.10-1.fc41.x86_64
```

Install the Python virtual environment package:

```shell
sudo dnf install python3-virtualenv
```

Create the virtual environment:

```shell
./create_venv.sh
```

Activate the virtual environment:

```shell
source .venv/bin/activate
```

### llama.cpp

If llama.cpp is not installed, check the [development readme](https://github.com/lenamerkli/PolitScanner/blob/main/README.md) for instructions.

### Download models

Run the downloader:

```shell
python3 download_ggufs.py
```

Move the PolitScanner model:

```shell
mv ./Qwen3-1.7B-PolitScanner-Q5_K_S.gguf /opt/llms/Qwen3-1.7B-PolitScanner-Q5_K_S.gguf
```

## Usage

Copy the political speech (preferably in swiss high german) to the `input.txt` file.

Run the program:

```shell
python3 main.py
```

The output will be written to the `output.txt` file.

## License

[MIT License](https://github.com/lenamerkli/PolitScanner/blob/main/LICENSE)

## Citation

bibtex:
```bibtex
@misc{merkli2025politscanner,
    title = {PolitScanner: Automatic Detection of common Incorrect Statements in Speeches of Swiss Politicians},
    author = {Lena Merkli},
    year = {2025},
    month = {07},
    url = {https://huggingface.co/lenamerkli/PolitScanner}
}
```
biblatex:
```biblatex
@online{merkli2025politscanner,
    title = {PolitScanner: Automatic Detection of common Incorrect Statements in Speeches of Swiss Politicians},
    author = {Lena Merkli},
    year = {2025},
    month = {07},
    url = {https://huggingface.co/lenamerkli/PolitScanner}
}
```
