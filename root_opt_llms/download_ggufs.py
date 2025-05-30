import sys
sys.path.append('/home/lena/Documents/python/PolitScanner/util')


from pathlib import Path
from os.path import join
from requests import get
from shutil import copyfile


MODELS = {
    'DeepSeek-R1-0528-Qwen3-8B-Q5_K_L.gguf': 'https://huggingface.co/bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-Q5_K_L.gguf',
    'Ministral-8B-Instruct-2410-Q4_K_S.gguf': 'https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF/resolve/main/Ministral-8B-Instruct-2410-Q4_K_S.gguf',
    'Qwen3-30B-A3B-Q5_K_M.gguf': 'https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/resolve/main/Qwen_Qwen3-30B-A3B-Q5_K_M.gguf',
    'Qwen3-8B-Q5_K_M.gguf': 'https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q5_K_M.gguf',
}


def download_large_file(url: str, directory: str, filename: str = None):
    Path(directory).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split('/')[-1].split('?')[0]
    filepath = join(directory, filename)
    with get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
    return filepath


def main() -> None:
    Path('/opt/llms').mkdir(exist_ok=True, parents=True)
    question = 'Which of these following models do you want to download?'
    for i, model in enumerate(MODELS.keys()):
        question += f"\n[{i + 1}] {model}"
    response = input(question).replace(' ', '').replace(';', ',')
    parsed = [int(x) for x in response.split(',')]
    if len(parsed) > 0:
        copyfile('./index.json', '/opt/llms/index.json')
    for i in parsed:
        model = list(MODELS.keys())[i - 1]
        url = MODELS[model]
        print(f"Downloading {model}")
        download_large_file(url, '/opt/llms/', model)
        print(f"Downloaded {model}")
    print('Done')


if __name__ == '__main__':
    main()


