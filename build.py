from pathlib import Path
from shutil import copyfile, rmtree
from os.path import join, exists
import typing as t


def relative_path(path: t.Union[str, Path]) -> str:
    return join(Path(__file__).resolve().parent, path)


def main() -> None:
    """
    Make the project ready to upload to HuggingFace.
    :return: None
    """
    # Remove the ./build directory if it exists
    if exists('./build'):
        rmtree('./build')
    Path('./build').mkdir()
    Path('./build/util').mkdir()
    Path('./build/sentence_splitter').mkdir()
    copyfile(relative_path('./inference/README.md'), Path('./build/README.md'))
    copyfile(relative_path('./inference/create_venv.sh'), Path('./build/create_venv.sh'))
    copyfile('/opt/llms/Qwen3-1.7B-PolitScanner-Q5_K_S.gguf', Path('./build/Qwen3-1.7B-PolitScanner-Q5_K_S.gguf'))
    copyfile(relative_path('./root_opt_llms/download_ggufs.py'), Path('./build/download_ggufs.py'))
    copyfile(relative_path('./root_opt_llms/index.json'), Path('./build/index.json'))
    copyfile(relative_path('./latex/main/politscanner.pdf'), Path('./build/politscanner.pdf'))
    copyfile(relative_path('./main/main.py'), Path('./build/main.py'))
    copyfile(relative_path('./main/input_example.txt'), Path('./build/input_example.txt'))
    copyfile(relative_path('./main/prompt.md'), Path('./build/prompt.md'))
    copyfile(relative_path('./util/llm.py'), Path('./build/util/llm.py'))
    copyfile(relative_path('./sentence_splitter/sentence_splitter.py'), Path('./build/sentence_splitter/sentence_splitter.py'))
    copyfile(relative_path('./sentence_splitter/function.py'), Path('./build/sentence_splitter/function.py'))


if __name__ == '__main__':
    main()
