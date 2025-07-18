from os.path import abspath
from os import listdir


def main() -> None:
    for file in listdir(abspath('./raw/')):
        if not file.endswith('.txt'):
            continue
        with open(abspath(f'./raw/{file}'), 'r') as f:
            text = f.read()
        print(f"{file.split('.')[0]}. {text.split('\n---\n')[0]} --- {text.split('\n---\n')[2]}")


if __name__ == '__main__':
    main()
