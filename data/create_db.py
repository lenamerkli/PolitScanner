import chromadb
from os.path import abspath, exists
from os import listdir
import typing as t
from json import load as json_load


def main() -> None:
    """
    Creates and populates the ChromaDB vector database with political statements.

    :return: None
    """
    if exists(abspath('./database.chroma')):
        return None
    client = chromadb.PersistentClient(path=abspath('./database.chroma'))
    collection = client.create_collection(name='PolitScanner')
    for file in listdir(abspath('./parsed/')):
        if file.endswith('.json') and file != 'hashes.json':
            with open(abspath(f'./parsed/{file}'), 'r') as f:
                data: t.Dict[str, t.Any] = json_load(f)
            for i, statement in enumerate(data['original_statements']):
                collection.add(
                    documents=[statement],
                    ids=[f"{file.rsplit('.', 1)[0]}-{i}"]
                )
    return None



if __name__ == '__main__':
    main()
