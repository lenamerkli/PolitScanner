from pathlib import Path
from os import listdir
from json import load as json_load
import typing as t


TEMPLATE = """
root ::= "```python\\n[" list "]\\n```"
list ::= %%
"""
TEMPLATE_ITEM = '["\'%%\'"]?'
SEPARATOR = ' [", "]? '


def main() -> None:
    titles = []
    for file in listdir(Path(__file__).resolve().parent.parent.absolute().__str__() + '/data/parsed/'):
        if file.endswith('.json') and file != 'hashes.json':
            with open(Path(__file__).resolve().parent.parent.absolute().__str__() + f"/data/parsed/{file}", 'r') as f:
                data: t.Dict[str, t.Any] = json_load(f)
            titles.append(data['topic'])
    titles.sort()
    items = []
    for title in titles:
        items.append(TEMPLATE_ITEM.replace('%%', title))
    grammar = TEMPLATE.replace('%%', SEPARATOR.join(items))
    with open('grammar.gbnf', 'w', encoding='utf-8') as f:
        f.write(grammar)


if __name__ == '__main__':
    main()
