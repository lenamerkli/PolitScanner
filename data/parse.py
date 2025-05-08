import sys
sys.path.append('/home/lena/Documents/python/PolitScanner/util')

from json import load as json_load, dump as json_dump
from os import listdir
from os.path import abspath, exists
from subprocess import run, PIPE
from time import sleep
from tqdm import tqdm
from util.llm import LLaMaCPP

import typing as t



def main() -> None:
    llm = LLaMaCPP()
    llm.set_model('Qwen3-30B-A3B-Q5_K_M.gguf')
    llm.load_model()
    try:
        while llm.is_loading() or not llm.is_running():
            sleep(1)
        sleep(2)
        system_prompt = llm.get_system_message()
        with open(abspath('./prompt.md'), 'r') as f:
            prompt_template = f.read()
        with open(abspath('./grammar.gbnf'), 'r') as f:
            grammar = f.read()
        hash_db = abspath('./parsed/hashes.json')
        if not exists(hash_db):
            with open(hash_db, 'w') as f:
                json_dump({}, f)
        with open(hash_db, 'r') as f:
            hashes: t.Dict[str, str] = json_load(f)
        for file in tqdm(listdir(abspath('./raw/'))):
            if not file.endswith('.txt'):
                continue
            file_hash = run(['b2sum', abspath(f'./raw/{file}')], stdout=PIPE).stdout.decode('utf-8').split(' ')[0]
            if hashes.get(file, '') == file_hash:
                continue
            hashes[file] = file_hash
            with open(abspath(f'./raw/{file}'), 'r') as f:
                text = f.read()
            contents = text.split('\n---\n')
            for i in range(len(contents)):
                contents[i] = contents[i].strip()
            statements = contents[1].split('\n\n')
            for i in range(len(statements)):
                statements[i] = statements[i].strip()
            sources = contents[3].split('\n\n')
            for i in range(len(sources)):
                sources[i] = sources[i].strip()
            statements_extended = statements.copy()
            for i, sentence in enumerate(contents[2].split('. ')):
                statements_extended.insert(i, sentence + '.')
            prompt = prompt_template.replace('{{topic}}', contents[0]).replace('{{statements}}', '\n$\n'.join(statements_extended))
            conversation = system_prompt.copy()
            conversation.append({'role': 'user', 'content': prompt})
            response = llm.generate(conversation, grammar=grammar)
            response_statements = response.split('```')[1].split('\n$\n')
            for i in range(len(response_statements)):
                response_statements[i] = response_statements[i].replace('text\n', '').strip()
            parsed = {
                'topic': contents[0],
                'original_statements': statements,
                'statements': statements + response_statements,
                'fact': contents[2],
                'sources': sources,
                'prompt': prompt,
                'response': response,
            }
            with open(abspath(f'./parsed/{file.replace('.txt', '.json')}'), 'w') as f:
                json_dump(parsed, f, indent=4, ensure_ascii=False)
        with open(hash_db, 'w') as f:
            json_dump(hashes, f, indent=4, ensure_ascii=False)
    finally:
        llm.stop()


if __name__ == '__main__':
    main()
