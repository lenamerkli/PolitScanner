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
    """
    Process raw text files and increase the quantity of the data using a Large Language Model.
    
    :return: None
    """
    llm = LLaMaCPP()
    # Any large enough model should work
    llm.set_model('Qwen3-30B-A3B-Q5_K_M.gguf')
    llm.load_model(print_log=False, seed=42, threads=24, kv_cache_type='q8_0', context=2048)
    try:
        # Wait for the model to finish loading
        while llm.is_loading() or not llm.is_running():
            sleep(1)
        sleep(2)
        # Get the system prompt and the prompt template
        system_prompt = llm.get_system_message()
        with open(abspath('./prompt.md'), 'r') as f:
            prompt_template = f.read()
        # Load the hash table
        hash_db = abspath('./parsed/hashes.json')
        if not exists(hash_db):
            with open(hash_db, 'w') as f:
                json_dump({}, f)
        with open(hash_db, 'r') as f:
            hashes: t.Dict[str, str] = json_load(f)
        for file in tqdm(listdir(abspath('./raw/'))):
            if not file.endswith('.txt'):
                continue
            # Check if the file has already been processed
            file_hash = run(['b2sum', abspath(f'./raw/{file}')], stdout=PIPE).stdout.decode('utf-8').split(' ')[0]
            if hashes.get(file, '') == file_hash:
                continue
            hashes[file] = file_hash
            # Get the contents
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
            # Prepare the prompt
            prompt = prompt_template.replace('{{topic}}', contents[0]).replace('{{statements}}', '\n$\n'.join(statements_extended))
            conversation = system_prompt.copy()
            conversation.append({'role': 'user', 'content': prompt})
            response = llm.generate(conversation, enable_thinking=False)
            # Parse the response
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
            # Write to disk
            with open(abspath(f'./parsed/{file.replace('.txt', '.json')}'), 'w') as f:
                json_dump(parsed, f, indent=4, ensure_ascii=False)
        with open(hash_db, 'w') as f:
            json_dump(hashes, f, indent=4, ensure_ascii=False)
    except KeyboardInterrupt:
        pass
    finally:
        llm.stop()


if __name__ == '__main__':
    main()
