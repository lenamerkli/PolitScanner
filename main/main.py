import sys
sys.path.append('/home/lena/Documents/python/PolitScanner/util')
sys.path.append('/home/lena/Documents/python/PolitScanner/sentence_splitter')
sys.path.append('/home/lena/Documents/python/PolitScanner/data')

import chromadb
from pathlib import Path
from util.llm import LLaMaCPP
from os.path import exists
from json import load as json_load
from time import sleep
from sentence_splitter import split  # noqa


MAX_DIFFERENCE = 1.3
MAX_DB_RESULTS = 10
with open('prompt.md', 'r', encoding='utf-8') as _f:
    PROMPT = _f.read()
GBNF_TEMPLATE = """
root ::= "```python\\n[" list "]\\n```"
list ::= %%
"""
GBNF_TEMPLATE_ITEM = '("\'%%\'")?'
GBNF_SEPARATOR = ' (", ")? '


def db_read(texts: list[str]):
    client = chromadb.PersistentClient(path=Path(__file__).resolve().parent.parent.absolute().__str__() + '/data/database.chroma')
    collection = client.get_collection(name='PolitScanner')
    return collection.query(query_texts=texts, n_results=MAX_DB_RESULTS)


def process(sentences: list, llm: LLaMaCPP) -> list:
    db_results = db_read(sentences)
    print(db_results)
    if len(db_results['ids'][0]) == 0:
        return []
    topic_ids = []
    for i, result in enumerate(db_results['ids'][0]):
        if db_results['distances'][0][i] < MAX_DIFFERENCE:
            id_ = result.split('-')[0]
            if id_ not in topic_ids:
                topic_ids.append(id_)
    if len(topic_ids) == 0:
        return []
    if len(topic_ids) == 1 and topic_ids[0] != '0':
        topic_ids.append('0')
    topics = []
    titles = {}
    for topic_id in topic_ids:
        with open(Path(__file__).resolve().parent.parent.absolute().__str__() + f"/data/parsed/{topic_id}.json", 'r') as f:
            topics.append(json_load(f))
            titles[topics[-1]['topic']] = len(topics) - 1
    formatted_topics = ''
    titles_list = list(titles.keys())
    titles_list.sort()
    items = []
    for title in titles_list:
        items.append(GBNF_TEMPLATE_ITEM.replace('%%', title))
    grammar = GBNF_TEMPLATE.replace('%%', GBNF_SEPARATOR.join(items))
    topics.sort(key=lambda x: x['topic'])
    for topic in topics:
        if len(formatted_topics) > 0:
            formatted_topics += '\n'
        # formatted_topics += f"'{topic['topic']}' - Beispielsätze:\n{'\n'.join('- ' + statement for i, statement in enumerate(topic['original_statements']) if i < 3)}"
        formatted_topics += f"'{topic['topic']}'"
    prompt = PROMPT.replace('{TOPICS}', formatted_topics)
    for i, sentence in enumerate(sentences):
        prompt = prompt.replace('{' + f'SENTENCE_{i+1}' + '}', sentence)
    prompt = f"<|im_start|>user\n{prompt}\n/no_think\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    print(prompt)
    output = llm.generate(prompt, enable_thinking=False, grammar=grammar, temperature=0.0)
    print(output)
    output = output.split('[')[-1].split(']')[0]
    truths = []
    for title in titles_list:
        if title in output:
            truths.append(topics[titles[title]]['fact'])
    return truths


def main() -> None:
    if not exists('input.txt'):
        raise FileNotFoundError('input.txt not found')
    with open('input.txt', 'r') as f:
        text = f.read()
    llm = LLaMaCPP()
    if exists('/opt/llms/Qwen3-1.7B-PolitScanner-Q5_K_S.gguf'):
        llm.set_model('Qwen3-1.7B-PolitScanner-Q5_K_S.gguf')
    else:
        llm.set_model('Qwen3-30B-A3B-Q5_K_M.gguf')
    sentences = split(text)
    print(f"{len(sentences)=}")
    chunked_sentences = []
    for i in range(0, len(sentences), 3):
        if i == 0:
            chunk2 = ['EMPTY'] + sentences[:4]
        elif i + 3 >= len(sentences):
            chunk2 = sentences[-5:-1] + ['EMPTY']
        else:
            chunk2 = sentences[i - 1:i + 4]
        chunked_sentences.append(chunk2)
    print(f"{len(chunked_sentences)=}")
    llm.load_model(print_log=True, threads=16, kv_cache_type='q8_0', context=8192)
    while llm.is_loading() or not llm.is_running():
        sleep(1)
    with open('output.txt', 'w', encoding='utf-8') as f:
        for chunked_sentences2 in chunked_sentences:
            truths = process(chunked_sentences2, llm)
            for truth in truths:
                f.write(f"  # Hinweis: {truth}\n")
            for i, sentence in enumerate(chunked_sentences2):
                if i in range(1, 4):
                    f.write(f"{sentence}\n")
            f.write('\n')
    print('REACHED `llm.stop()`')
    llm.stop()


if __name__ == '__main__':
    main()
