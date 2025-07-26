import sys
sys.path.append('/home/lena/Documents/python/PolitScanner/util')
sys.path.append('/home/lena/Documents/python/PolitScanner/sentence_splitter')
from data import DATA  # noqa
from util.llm import LLaMaCPP
from random import randint
from time import sleep
from pathlib import Path

with open(Path(__file__).resolve().parent.absolute().__str__() + '/prompt.md', 'r', encoding='utf-8') as _f:
    PROMPT = _f.read()
SPECIAL = [c.encode('utf-8') for c in 'äöüÄÖÜéèà'] + [b'\xc2\xab', b'\xc2\xbb']  # noqa


def split(text: str) -> list:
    from function import split as split_func
    return split_func(text)


def fix_multibyte_chars(text: bytes, indices: list) -> (bytes, list):
    if text.startswith(b'\x80\x93'):  # noqa
        text = b'\xe2' + text
        for i in range(len(indices)):
            indices[i] += 1
    elif text.startswith(b'\x93'):
        text = b'\xe2\x80' + text
        for i in range(len(indices)):
            indices[i] += 2
    if text.endswith(b'\xe2\x80'):
        text += b'\x93'
    elif text.endswith(b'\xe2'):
        text += b'\x80\x93'
    first = text[0]
    for c in SPECIAL:
        if first == c[1]:
            text = bytes([c[0]]) + text
            for i in range(len(indices)):
                indices[i] += 1
            break
    last = text[-1]
    for c in SPECIAL:
        if last == c[0]:
            text += bytes([c[1]])
            break
    return text, indices

def run_ai(llm: LLaMaCPP, error: Exception, string: str, sentences: list, sentences_ai: list) -> None:
    with open('function.py', 'r', encoding='utf-8') as f:
        function = f.read()
    string = repr(string)
    sentences_ = [repr(s) for s in sentences]
    sentences_ai_ = [repr(s) for s in sentences_ai]
    prompt = PROMPT.replace('{PROGRAM}', function)
    prompt = prompt.replace('{ERROR}', repr(error))
    prompt = prompt.replace('{STRING}', string)
    prompt = prompt.replace('{SENTENCES}', f"[{', '.join(sentences_)}]")
    prompt = prompt.replace('{SENTENCES_AI}', f"[{', '.join(sentences_ai_)}]")
    conversation = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    print(conversation)
    output = llm.generate(conversation)
    print(output)
    output = output.replace('\n```\n```', '\n```')
    output = output.rsplit('```')[-2]
    with open('function.py', 'w', encoding='utf-8') as f:
        f.write(output)


def train() -> None:
    # dataset = SentenceSplitterDataset(train=True, transform=None, min_length=128, max_length=256, output_size=0,
    #                                   disable_pytorch=True)
    llm = LLaMaCPP()
    # llm.set_model('Qwen3-30B-A3B-Q5_K_M.gguf')
    # llm.set_model('Qwen3-8B-Q5_K_M.gguf')
    llm.set_model('Qwen3-32B-Q4_K_S.gguf')
    llm.load_model(print_log=True, seed=42, threads=16, kv_cache_type='q8_0', context=16384)
    while llm.is_loading() or not llm.is_running():
        sleep(1)
    for element in DATA:
        try:
            sentences = element['sentences']
            string = element['string']
            sentences_ai = []
            try:
                from function import split
                sentences_ai = split(text=string)
                assert sentences == sentences_ai
            except Exception as e:
                e.add_note(f"Error with datapoint ```{string}```")
                run_ai(llm, e, string, sentences, sentences_ai)
            finally:
                del split
        except KeyboardInterrupt:
            break
    llm.stop()


if __name__ == '__main__':
    train()
