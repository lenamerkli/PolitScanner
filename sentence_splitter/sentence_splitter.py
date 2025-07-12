import sys
sys.path.append('/home/lena/Documents/python/PolitScanner/util')

from time import sleep
from pathlib import Path
from util.llm import LLaMaCPP
import typing as t

INPUT_SIZE = 1024
DTYPE = None
LOAD_IN_4BIT: bool = True
BASE_MODEL_NAME = 'unsloth/Qwen3-0.6B-unsloth-bnb-4bit'
ESCAPES = [('\\', '\\\\'), ('\n', '\\n'), ('\t', '\\t')]
SPECIAL = [c.encode('utf-8') for c in 'äöüÄÖÜéèà'] + [b'\xc2\xab', b'\xc2\xbb']  # noqa


def escape(text: str) -> str:
    for a, b in ESCAPES:
        text = text.replace(a, b)
    return text


def unescape(text: str) -> str:
    for a, b in ESCAPES:
        text = text.replace(b, a)
    return text


def fix_multibyte_chars(text: bytes, indices: list) -> (bytes, list):
    if text.startswith(b'\x80\x93'):
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

def split(text: str, llm: t.Optional[LLaMaCPP] = None) -> list:
    with open(f"{Path(__file__).resolve().parent.absolute()}/prompt.md", 'r', encoding='utf-8') as _f:
        prompt = _f.read()
    text = escape(text)
    user = prompt.replace('{input}', text)
    conversation = f"<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    print(conversation)
    if llm is None:
        llm = LLaMaCPP()
        llm.set_model('sentence_splitter_Q6_K.gguf')
        llm.load_model(print_log=True, seed=42, threads=24, kv_cache_type='q8_0', context=4096)
    while llm.is_loading() or not llm.is_running():
        sleep(1)
    output = llm.generate(conversation)
    print(repr(output))
    output = output.replace('\n```\n```', '\n```')
    output = output.rsplit('```')[-2]
    outputs = output.split('\n')
    return [i.replace('\n', '') for i in outputs if len(i) > 6]
