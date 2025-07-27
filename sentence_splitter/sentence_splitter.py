import sys
sys.path.append('/home/lena/Documents/python/PolitScanner/util')
sys.path.append('/home/lena/Documents/python/PolitScanner/sentence_splitter')
from data import DATA  # noqa
from util.llm import LLaMaCPP
from time import sleep
from pathlib import Path

with open(Path(__file__).resolve().parent.absolute().__str__() + '/prompt.md', 'r', encoding='utf-8') as _f:
    PROMPT = _f.read()
SPECIAL = [c.encode('utf-8') for c in 'äöüÄÖÜéèà'] + [b'\xc2\xab', b'\xc2\xbb']  # noqa


def split(text: str) -> list:
    """
    Splits a text into sentences using the generated function
    :param text: The text to split
    :return: A list of sentences
    """
    from function import split as split_func
    return split_func(text)


def fix_multibyte_chars(text: bytes, indices: list) -> (bytes, list):
    """
    Fixes UTF-8 encoded multibyte characters that might have been split incorrectly at chunk boundaries.
    
    It handles several cases:
    1. Fixes incomplete em-dash characters (U+2014, encoded as \xe2\x80\x93) at the beginning of text
    2. Completes incomplete em-dash characters at the end of text
    3. Handles special characters like german umlauts, accented characters, and quotation marks
    4. Adjusts the indices list to account for any added bytes
    
    :param text: The byte string that may contain incomplete multibyte characters
    :param indices: A list of indices into the byte string that need to be adjusted if bytes are added
    :return: A tuple containing the fixed byte string and the adjusted indices list
    """
    # Fix incomplete em-dash characters at the beginning
    if text.startswith(b'\x80\x93'):
        text = b'\xe2' + text
        for i in range(len(indices)):
            indices[i] += 1
    elif text.startswith(b'\x93'):
        text = b'\xe2\x80' + text
        for i in range(len(indices)):
            indices[i] += 2
    # Fix incomplete em-dash characters at the end
    if text.endswith(b'\xe2\x80'):
        text += b'\x93'
    elif text.endswith(b'\xe2'):
        text += b'\x80\x93'
    first = text[0]
    # Handle special characters like german umlauts at the beginning
    for c in SPECIAL:
        if first == c[1]:
            text = bytes([c[0]]) + text
            for i in range(len(indices)):
                indices[i] += 1
            break
    # Handle special characters at the end
    last = text[-1]
    for c in SPECIAL:
        if last == c[0]:
            text += bytes([c[1]])
            break
    return text, indices

def run_ai(llm: LLaMaCPP, error: Exception, string: str, sentences: list, sentences_ai: list) -> None:
    """
    Use an AI language model to fix the sentence splitting function when it fails to correctly process text.
    
    :param llm: The LLaMaCPP language model instance to use for generating the improved function
    :param error: The exception that was raised during sentence splitting
    :param string: The original text string that caused the error
    :param sentences: The expected correct sentence splitting result (ground truth)
    :param sentences_ai: The incorrect sentence splitting result produced by the current implementation
    :return: None
    """
    # Read the current implementation
    with open('function.py', 'r', encoding='utf-8') as f:
        function = f.read()
    string = repr(string)
    sentences_ = [repr(s) for s in sentences]
    sentences_ai_ = [repr(s) for s in sentences_ai]
    # Construct the prompt
    prompt = PROMPT.replace('{PROGRAM}', function)
    prompt = prompt.replace('{ERROR}', repr(error))
    prompt = prompt.replace('{STRING}', string)
    prompt = prompt.replace('{SENTENCES}', f"[{', '.join(sentences_)}]")
    prompt = prompt.replace('{SENTENCES_AI}', f"[{', '.join(sentences_ai_)}]")
    # Use a simplified conversation template for Qwen3
    conversation = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    print(conversation)
    output = llm.generate(conversation)
    print(output)
    # Extract the function
    output = output.replace('\n```\n```', '\n```')
    output = output.rsplit('```')[-2]
    # Write to disk
    with open('function.py', 'w', encoding='utf-8') as f:
        f.write(output)


def train() -> None:
    """
    Iteratively improve the sentence splitting function using an AI language model. Use `ctrl+c` to stop the training.
    :return: None.
    """
    llm = LLaMaCPP()
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
