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
