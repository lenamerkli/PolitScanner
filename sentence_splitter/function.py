def split_list(array: list[str], separator: str) -> list[str]:
    r = []
    placeholder = "\uE000"
    for s in array:
        s_with_marker = s.replace(separator, separator + placeholder)
        parts = s_with_marker.split(placeholder)
        r.extend(parts)
    return r


def split(text: str) -> list[str]:
    for replacement in [' \n', '\n ', '\n\n']:
        while replacement in text:
            text = text.replace(replacement, '\n')
    protections = ['d. h.', 'Abs.', 'Art.', 'Bem.', 'Bst.', ' ff.', ' f.', '(ff.', '(f.', 'insbes.', 'S.', 'V.']
    for protection in protections:
        text = text.replace(protection, protection.replace('.', '\uE000'))
    placeholder = "\uE001"
    for i in range(3, len(text) - 3):
        if text[i] == '.':
            if (
                (text[i - 2] == ' ') or
                ( not text[i + 2].isupper()) or
                (text[i - 1].isdigit())
            ):
                text = text[:i] + placeholder + text[i+1:]
    array = [text]
    for value in ['\n', '. ', '? ']:
        array = split_list(array, value)
    final_list = []
    for s in array:
        cleaned_s = s.replace(placeholder, '.').strip()
        final_list.append(cleaned_s)
    return final_list
