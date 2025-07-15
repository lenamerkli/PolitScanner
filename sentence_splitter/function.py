def split_list(array: list[str], separator: str) -> list[str]:
    r = []
    for s in array:
        r.extend(s.split(separator))
    return r


def split(text: str) -> list[str]:
    r = [text]
    for value in ['\n', '. ', '? ']:
        r = split_list(r, value)
    return r
