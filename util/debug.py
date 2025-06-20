

def get_dict_structure(d):
    if isinstance(d, dict):
        return {k: get_dict_structure(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [get_dict_structure(v) for v in d]
    else:
        return type(d)

