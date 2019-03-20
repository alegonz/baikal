def datatuple2list(datatuple, unpack_singleton=False):
    ls = [array for _, array in datatuple]
    if unpack_singleton and len(ls) == 1:
        ls = ls[0]
    return ls


def datatuple2datadict(datatuple):
    return {data: value for data, value in datatuple}


def datatuple2strdict(datatuple):
    return {data.name: value for data, value in datatuple}
