def make_name(*parts, sep='/'):
    return sep.join([str(p) for p in parts])


def listify(x):
    if isinstance(x, list):
        pass
    elif isinstance(x, tuple):
        x = list(x)
    else:
        x = [x]
    return x
