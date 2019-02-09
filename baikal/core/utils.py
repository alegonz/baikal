def make_name(*parts, sep='/'):
    return sep.join([str(p) for p in parts])


def listify(x):
    if not isinstance(x, list):
        x = [x]
    return x
