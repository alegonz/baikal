def listify(x):
    if isinstance(x, list):
        pass
    elif isinstance(x, tuple):
        x = list(x)
    else:
        x = [x]
    return x


def make_name(*parts, sep='/'):
    return sep.join([str(p) for p in parts])


def make_repr(obj, attrs):
    args = ', '.join(['{}={}'.format(attr, str(getattr(obj, attr))) for attr in attrs])
    return '{}({})'.format(obj.__class__.__name__, args)
