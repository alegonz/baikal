from typing import Union, Any, List, Tuple


def listify(x: Union[Any, List[Any], Tuple[Any, ...]]) -> List[Any]:
    if isinstance(x, list):
        pass
    elif isinstance(x, tuple):
        x = list(x)
    else:
        x = [x]
    return x


def unlistify(x: List[Any]) -> Union[List[Any], Any]:
    if not isinstance(x, list):
        raise ValueError("x must be a list.")
    if len(x) == 1:
        return x[0]
    return x


def safezip2(seq1, seq2):
    """A zip that raises an error when the sequences have different length.
    It can only handle two sequences.
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            "Lengths of iterators differ: {} != {}.".format(len(seq1), len(seq2))
        )
    return zip(seq1, seq2)


def make_name(*parts, sep="/"):
    return sep.join([str(p) for p in parts])


def make_args_from_attrs(obj, attrs):
    args = []
    for attr in attrs:
        attr_value = getattr(obj, attr)
        arg = repr(attr_value)
        args.append("{}={}".format(attr, arg))
    return ", ".join(args)


def make_repr(obj, attrs):
    args = make_args_from_attrs(obj, attrs)
    return "{}({})".format(obj.__class__.__name__, args)


def find_duplicated_items(items):
    seen_items = {}
    for item in items:
        if item not in seen_items:
            seen_items[item] = 1
        else:
            seen_items[item] += 1
    duplicated_items = [item for item, count in seen_items.items() if count > 1]
    return duplicated_items


class SimpleCache:
    """A simple cache that updates its stats upon checking (not retrieval)
    """

    def __init__(self):
        self._hits = 0
        self._misses = 0
        self._cache = {}

    def __contains__(self, key):
        if key in self._cache:
            self._hits += 1
            return True
        self._misses += 1
        return False

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self._cache[key] = value

    @property
    def hits(self):
        return self._hits

    @property
    def misses(self):
        return self._misses
