from contextlib import contextmanager

import pytest

from baikal._core.utils import (
    listify,
    unlistify,
    safezip2,
    find_duplicated_items,
    SimpleCache,
)


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize("x,expected", [(1, [1]), ((1,), [1]), ([1], [1])])
def test_listify(x, expected):
    assert listify(x) == expected


@pytest.mark.parametrize(
    "x,expected,raises",
    [
        ([1], 1, does_not_raise()),
        ((1,), None, pytest.raises(ValueError)),
        ([1, 2], [1, 2], does_not_raise()),
    ],
)
def test_unlistify(x, expected, raises):
    with raises:
        assert unlistify(x) == expected


@pytest.mark.parametrize(
    "x,y,raises",
    [
        ((1, 2), (1, 2), does_not_raise()),
        ((1,), (1, 2), pytest.raises(ValueError)),
        ((1, 2), (1,), pytest.raises(ValueError)),
    ],
)
def test_safezip2(x, y, raises):
    with raises:
        z = list(safezip2(x, y))
        assert z == list(zip(x, y))


@pytest.mark.parametrize(
    "x,expected", [((1, 1, 2, 2), [1, 2]), ((1, 2, 3), []), ([], [])]
)
def test_find_duplicated_items(x, expected):
    assert find_duplicated_items(x) == expected


def test_simple_cache():
    cache = SimpleCache()
    assert cache.hits == 0 and cache.misses == 0
    assert "a" not in cache
    assert cache.hits == 0 and cache.misses == 1
    cache["a"] = 1
    assert cache.hits == 0 and cache.misses == 1
    assert "a" in cache
    assert cache.hits == 1 and cache.misses == 1
    with pytest.raises(KeyError):
        cache["b"]
