import numpy as np
import pytest
from numpy.testing import assert_array_equal

from baikal import Model, Input
from baikal._core.utils import listify, safezip2
from baikal.steps import ColumnStack, Concatenate, Stack, Split

from tests.helpers.fixtures import teardown


def test_columnstack(teardown):
    x1 = Input()
    x2 = Input()
    y = ColumnStack()([x1, x2])
    model = Model([x1, x2], y)

    x1_data = np.array([1, 10, 100])
    x2_data = np.array([2, 20, 200])

    y_expected = np.column_stack([x1_data, x2_data])

    y_pred = model.predict([x1_data, x2_data])

    assert_array_equal(y_pred, y_expected)


def test_concatenate(teardown):
    x1 = Input()
    x2 = Input()
    y = Concatenate(axis=1)([x1, x2])
    model = Model([x1, x2], y)

    x1_data = np.array([[1, 2], [10, 20]])
    x2_data = np.array([[3, 4, 5], [30, 40, 50]])
    y_expected = np.concatenate([x1_data, x2_data], axis=1)
    y_pred = model.predict([x1_data, x2_data])

    assert_array_equal(y_pred, y_expected)


def test_stack(teardown):
    x1 = Input()
    x2 = Input()
    y = Stack(axis=1)([x1, x2])
    model = Model([x1, x2], y)

    x1_data = np.array([[1, 2], [10, 20]])
    x2_data = np.array([[3, 4], [30, 40]])
    y_expected = np.stack([x1_data, x2_data], axis=1)
    y_pred = model.predict([x1_data, x2_data])

    assert_array_equal(y_pred, y_expected)


@pytest.mark.parametrize(
    "x,indices_or_sections", [(np.array([1, 2, 3]), 3), (np.array([1, 2, 3]), [1]),]
)
def test_split(x, indices_or_sections, teardown):
    x1 = Input()
    ys = Split(indices_or_sections, axis=0)(x1)
    model = Model(x1, ys)

    y_expected = np.split(x, indices_or_sections, axis=0)
    y_pred = model.predict(x)
    y_pred = listify(y_pred)

    for actual, expected in safezip2(y_pred, y_expected):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("step_class", [ColumnStack, Concatenate, Stack])
def test_single_input(step_class, teardown):
    x = Input()
    y = step_class()(x)
    model = Model(x, y)

    x_data = np.array([[1, 2], [3, 4]])
    if step_class is Stack:
        assert_array_equal(x_data.reshape((2, 2, 1)), model.predict(x_data))
    else:
        assert_array_equal(x_data, model.predict(x_data))
