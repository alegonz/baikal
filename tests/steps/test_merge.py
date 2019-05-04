import numpy as np
from numpy.testing import assert_array_equal

from baikal.core.model import Model
from baikal.core.step import Input
from baikal.steps import Concatenate, Stack

from tests.helpers.fixtures import teardown


def test_concatenate(teardown):
    x1 = Input()
    x2 = Input()
    y = Concatenate(axis=1)([x1, x2])
    model = Model([x1, x2], y)

    x1_data = np.array([[1, 2],
                        [10, 20]])

    x2_data = np.array([[3, 4, 5],
                        [30, 40, 50]])

    y_expected = np.concatenate([x1_data, x2_data], axis=1)

    y_pred = model.predict([x1_data, x2_data])

    assert_array_equal(y_pred, y_expected)


def test_stack(teardown):
    x1 = Input()
    x2 = Input()
    y = Stack(axis=1)([x1, x2])
    model = Model([x1, x2], y)

    x1_data = np.array([[1, 2],
                        [10, 20]])

    x2_data = np.array([[3, 4],
                        [30, 40]])

    y_expected = np.stack([x1_data, x2_data], axis=1)

    y_pred = model.predict([x1_data, x2_data])

    assert_array_equal(y_pred, y_expected)
