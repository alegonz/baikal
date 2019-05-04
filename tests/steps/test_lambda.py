import numpy as np
from numpy.testing import assert_array_equal

from baikal.core.model import Model
from baikal.core.step import Input
from baikal.steps._lambda import Lambda

from tests.helpers.fixtures import teardown


def test_lambda(teardown):
    def function(x1, x2):
        return 2 * x1, x2 / 2

    x = Input()
    y1, y2 = Lambda(function, n_outputs=2)([x, x])
    model = Model(x, [y1, y2])

    x_data = np.array([[1.0, 2.0],
                       [3.0, 4.0]])

    y1_expected = np.array([[2.0, 4.0],
                            [6.0, 8.0]])
    y2_expected = np.array([[0.5, 1.0],
                            [1.5, 2.0]])

    y1_pred, y2_pred = model.predict(x_data)

    assert_array_equal(y1_pred, y1_expected)
    assert_array_equal(y2_pred, y2_expected)
