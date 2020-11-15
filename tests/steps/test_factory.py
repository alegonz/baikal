from contextlib import contextmanager

import pytest
import sklearn.linear_model

from baikal import make_step, Step


@contextmanager
def does_not_warn():
    yield


@pytest.mark.parametrize(
    "class_name,expected,warns",
    [
        (None, "LogisticRegression", pytest.warns(FutureWarning)),
        ("LogisticRegressionStep", "LogisticRegressionStep", does_not_warn()),
    ],
)
def test_make_step(class_name, expected, warns):
    def some_method(self):
        pass

    with warns:
        LogisticRegression = make_step(
            sklearn.linear_model.LogisticRegression,
            {"some_method": some_method},
            class_name,
        )

    assert issubclass(LogisticRegression, Step)
    assert issubclass(LogisticRegression, sklearn.linear_model.LogisticRegression)
    assert hasattr(LogisticRegression, "get_params")
    assert hasattr(LogisticRegression, "set_params")
    assert hasattr(LogisticRegression, "fit")
    assert hasattr(LogisticRegression, "predict")
    assert hasattr(LogisticRegression, "some_method")
    assert LogisticRegression.__name__ == expected
