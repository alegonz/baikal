import pytest
import sklearn.linear_model

from baikal import make_step, Step


@pytest.mark.parametrize(
    "class_name,expected",
    [
        (None, "LogisticRegression"),
        ("LogisticRegressionStep", "LogisticRegressionStep"),
    ],
)
def test_make_step(class_name, expected):
    def some_method(self):
        pass

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
    assert hasattr(LogisticRegression, "_nodes")
    assert LogisticRegression.__name__ == expected
