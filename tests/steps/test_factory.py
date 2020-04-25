import sklearn.linear_model

from baikal import make_step, Step


def test_make_step():
    def some_method(self):
        pass

    LogisticRegression = make_step(
        sklearn.linear_model.LogisticRegression, attr_dict={"some_method": some_method}
    )

    assert issubclass(LogisticRegression, Step)
    assert issubclass(LogisticRegression, sklearn.linear_model.LogisticRegression)
    assert hasattr(LogisticRegression, "get_params")
    assert hasattr(LogisticRegression, "set_params")
    assert hasattr(LogisticRegression, "fit")
    assert hasattr(LogisticRegression, "predict")
    assert hasattr(LogisticRegression, "some_method")
    assert LogisticRegression.__name__ == "LogisticRegression"
