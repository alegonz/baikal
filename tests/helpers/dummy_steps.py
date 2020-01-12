from sklearn.base import BaseEstimator

from baikal import Step, make_step


class DummySISO(Step):
    """Dummy step that takes a single input and produces a single output.
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def transform(self, X):
        return 2 * X


class DummySIMO(Step):
    """Dummy step that takes a single input and produces multiple outputs.
    """

    def __init__(self, name=None):
        super().__init__(name=name, n_outputs=2)

    def transform(self, X):
        return X + 1.0, X - 1.0


class DummyMISO(Step):
    """Dummy step that takes multiple inputs and produces a single output.
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def transform(self, Xs):
        x1, x2 = Xs
        return x1 + x2

    def fit(self, Xs, ys):
        # Suppose that this dummy model expects two inputs and two targets
        self.fitted_ = True
        return self


class DummyMIMO(Step):
    """Dummy step that takes multiple inputs and produces multiple outputs.
    """

    def __init__(self, name=None):
        super().__init__(name=name, n_outputs=2)

    def transform(self, Xs):
        x1, x2 = Xs
        return x1 * 10.0, x2 / 10.0

    def fit(self, X):
        return self


class DummyImproperlyDefined(Step):
    """Dummy step that returns two outputs but defines only one.
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def transform(self, X):
        return X + 1.0, X - 1.0


class _DummyEstimator(BaseEstimator):
    def __init__(self, x=123, y="abc"):
        self.x = x
        self.y = y
        self.fit_calls = 0
        self.fit_predict_calls = 0

    def predict(self, X):
        return X

    def predict_proba(self, X):
        return X

    def fit(self, X, y):
        self.fit_calls += 1
        return self

    def fit_predict(self, X, y):
        self.fit_predict_calls += 1
        return X

    def fit_predict_proba(self, X, y):
        return X


DummyEstimator = make_step(_DummyEstimator)
