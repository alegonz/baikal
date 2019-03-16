import pytest
import sklearn.decomposition
import sklearn.linear_model
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from baikal.core.digraph import Node, default_graph
from baikal.core.step import Step


@pytest.fixture
def teardown():
    yield
    Node.clear_names()
    default_graph.clear()


@pytest.fixture
def sklearn_classifier_step():
    class LogisticRegression(Step, sklearn.linear_model.logistic.LogisticRegression):
        def __init__(self, name=None, **kwargs):
            super(LogisticRegression, self).__init__(name=name, **kwargs)

        def build_output_shapes(self, input_shapes):
            return [(1,)]

        def compute(self, x):
            return self.predict(x)

        @property
        def fitted(self):
            try:
                return check_is_fitted(self, ['coef_'], all_or_any=all) is None
            except NotFittedError:
                return False

    return LogisticRegression


@pytest.fixture
def sklearn_transformer_step():
    class PCA(Step, sklearn.decomposition.PCA):
        def __init__(self, name=None, **kwargs):
            super(PCA, self).__init__(name=name, **kwargs)

        def build_output_shapes(self, input_shapes):
            # TODO: How to handle when n_components is determined during fit?
            # This occurs when passed as a percentage of total variance (0 < n_components < 1)
            return [(self.n_components,)]

        def compute(self, x):
            return self.transform(x)

        @property
        def fitted(self):
            try:
                return check_is_fitted(self, ['mean_', 'components_'], all_or_any=all) is None
            except NotFittedError:
                return False

    return PCA
