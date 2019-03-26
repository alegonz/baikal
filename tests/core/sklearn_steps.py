import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from baikal.core.step import Step


class LogisticRegression(Step, sklearn.linear_model.LogisticRegression):
    def __init__(self, name=None, **kwargs):
        super(LogisticRegression, self).__init__(name=name, **kwargs)

    def build_output_shapes(self, input_shapes):
        return [(1,)]

    @property
    def fitted(self):
        try:
            return check_is_fitted(self, ['coef_'], all_or_any=all) is None
        except NotFittedError:
            return False


class RandomForestClassifier(Step, sklearn.ensemble.RandomForestClassifier):
    def __init__(self, name=None, **kwargs):
        super(RandomForestClassifier, self).__init__(name=name, **kwargs)

    def build_output_shapes(self, input_shapes):
        # TODO: How to handle when n_outputs is determined during fit?
        return [(1,)]

    @property
    def fitted(self):
        try:
            return check_is_fitted(self, ['estimators_'], all_or_any=all) is None
        except NotFittedError:
            return False


class PCA(Step, sklearn.decomposition.PCA):
    def __init__(self, name=None, **kwargs):
        super(PCA, self).__init__(name=name, **kwargs)

    def build_output_shapes(self, input_shapes):
        # TODO: How to handle when n_components is determined during fit?
        # This occurs when passed as a percentage of total variance (0 < n_components < 1)
        return [(self.n_components,)]

    @property
    def fitted(self):
        try:
            return check_is_fitted(self, ['mean_', 'components_'], all_or_any=all) is None
        except NotFittedError:
            return False
