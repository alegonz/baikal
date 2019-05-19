import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from baikal import Step


class LogisticRegression(Step, sklearn.linear_model.LogisticRegression):
    def __init__(self, name=None, **kwargs):
        super(LogisticRegression, self).__init__(name=name, **kwargs)
        self.n_outputs = 1

    @property
    def fitted(self):
        try:
            return check_is_fitted(self, ['coef_'], all_or_any=all) is None
        except NotFittedError:
            return False


class RandomForestClassifier(Step, sklearn.ensemble.RandomForestClassifier):
    def __init__(self, name=None, **kwargs):
        super(RandomForestClassifier, self).__init__(name=name, **kwargs)
        self.n_outputs = 1

    @property
    def fitted(self):
        try:
            return check_is_fitted(self, ['estimators_'], all_or_any=all) is None
        except NotFittedError:
            return False


class ExtraTreesClassifier(Step, sklearn.ensemble.ExtraTreesClassifier):
    def __init__(self, name=None, **kwargs):
        super(ExtraTreesClassifier, self).__init__(name=name, **kwargs)
        self.n_outputs = 1

    @property
    def fitted(self):
        try:
            return check_is_fitted(self, ['estimators_'], all_or_any=all) is None
        except NotFittedError:
            return False


class PCA(Step, sklearn.decomposition.PCA):
    def __init__(self, name=None, **kwargs):
        super(PCA, self).__init__(name=name, **kwargs)
        self.n_outputs = 1

    @property
    def fitted(self):
        try:
            return check_is_fitted(self, ['mean_', 'components_'], all_or_any=all) is None
        except NotFittedError:
            return False


class StandardScaler(Step, sklearn.preprocessing.StandardScaler):
    def __init__(self, name=None, **kwargs):
        super(StandardScaler, self).__init__(name=name, **kwargs)
        self.n_outputs = 1

    @property
    def fitted(self):
        try:
            return check_is_fitted(self, 'scale_') is None
        except NotFittedError:
            return False
