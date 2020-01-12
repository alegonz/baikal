import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.model_selection import cross_val_predict

from baikal import make_step


def _fit_predict_proba(self, X, y):
    self.fit(X, y)
    return cross_val_predict(self, X, y, method="predict_proba")


def _fit_decision_function(self, X, y):
    self.fit(X, y)
    return cross_val_predict(self, X, y, method="decision_function")


LinearRegression = make_step(sklearn.linear_model.LinearRegression)
LogisticRegression = make_step(sklearn.linear_model.LogisticRegression)
LinearSVC = make_step(sklearn.svm.LinearSVC)
LinearSVCOOF = make_step(
    sklearn.svm.LinearSVC, attr_dict={"fit_predict": _fit_decision_function}
)
RandomForestClassifier = make_step(sklearn.ensemble.RandomForestClassifier)
RandomForestClassifierOOF = make_step(
    sklearn.ensemble.RandomForestClassifier,
    attr_dict={"fit_predict": _fit_predict_proba},
)
ExtraTreesClassifier = make_step(sklearn.ensemble.ExtraTreesClassifier)
PCA = make_step(sklearn.decomposition.PCA)
LabelEncoder = make_step(sklearn.preprocessing.LabelEncoder)
StandardScaler = make_step(sklearn.preprocessing.StandardScaler)
