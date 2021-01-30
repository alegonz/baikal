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


LinearRegression = make_step(
    sklearn.linear_model.LinearRegression, class_name="LinearRegression"
)
LogisticRegression = make_step(
    sklearn.linear_model.LogisticRegression, class_name="LogisticRegression"
)
LinearSVC = make_step(sklearn.svm.LinearSVC, class_name="LinearSVC")
LinearSVCOOF = make_step(
    sklearn.svm.LinearSVC,
    attr_dict={"fit_predict": _fit_decision_function},
    class_name="LinearSVCOOF",
)
RandomForestClassifier = make_step(
    sklearn.ensemble.RandomForestClassifier, class_name="RandomForestClassifier"
)
RandomForestClassifierOOF = make_step(
    sklearn.ensemble.RandomForestClassifier,
    attr_dict={"fit_predict": _fit_predict_proba},
    class_name="RandomForestClassifierOOF",
)
ExtraTreesClassifier = make_step(
    sklearn.ensemble.ExtraTreesClassifier, class_name="ExtraTreesClassifier"
)
PCA = make_step(sklearn.decomposition.PCA, class_name="PCA")
LabelEncoder = make_step(sklearn.preprocessing.LabelEncoder, class_name="LabelEncoder")
StandardScaler = make_step(
    sklearn.preprocessing.StandardScaler, class_name="StandardScaler"
)
