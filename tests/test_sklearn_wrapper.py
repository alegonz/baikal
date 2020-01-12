import pytest
import sklearn.decomposition
import sklearn.linear_model
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from baikal import Model, Input
from baikal.sklearn import SKLearnWrapper

from tests.helpers.sklearn_steps import PCA, LogisticRegression, RandomForestClassifier


iris = datasets.load_iris()
x_data = iris.data
y_t_data = iris.target
random_state = 123
verbose = 0
cv = StratifiedKFold(3)  # cv will default to KFold if the estimator is a baikal Model


def test_grid_search_cv():
    param_grid = {
        "pca__n_components": [2, 4],
        "logreg__C": [0.1, 1.0, 10],
        "logreg__penalty": ["l1", "l2"],
    }

    # baikal way
    def build_fn():
        x = Input()
        y_t = Input()
        h = PCA(random_state=random_state, name="pca")(x)
        y = LogisticRegression(
            random_state=random_state, solver="liblinear", name="logreg"
        )(h, y_t)
        model = Model(x, y, y_t)
        return model

    sk_model = SKLearnWrapper(build_fn)
    assert isinstance(sk_model.model, Model)

    gscv_baikal = GridSearchCV(
        sk_model,
        param_grid,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        verbose=verbose,
    )
    gscv_baikal.fit(x_data, y_t_data)

    # traditional way
    pca = PCA(random_state=random_state)
    logreg = LogisticRegression(random_state=random_state, solver="liblinear")
    pipe = Pipeline([("pca", pca), ("logreg", logreg)])

    gscv_traditional = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        verbose=verbose,
    )
    gscv_traditional.fit(x_data, y_t_data)

    assert gscv_baikal.best_params_ == gscv_traditional.best_params_
    assert_array_equal(
        gscv_traditional.cv_results_["mean_train_score"],
        gscv_baikal.cv_results_["mean_train_score"],
    )
    assert_array_equal(
        gscv_traditional.cv_results_["mean_test_score"],
        gscv_baikal.cv_results_["mean_test_score"],
    )


def test_grid_search_cv_with_tunable_step():
    param_grid = {
        "classifier": [
            LogisticRegression(random_state=random_state),
            RandomForestClassifier(random_state=random_state),
        ],
        "pca__n_components": [2, 4],
    }

    # baikal way
    def build_fn():
        x = Input()
        y_t = Input()
        h = PCA(random_state=random_state, name="pca")(x)
        y = LogisticRegression(random_state=random_state, name="classifier")(h, y_t)
        model = Model(x, y, y_t)
        return model

    sk_model = SKLearnWrapper(build_fn)
    gscv_baikal = GridSearchCV(
        sk_model,
        param_grid,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        verbose=verbose,
    )
    gscv_baikal.fit(x_data, y_t_data)

    # traditional way
    pca = PCA(random_state=random_state)
    classifier = LogisticRegression(random_state=random_state)
    pipe = Pipeline([("pca", pca), ("classifier", classifier)])

    gscv_traditional = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        verbose=verbose,
    )
    gscv_traditional.fit(x_data, y_t_data)

    assert gscv_baikal.best_params_ == gscv_traditional.best_params_
    assert_array_equal(
        gscv_traditional.cv_results_["mean_train_score"],
        gscv_baikal.cv_results_["mean_train_score"],
    )
    assert_array_equal(
        gscv_traditional.cv_results_["mean_test_score"],
        gscv_baikal.cv_results_["mean_test_score"],
    )
