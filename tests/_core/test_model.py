import pickle
import tempfile
from contextlib import contextmanager
from typing import List, Dict

import joblib
import numpy as np
import pytest
import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
from numpy.testing import assert_array_equal, assert_allclose
from sklearn import datasets
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from baikal import Model, Input
from baikal._core.data_placeholder import DataPlaceholder
from baikal._core.typing import ArrayLike
from baikal.steps import Concatenate, Stack, Lambda

from tests.helpers.fixtures import teardown
from tests.helpers.sklearn_steps import (
    LinearRegression,
    LogisticRegression,
    RandomForestClassifier,
    ExtraTreesClassifier,
    PCA,
    LabelEncoder,
    StandardScaler,
)
from tests.helpers.dummy_steps import (
    DummySISO,
    DummySIMO,
    DummyMISO,
    DummyImproperlyDefined,
    DummyEstimator,
)


iris = datasets.load_iris()


@contextmanager
def does_not_raise():
    yield


skip_sklearn_0_22 = pytest.mark.skipif(
    sklearn.__version__ == "0.22",
    reason="sklearn.utils.validation.check_is_fitted in 0.22 yields false positives "
    "when the class has private attributes."
    "see: https://github.com/scikit-learn/scikit-learn/issues/15845",
)


class TestInit:
    def test_simple(self, teardown):
        x1 = Input()
        x2 = Input()
        y_t = Input()

        x1_transformed = PCA()(x1)
        y_t_encoded = LabelEncoder()(y_t)
        z = Concatenate()([x1_transformed, x2])
        y = LogisticRegression()(z, y_t_encoded)
        # TODO: support shareable steps to reuse LabelEncoder(function="inverse_transform")

        # full model
        Model([x1, x2], y, y_t)

        # submodels
        Model(x1, x1_transformed)
        Model(z, y, y_t_encoded)

    def test_with_wrong_type(self, teardown):
        x = Input()
        y_t = Input()
        y = LogisticRegression()(x, y_t)

        wrong = np.zeros((10,))
        with pytest.raises(ValueError):
            Model(wrong, y, y_t)

        with pytest.raises(ValueError):
            Model(x, wrong, y_t)

        with pytest.raises(ValueError):
            Model(x, y, wrong)

    def test_with_missing_inputs(self, teardown):
        x1 = Input()
        x2 = Input()
        c = Concatenate()([x1, x2])

        with pytest.raises(ValueError):
            Model(x1, c)

    @pytest.mark.parametrize("step_class", [PCA, LogisticRegression])
    @pytest.mark.parametrize("trainable", [True, False])
    @pytest.mark.filterwarnings(
        "ignore:You are passing targets to a non-trainable step."
    )
    def test_with_missing_targets(self, step_class, trainable, teardown):
        x = Input()
        y_t = Input()
        y = step_class(trainable=trainable)(x, y_t)
        with pytest.raises(ValueError):
            Model(x, y)

    def test_with_unnecessary_inputs(self, teardown):
        x1 = Input()
        x2 = Input()
        y_t = Input()
        h = PCA()(x1)
        y = LogisticRegression()(h, y_t)

        with pytest.raises(ValueError):
            Model([x1, x2], y, y_t)

        with pytest.raises(ValueError):
            Model([x1, h], y, y_t)  # x1 is an unnecessary input upstream of h

    # TODO: Add case of class without fit method
    @pytest.mark.parametrize("step_class", [PCA, LogisticRegression])
    @pytest.mark.parametrize("trainable", [True, False])
    def test_with_unnecessary_targets(self, step_class, trainable, teardown):
        x = Input()
        y_t = Input()
        y = step_class(trainable=trainable)(x)
        with pytest.raises(ValueError):
            Model(x, y, y_t)  # y_t was not used anywhere

    def test_with_duplicated_inputs(self, teardown):
        x = Input()
        y_t = Input()
        y = LogisticRegression()(x, y_t)
        with pytest.raises(ValueError):
            Model([x, x], y, y_t)

    def test_with_duplicated_outputs(self, teardown):
        x = Input()
        y_t = Input()
        y = LogisticRegression()(x, y_t)
        with pytest.raises(ValueError):
            Model(x, [y, y], y_t)

    def test_with_duplicated_targets(self, teardown):
        x = Input()
        y_t = Input()
        y = LogisticRegression()(x, y_t)
        with pytest.raises(ValueError):
            Model(x, y, [y_t, y_t])

    def test_with_steps_with_duplicated_names(self, teardown):
        x = Input()
        h = PCA(name="duplicated-name")(x)
        y = LogisticRegression(name="duplicated-name")(h)

        with pytest.raises(RuntimeError):
            Model(x, y)


class TestFit:
    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_t_data = iris.target

    @pytest.fixture
    def dataplaceholders(self):
        x1 = Input(name="x1")
        x2 = Input(name="x2")
        y1_t = Input(name="y1_t")
        y1 = LogisticRegression()(x1, y1_t)
        y2 = PCA()(x2)
        return x1, x2, y1, y2, y1_t

    @pytest.fixture
    def model(self, dataplaceholders):
        x1, x2, y1, y2, y1_t = dataplaceholders
        return Model([x1, x2], [y1, y2], y1_t)

    @pytest.fixture(
        params=[List[ArrayLike], Dict[DataPlaceholder, ArrayLike], Dict[str, ArrayLike]]
    )
    def X_y_proper(self, dataplaceholders, request):
        x1, x2, y1, y2, y1_t = dataplaceholders
        X_y_type = request.param

        if X_y_type == List[ArrayLike]:
            return [self.x1_data, self.x2_data], self.y1_t_data
        elif X_y_type == Dict[DataPlaceholder, ArrayLike]:
            return {x1: self.x1_data, x2: self.x2_data}, {y1_t: self.y1_t_data}
        elif X_y_type == Dict[str, ArrayLike]:
            return {"x1": self.x1_data, "x2": self.x2_data}, {"y1_t": self.y1_t_data}

    @pytest.fixture(
        params=[List[ArrayLike], Dict[DataPlaceholder, ArrayLike], Dict[str, ArrayLike]]
    )
    def X_y_missing_input(self, dataplaceholders, request):
        x1, x2, y1, y2, y1_t = dataplaceholders
        X_y_type = request.param

        if X_y_type == List[ArrayLike]:
            return [self.x1_data], self.y1_t_data
        elif X_y_type == Dict[DataPlaceholder, ArrayLike]:
            return {x1: self.x1_data}, {y1_t: self.y1_t_data}
        elif X_y_type == Dict[str, ArrayLike]:
            return {"x1": self.x1_data}, {"y1_t": self.y1_t_data}

    @pytest.fixture(
        params=[List[ArrayLike], Dict[DataPlaceholder, ArrayLike], Dict[str, ArrayLike]]
    )
    def X_y_missing_target(self, dataplaceholders, request):
        x1, x2, y1, y2, y1_t = dataplaceholders
        X_y_type = request.param

        if X_y_type == List[ArrayLike]:
            return [self.x1_data, self.x2_data], []
        elif X_y_type == Dict[DataPlaceholder, ArrayLike]:
            return {x1: self.x1_data, x2: self.x2_data}, {}
        elif X_y_type == Dict[str, ArrayLike]:
            return {"x1": self.x1_data, "x2": self.x2_data}, {}

    def test_with_proper_inputs_and_targets(self, model, X_y_proper, teardown):
        X, y = X_y_proper
        model.fit(X, y)

    def test_with_missing_input(self, model, X_y_missing_input, teardown):
        X, y = X_y_missing_input
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_with_missing_target(self, model, X_y_missing_target, teardown):
        X, y = X_y_missing_target
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_with_unknown_input(self, model, teardown):
        with pytest.raises(ValueError):
            model.fit(
                {"x1": self.x1_data, "unknown-input": self.x2_data},
                {"y1_t_data": self.y1_t_data},
            )

    def test_with_unknown_target(self, model, teardown):
        with pytest.raises(ValueError):
            model.fit(
                {"x1": self.x1_data, "x2": self.x2_data},
                {"unknown-target": self.y1_t_data},
            )

    # TODO: Add test of unknown input/target passed as a list

    def test_with_undefined_target(self, teardown):
        x = Input()
        y = LogisticRegression(trainable=True)(x)
        model = Model(inputs=x, outputs=y)
        with pytest.raises(TypeError):
            # LogisticRegression.fit will be called with not enough arguments
            # hence the TypeError
            model.fit(iris.data)

    def test_with_unnecessarily_defined_but_missing_target(self, teardown):
        x = Input()
        y_t = Input()
        pca = PCA(trainable=True)
        # The target passed to PCA is unnecessary (see notes in Step.__call__)
        y = pca(x, y_t)
        model = Model(inputs=x, outputs=y, targets=y_t)
        with pytest.raises(ValueError):
            # fails because of the model target specification and trainable=True
            model.fit(iris.data)

    def test_with_unnecessary_target(self, teardown):
        x = Input()
        y_t = Input()
        classifier = RandomForestClassifier()
        y_p = classifier(x, y_t)
        model = Model(x, y_p, y_t)

        model.fit(iris.data, iris.target)

        # won't require the target is trainable was set to False,
        # but won't complain if it was passed to fit
        classifier.trainable = False
        model.fit(iris.data, iris.target)

    def test_with_non_trainable_step(self, teardown):
        x = Input()
        y = PCA(trainable=False)(x)
        model = Model(x, y)
        # this should not raise an error because PCA has no successor steps
        model.fit(iris.data)

    @skip_sklearn_0_22
    def test_with_non_fitted_non_trainable_step(self, teardown):
        x = Input()
        y_t = Input()
        z = PCA(trainable=False)(x)
        y = LogisticRegression()(z, y_t)
        model = Model(x, y, y_t)
        with pytest.raises(NotFittedError):
            # this will raise an error when calling compute
            # on PCA which was flagged as trainable=False but
            # hasn't been fitted
            model.fit(iris.data, iris.target)

    @pytest.mark.parametrize(
        "step_class, trainable",
        [(PCA, True), (PCA, False), (LogisticRegression, False)],
    )
    @pytest.mark.filterwarnings(
        "ignore:You are passing targets to a non-trainable step."
    )
    def test_with_superfluous_target(self, step_class, trainable, teardown):
        x = Input()
        y_t = Input()
        z = step_class(trainable=trainable)(x, y_t)
        model = Model(x, z, y_t)
        model.fit(iris.data, iris.target)  # should not raise any errors


class TestPredict:
    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_t_data = iris.target

    @pytest.fixture
    def dataplaceholders(self):
        x1 = Input(name="x1")
        x2 = Input(name="x2")
        y1_t = Input(name="y1_t")
        x1_rescaled = StandardScaler()(x1)
        y1 = LogisticRegression()(x1_rescaled, y1_t)
        y2 = PCA()(x2)
        return x1, x2, x1_rescaled, y1, y2, y1_t

    @pytest.fixture
    def model(self, dataplaceholders):
        x1, x2, _, y1, y2, y1_t = dataplaceholders
        model = Model(inputs=[x1, x2], outputs=[y1, y2], targets=y1_t)
        model.fit([self.x1_data, self.x2_data], self.y1_t_data)
        return model

    @pytest.fixture(
        params=[List[ArrayLike], Dict[DataPlaceholder, ArrayLike], Dict[str, ArrayLike]]
    )
    def X_proper(self, dataplaceholders, request):
        x1, x2, *_ = dataplaceholders
        X_type = request.param

        if X_type == List[ArrayLike]:
            return [self.x1_data, self.x2_data]
        elif X_type == Dict[DataPlaceholder, ArrayLike]:
            return {x1: self.x1_data, x2: self.x2_data}
        elif X_type == Dict[str, ArrayLike]:
            return {"x1": self.x1_data, "x2": self.x2_data}

    @pytest.fixture(
        params=[List[ArrayLike], Dict[DataPlaceholder, ArrayLike], Dict[str, ArrayLike]]
    )
    def X_missing_input(self, dataplaceholders, request):
        x1, x2, *_ = dataplaceholders
        X_type = request.param

        if X_type == List[ArrayLike]:
            return [self.x1_data]
        elif X_type == Dict[DataPlaceholder, ArrayLike]:
            return {x1: self.x1_data}
        elif X_type == Dict[str, ArrayLike]:
            return {"x1": self.x1_data}

    def test_with_proper_inputs(self, model, X_proper, teardown):
        model.predict(X_proper)

    def test_with_missing_input(self, model, X_missing_input, teardown):
        with pytest.raises(ValueError):
            model.predict(X_missing_input)

    def test_with_nonexisting_input(self, model, teardown):
        with pytest.raises(ValueError):
            model.predict({"x1": self.x1_data, "nonexisting-input": self.x2_data})

    def test_with_nonexisting_output(self, model, teardown):
        with pytest.raises(ValueError):
            model.predict(
                {"x1": self.x1_data, "x2": self.x2_data},
                ["non-existing-output", "PCA_0/0"],
            )

    def test_with_unnecessary_input(self, model, teardown):
        # x2 is not needed to compute PCA_0/0
        model.predict({"x1": self.x1_data, "x2": self.x2_data}, "PCA_0/0")

    def test_with_duplicated_output(self, model, teardown):
        with pytest.raises(ValueError):
            model.predict(
                [self.x1_data, self.x2_data],
                ["LogisticRegression_0/0", "LogisticRegression_0/0", "PCA_0/0"],
            )

    @pytest.mark.parametrize("output", ["LogisticRegression_0/0", "StandardScaler_0/0"])
    def test_with_specified_output(self, model, output, teardown):
        model.predict({"x1": self.x1_data}, output)

    def test_with_improperly_defined_step(self, teardown):
        x = Input()
        y = DummyImproperlyDefined()(x)
        model = Model(x, y)

        with pytest.raises(RuntimeError):
            model.predict(iris.data)

    @skip_sklearn_0_22
    def test_predict_with_not_fitted_steps(self, teardown):
        x_data = iris.data

        x = Input(name="x")
        xt = PCA(n_components=2)(x)
        y = LogisticRegression(multi_class="multinomial", solver="lbfgs")(xt)

        model = Model(x, y)
        with pytest.raises(NotFittedError):
            model.predict(x_data)


def test_steps_cache(teardown):
    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_t_data = iris.target

    x1 = Input(name="x1")
    x2 = Input(name="x2")
    y1_t = Input(name="y1_t")
    y1 = LogisticRegression(name="LogReg")(x1, y1_t)
    y2 = PCA(name="PCA")(x2)

    hits, misses = 0, 0

    # 1) instantiation always misses
    misses += 1
    model = Model([x1, x2], [y1, y2], y1_t)
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 2) calling fit for the first time, hence a miss
    misses += 1
    model.fit([x1_data, x2_data], y1_t_data)
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 3) same as above, just different format, hence a hit
    hits += 1
    model.fit({x1: x1_data, x2: x2_data}, {y1_t: y1_t_data})
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 4) trainable flags are considered in cache keys, hence a miss
    misses += 1
    model.get_step("LogReg").trainable = False
    model.fit(
        [x1_data, x2_data], y1_t_data
    )  # NOTE: target is superfluous, but it affects caching
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 5) same as above, just different format, hence a hit
    hits += 1
    model.fit({x1: x1_data, x2: x2_data}, y1_t_data)
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 6) we drop the (superflous) target, hence a miss
    misses += 1
    model.fit({x1: x1_data, x2: x2_data})
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 7) same as above, hence a hit
    hits += 1
    model.fit({x1: x1_data, x2: x2_data})
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 8) we restore the flag, becoming the same as 2) and 3), hence a hit
    hits += 1
    model.get_step("LogReg").trainable = True
    model.fit({x1: x1_data, x2: x2_data}, y1_t_data)
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 9) new inputs/targets/outputs signature, hence a miss
    misses += 1
    model.predict([x1_data, x2_data])
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 10) same inputs/outputs signature as 9), hence a hit
    hits += 1
    model.predict({"x1": x1_data, "x2": x2_data}, ["PCA/0", "LogReg/0"])
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 11) new inputs/outputs signature, hence a miss
    misses += 1
    model.predict({x1: x1_data}, "LogReg/0")
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses

    # 12) same as above, hence a hit
    hits += 1
    model.predict({x1: x1_data}, "LogReg/0")
    assert hits == model._steps_cache.hits and misses == model._steps_cache.misses


def test_multiedge(teardown):
    x = Input()
    z1, z2 = DummySIMO()(x)
    y = DummyMISO()([z1, z2])
    model = Model(x, y)

    x_data = np.array([[1], [2]])
    y_out = model.predict(x_data)

    assert_array_equal(y_out, np.array([[2], [4]]))


def test_lazy_model(teardown):
    x_data = np.array([[1, 2], [3, 4]])

    x = Input()
    model = Model(x, x)
    model.fit(x_data)  # nothing to fit
    x_pred = model.predict(x_data)

    assert_array_equal(x_pred, x_data)


def test_transformed_target(teardown):
    x = Input()
    y_t = Input()
    y_t_mod = Lambda(lambda y: np.log(y))(y_t)
    y_p_mod = LinearRegression()(x, y_t_mod)
    y_p = Lambda(lambda y: np.exp(y))(y_p_mod)
    model = Model(x, y_p, y_t)

    x_data = np.arange(4).reshape(-1, 1)
    y_t_data = np.exp(2 * x_data).ravel()
    model.fit(x_data, y_t_data)

    assert_array_equal(model.get_step("LinearRegression_0").coef_, np.array([2.0]))


def test_fit_and_predict_model_with_no_fittable_steps(teardown):
    X1_data = np.array([[1, 2], [3, 4]])
    X2_data = np.array([[5, 6], [7, 8]])
    y_expected = np.array([[12, 16], [20, 24]])

    x1 = Input()
    x2 = Input()
    z = DummyMISO()([x1, x2])
    y = DummySISO()(z)

    model = Model([x1, x2], y)
    model.fit([X1_data, X2_data])  # nothing to fit
    y_pred = model.predict([X1_data, X2_data])

    assert_array_equal(y_pred, y_expected)


def test_fit_predict_pipeline(teardown):
    x_data = iris.data
    y_t_data = iris.target
    random_state = 123
    n_components = 2

    # baikal way
    x = Input()
    y_t = Input()
    x_pca = PCA(n_components=n_components, random_state=random_state, name="pca")(x)
    y = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=random_state,
        name="logreg",
    )(x_pca, y_t)

    model = Model(x, y, y_t)
    y_pred_baikal = model.fit(x_data, y_t_data).predict(x_data)

    # traditional way
    pca = sklearn.decomposition.PCA(
        n_components=n_components, random_state=random_state
    )
    logreg = sklearn.linear_model.LogisticRegression(
        multi_class="multinomial", solver="lbfgs", random_state=random_state
    )
    x_data_transformed = pca.fit_transform(x_data)
    y_pred_traditional = logreg.fit(x_data_transformed, y_t_data).predict(
        x_data_transformed
    )

    assert_array_equal(y_pred_baikal, y_pred_traditional)


def test_fit_predict_ensemble(teardown):
    mask = iris.target != 2  # Reduce to binary problem to avoid ConvergenceWarning
    x_data = iris.data
    y_t_data = iris.target
    random_state = 123

    # baikal way
    x = Input()
    y_t = Input()
    y1 = LogisticRegression(random_state=random_state, solver="liblinear")(x, y_t)
    y2 = RandomForestClassifier(random_state=random_state)(x, y_t)
    features = Stack(axis=1)([y1, y2])
    y = LogisticRegression(random_state=random_state, solver="liblinear")(features, y_t)

    model = Model(x, y, y_t)
    model.fit(x_data, y_t_data)
    y_pred_baikal = model.predict(x_data)

    # traditional way
    logreg = sklearn.linear_model.LogisticRegression(
        random_state=random_state, solver="liblinear"
    )
    logreg.fit(x_data, y_t_data)
    logreg_pred = logreg.predict(x_data)

    random_forest = sklearn.ensemble.RandomForestClassifier(random_state=random_state)
    random_forest.fit(x_data, y_t_data)
    random_forest_pred = random_forest.predict(x_data)

    features = np.stack([logreg_pred, random_forest_pred], axis=1)
    ensemble = sklearn.linear_model.LogisticRegression(
        random_state=random_state, solver="liblinear"
    )
    ensemble.fit(features, y_t_data)
    y_pred_traditional = ensemble.predict(features)

    assert_array_equal(y_pred_baikal, y_pred_traditional)


def test_fit_predict_ensemble_with_proba_features(teardown):
    mask = iris.target != 2  # Reduce to binary problem to avoid ConvergenceWarning
    x_data = iris.data[mask]
    y_t_data = iris.target[mask]
    random_state = 123
    n_estimators = 5

    # baikal way
    x = Input()
    y_t = Input()
    y1 = LogisticRegression(random_state=random_state, function="predict_proba")(x, y_t)
    y2 = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, function="apply"
    )(x, y_t)
    features = Concatenate(axis=1)([y1, y2])
    y = LogisticRegression(random_state=random_state)(features, y_t)

    model = Model(x, y, y_t)
    model.fit(x_data, y_t_data)
    y_pred_baikal = model.predict(x_data)

    # traditional way
    logreg = sklearn.linear_model.LogisticRegression(random_state=random_state)
    logreg.fit(x_data, y_t_data)
    logreg_proba = logreg.predict_proba(x_data)

    random_forest = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    random_forest.fit(x_data, y_t_data)
    random_forest_leafidx = random_forest.apply(x_data)

    features = np.concatenate([logreg_proba, random_forest_leafidx], axis=1)
    ensemble = sklearn.linear_model.LogisticRegression(random_state=random_state)
    ensemble.fit(features, y_t_data)
    y_pred_traditional = ensemble.predict(features)

    assert_array_equal(y_pred_baikal, y_pred_traditional)


@skip_sklearn_0_22
def test_nested_model(teardown):
    x_data = iris.data
    y_t_data = iris.target

    # Sub-model
    x = Input()
    y_t = Input()
    h = PCA(n_components=2)(x)
    y = LogisticRegression()(h, y_t)
    submodel = Model(x, y, y_t)

    # Model
    x = Input()
    y_t = Input()
    y = submodel(x, y_t)
    model = Model(x, y, y_t)

    with pytest.raises(NotFittedError):
        submodel.predict(x_data)

    model.fit(x_data, y_t_data)
    y_pred = model.predict(x_data)
    y_pred_sub = submodel.predict(x_data)

    assert_array_equal(y_pred, y_pred_sub)


def test_nested_model_ensemble(teardown):
    x_data = iris.data
    y_t_data = iris.target
    random_state = 123
    n_components = 2

    # ----------- baikal way
    ensemble_model_baikal = make_ensemble_model(
        n_components, random_state, x_data, y_t_data
    )
    y_pred_baikal = ensemble_model_baikal.predict(x_data)

    # ----------- traditional way
    # Submodel 1
    submodel1 = sklearn.linear_model.LogisticRegression(
        multi_class="multinomial", solver="lbfgs", random_state=random_state
    )
    pca = sklearn.decomposition.PCA(
        n_components=n_components, random_state=random_state
    )
    pca.fit(x_data)
    pca_trans = pca.transform(x_data)
    submodel1.fit(pca_trans, y_t_data)
    submodel1_pred = submodel1.predict(pca_trans)

    # Submodel 2 (a nested ensemble model)
    random_forest = sklearn.ensemble.RandomForestClassifier(random_state=random_state)
    random_forest.fit(x_data, y_t_data)
    random_forest_pred = random_forest.predict(x_data)

    extra_trees = sklearn.ensemble.ExtraTreesClassifier(random_state=random_state)
    extra_trees.fit(x_data, y_t_data)
    extra_trees_pred = extra_trees.predict(x_data)

    features = np.stack([random_forest_pred, extra_trees_pred], axis=1)
    submodel2 = sklearn.linear_model.LogisticRegression(
        multi_class="multinomial", solver="lbfgs", random_state=random_state
    )
    submodel2.fit(features, y_t_data)
    submodel2_pred = submodel2.predict(features)

    # Ensemble model
    features = np.stack([submodel1_pred, submodel2_pred], axis=1)
    ensemble_model_traditional = sklearn.linear_model.LogisticRegression(
        multi_class="multinomial", solver="lbfgs", random_state=random_state
    )
    ensemble_model_traditional.fit(features, y_t_data)
    y_pred_traditional = ensemble_model_traditional.predict(features)

    assert_array_equal(y_pred_baikal, y_pred_traditional)


@pytest.mark.parametrize(
    "dump,load", [(joblib.dump, joblib.load), (pickle.dump, pickle.load)]
)
def test_model_joblib_serialization(teardown, dump, load):
    x_data = iris.data
    y_t_data = iris.target
    random_state = 123
    n_components = 2

    ensemble_model_baikal = make_ensemble_model(
        n_components, random_state, x_data, y_t_data
    )
    y_pred_baikal = ensemble_model_baikal.predict(x_data)

    # Persist model to a file
    f = tempfile.TemporaryFile()
    dump(ensemble_model_baikal, f)
    f.seek(0)
    ensemble_model_baikal_2 = load(f)
    y_pred_baikal_2 = ensemble_model_baikal_2.predict(x_data)

    assert_array_equal(y_pred_baikal_2, y_pred_baikal)


def test_fit_params(teardown):
    x_data = iris.data
    y_t_data = iris.target
    random_state = 123
    n_components = 2

    sample_weight = y_t_data + 1  # Just weigh the classes differently
    fit_params = {"logreg__sample_weight": sample_weight}

    # baikal way
    x = Input()
    y_t = Input()
    x_pca = PCA(n_components=n_components, random_state=random_state, name="pca")(x)
    y = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=random_state,
        name="logreg",
    )(x_pca, y_t)

    model = Model(x, y, y_t)
    model.fit(x_data, y_t_data, **fit_params)

    # traditional way
    pca = sklearn.decomposition.PCA(
        n_components=n_components, random_state=random_state
    )
    logreg = sklearn.linear_model.LogisticRegression(
        multi_class="multinomial", solver="lbfgs", random_state=random_state
    )
    pipe = Pipeline([("pca", pca), ("logreg", logreg)])
    pipe.fit(x_data, y_t_data, **fit_params)

    # Use assert_allclose instead of all equal due to small numerical differences
    # between fit_transform(...) and fit(...).transform(...)
    assert_allclose(model.get_step("logreg").coef_, pipe.named_steps["logreg"].coef_)


def test_get_params(teardown):
    dummy1 = DummyEstimator(name="dummy1")
    dummy2 = DummyEstimator(x=456, y="def", name="dummy2")
    concat = Concatenate(name="concat")  # a step without get_params/set_params

    x = Input()
    h = dummy1(x)
    c = concat([x, h])
    y = dummy2(c)
    model = Model(x, y)

    expected = {
        "dummy1": dummy1,
        "dummy2": dummy2,
        "concat": concat,
        "dummy1__x": 123,
        "dummy1__y": "abc",
        "dummy2__x": 456,
        "dummy2__y": "def",
    }

    params = model.get_params()
    assert expected == params


def test_set_params(teardown):
    dummy1 = DummyEstimator(name="dummy1")
    dummy2 = DummyEstimator(x=456, y="def", name="dummy2")
    concat = Concatenate(name="concat")  # a step without get_params/set_params

    x = Input()
    h = dummy1(x)
    c = concat([x, h])
    y = dummy2(c)
    model = Model(x, y)

    # Fails when setting params on step that does not implement set_params
    new_params_wrong = {"concat__axis": 2}
    with pytest.raises(AttributeError):
        model.set_params(**new_params_wrong)

    # Fails when setting params on step that does not exist
    new_params_wrong = {"non_existent_step__param": 42}
    with pytest.raises(ValueError):
        model.set_params(**new_params_wrong)

    # Fails when setting a non-existent param in a step
    new_params_wrong = {"dummy1__non_existent_param": 42}
    with pytest.raises(ValueError):
        model.set_params(**new_params_wrong)

    new_dummy = DummyEstimator()
    new_params = {
        "dummy2": new_dummy,
        "dummy1__x": 100,
        "dummy1__y": "pqr",
        "dummy2__x": 789,
        "dummy2__y": "ijk",
    }

    model.set_params(**new_params)
    params = model.get_params()

    expected = {
        "dummy1": dummy1,
        "dummy2": new_dummy,
        "concat": concat,
        "dummy1__x": 100,
        "dummy1__y": "pqr",
        "dummy2__x": 789,
        "dummy2__y": "ijk",
    }

    assert expected == params


def test_get_set_params_invariance(teardown):
    pca = PCA(name="pca")
    classifier = RandomForestClassifier(name="classifier")

    x = Input()
    h = pca(x)
    y = classifier(h)
    model = Model(x, y)

    params1 = model.get_params()
    model.set_params(**params1)
    params2 = model.get_params()
    assert params1 == params2


def test_trainable_flag(teardown):
    x_data = iris.data
    y_t_data = iris.target
    random_state = 123
    n_components = 2

    ensemble_model_baikal = make_ensemble_model(
        n_components, random_state, x_data, y_t_data
    )

    # Set sub-model 1's LogisticRegression to untrainable and
    # retrain model on a subset of the data
    np.random.seed(456)
    n_samples = len(x_data) // 2
    idx = np.random.choice(np.arange(len(x_data)), size=n_samples, replace=False)
    x_data_sub, y_t_data_sub = x_data[idx], y_t_data[idx]

    logreg_sub1 = ensemble_model_baikal.get_step("submodel1").get_step("logreg_sub1")
    logreg_ensemble = ensemble_model_baikal.get_step("logreg_ensemble")

    logreg_sub1_coef_original = logreg_sub1.coef_.copy()  # This one should not change
    logreg_ensemble_coef_original = (
        logreg_ensemble.coef_.copy()
    )  # This one should change
    logreg_sub1.trainable = False

    ensemble_model_baikal.fit(x_data_sub, y_t_data_sub)
    logreg_sub1_coef_retrained = logreg_sub1.coef_
    logreg_ensemble_coef_retrained = logreg_ensemble.coef_

    assert_array_equal(logreg_sub1_coef_original, logreg_sub1_coef_retrained)
    with pytest.raises(AssertionError):
        assert_array_equal(
            logreg_ensemble_coef_original, logreg_ensemble_coef_retrained
        )


def make_ensemble_model(n_components, random_state, x_data, y_t_data):
    # An unnecessarily complex Model

    # Sub-model 1
    x1 = Input(name="x1")
    y1_t = Input(name="y1_t")
    h1 = PCA(n_components=n_components, random_state=random_state, name="pca_sub1")(x1)
    y1 = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=random_state,
        name="logreg_sub1",
    )(h1, y1_t)
    submodel1 = Model(x1, y1, y1_t, name="submodel1")

    # Sub-model 2 (a nested ensemble model)
    x2 = Input(name="x2")
    y2_t = Input(name="y2_t")
    y2_1 = RandomForestClassifier(random_state=random_state, name="rforest_sub2")(
        x2, y2_t
    )
    y2_2 = ExtraTreesClassifier(random_state=random_state, name="extrees_sub2")(
        x2, y2_t
    )
    features = Stack(axis=1, name="stack_sub2")([y2_1, y2_2])
    y2 = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=random_state,
        name="logreg_sub2",
    )(features, y2_t)
    submodel2 = Model(x2, y2, y2_t, name="submodel2")

    # Ensemble of submodels
    x = Input(name="x")
    y_t = Input(name="y_t")
    y1 = submodel1(x, y_t)
    y2 = submodel2(x, y_t)
    features = Stack(axis=1, name="stack")([y1, y2])
    y = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=random_state,
        name="logreg_ensemble",
    )(features, y_t)
    ensemble_model_baikal = Model(x, y, y_t, name="ensemble")

    ensemble_model_baikal.fit(x_data, y_t_data)

    return ensemble_model_baikal
