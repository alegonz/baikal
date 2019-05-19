import tempfile
from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
from sklearn import datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.externals.joblib
from sklearn.exceptions import NotFittedError
import sklearn.linear_model
from sklearn.pipeline import Pipeline

from baikal import Model, Input
from baikal.steps import Stack, Concatenate

from tests.helpers.fixtures import teardown
from tests.helpers.sklearn_steps import (LogisticRegression, RandomForestClassifier, ExtraTreesClassifier,
                                         PCA, StandardScaler)
from tests.helpers.dummy_steps import (DummySISO, DummySIMO, DummyMISO, DummyMIMO,
                                       DummyWithoutTransform, DummyImproperlyDefined)


pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning:sklearn',
                                        'ignore::FutureWarning:sklearn')


iris = datasets.load_iris()


@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def dummy_model_data_placeholders_and_arrays():
    x1 = Input(name='x1')
    x2 = Input(name='x2')

    z1 = DummySISO()(x1)
    z2, z3 = DummySIMO()(x2)
    z4 = DummyMISO()([z1, z2])
    z5, z6 = DummyMIMO()([z4, z3])

    data_placeholders = {'x1': x1,
                         'x2': x2,
                         'z1': z1,
                         'z2': z2,
                         'z3': z3,
                         'z4': z4,
                         'z5': z5,
                         'z6': z6}

    # arrays have explicit sample (rows) and feature (columns) axes
    arrays = {'x1': np.array([[1]]),
              'x2': np.array([[2]]),
              'z1': np.array([[2]]),
              'z2': np.array([[2]]),
              'z3': np.array([[0]]),
              'z4': np.array([[4]]),
              'z5': np.array([[40]]),
              'z6': np.array([[0]])}

    return data_placeholders, arrays


@pytest.mark.parametrize('inputs,outputs,expectation',
                         [(['x1'], ['z1'], does_not_raise()),
                          (['x1'], ['x1'], does_not_raise()),
                          (['x1', 'x2'], ['z5', 'z6'], does_not_raise()),
                          (['x1', 'x2'], ['z1', 'z2'], does_not_raise()),
                          (['z3', 'z4'], ['z5'], does_not_raise()),
                          (['x1', 'x1'], ['z1'], pytest.raises(ValueError)),  # duplicated input
                          (['x1'], ['z1', 'z1'], pytest.raises(ValueError)),  # duplicated output
                          (['x1'], ['x2'], pytest.raises(RuntimeError)),
                          (['z1'], ['z4'], pytest.raises(RuntimeError)),
                          (['z1', 'z2'], ['z5'], pytest.raises(RuntimeError)),
                          (['x1', 'x2'], ['z1'], pytest.raises(RuntimeError)),
                          (['z1', 'z2', 'x1'], ['z4'], pytest.raises(RuntimeError)),
                          (['z1', 'z2', 'x2'], ['z4'], pytest.raises(RuntimeError))])
def test_instantiation(inputs, outputs, expectation,
                       dummy_model_data_placeholders_and_arrays, teardown):
    data_placeholders, _ = dummy_model_data_placeholders_and_arrays

    inputs = [data_placeholders[i] for i in inputs]
    outputs = [data_placeholders[o] for o in outputs]

    with expectation:
        Model(inputs, outputs)


def test_fit_call(teardown):
    x1 = Input(name='x1')
    x2 = Input(name='x2')
    y1 = LogisticRegression()(x1)
    y2 = PCA()(x2)
    model = Model([x1, x2], [y1, y2])

    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_data = iris.target

    # ------ Correct calls. Should not raise errors.
    # Call with lists
    model.fit([x1_data, x2_data], [y1_data, None])

    # Call with dicts (data_placeholder keys)
    model.fit({x1: x1_data, x2: x2_data}, {y1: y1_data, y2: None})

    # Call with dicts (name (str) keys)
    model.fit({'x1': x1_data, 'x2': x2_data}, {'LogisticRegression_0/0': y1_data, 'PCA_0/0': None})

    # ------ Missing input
    # Call with lists
    with pytest.raises(ValueError):
        model.fit([x1_data], [y1_data, None])

    # Call with dicts (data_placeholder keys)
    with pytest.raises(ValueError):
        model.fit({x1: x1_data}, {y1: y1_data, y2: None})

    # Call with dicts (name (str) keys)
    with pytest.raises(ValueError):
        model.fit({'x1': x1_data}, {'LogisticRegression_0/0': y1_data, 'PCA_0/0': None})

    # ------ Missing output
    # Call with lists
    with pytest.raises(ValueError):
        model.fit([x1_data, x2_data], [None])

    # Call with dicts (data_placeholder keys)
    with pytest.raises(ValueError):
        model.fit({x1: x1_data, x2: x2_data}, {y2: None})

    # Call with dicts (name (str) keys)
    with pytest.raises(ValueError):
        model.fit({'x1': x1_data, 'x2': x2_data}, {'PCA_0/0': None})

    # ------ Non-existing inputs
    with pytest.raises(ValueError):
        model.fit({'x1': x1_data, 'x3': x2_data}, {'LogisticRegression_0/0': y1_data, 'PCA_0/0': None})

    # ------ Non-existing outputs
    with pytest.raises(ValueError):
        model.fit({'x1': x1_data, 'x2': x2_data}, {'non-existing-output': y1_data, 'PCA_0/0': None})


def test_predict_call(teardown):
    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_data = iris.target

    x1 = Input(name='x1')
    x2 = Input(name='x2')
    x1_rescaled = StandardScaler()(x1)
    y1 = LogisticRegression()(x1_rescaled)
    y2 = PCA()(x2)
    model = Model([x1, x2], [y1, y2])

    model.fit([x1_data, x2_data], [y1_data, None])

    # ------ Correct calls. Should not raise errors.
    # Call with list input. Get all outputs.
    model.predict([x1_data, x2_data])

    # Call with dict input (data_placeholder keys). Get all outputs.
    model.predict({x1: x1_data, x2: x2_data})

    # Call with dict input (name (str) keys). Get all outputs.
    model.predict({'x1': x1_data, 'x2': x2_data})

    # Call with dict input. Get an specific output. Call with just the needed input
    model.predict({x1: x1_data}, 'LogisticRegression_0/0')

    # Call with dict input. Get intermediate output.
    model.predict({x1: x1_data}, 'StandardScaler_0/0')

    # ------ Missing input
    # Call with list input. Get all outputs.
    with pytest.raises(ValueError):
        model.predict(x1_data)

    # Call with dict input (data_placeholder keys). Get all outputs.
    with pytest.raises(RuntimeError):
        model.predict({x1: x1_data})

    # Call with dict input (name (str) keys). Get all outputs.
    with pytest.raises(RuntimeError):
        model.predict({'x1': x1_data})

    # ------ Non-existing inputs
    with pytest.raises(ValueError):
        model.predict({'x1': x1_data, 'x3': x2_data})

    # ------ Non-existing outputs
    with pytest.raises(ValueError):
        model.predict({'x1': x1_data, 'x2': x2_data}, ['non-existing-output', 'PCA_0/0'])

    # ------ Unnecessary inputs
    with pytest.raises(RuntimeError):
        model.predict({'x1': x1_data, 'x2': x2_data}, 'PCA_0/0')

    # ------ Duplicated outputs
    with pytest.raises(ValueError):
        model.predict([x1_data, x2_data],
                      ['LogisticRegression_0/0', 'LogisticRegression_0/0', 'PCA_0/0'])


def test_with_improperly_defined_step(teardown):
    x = Input()
    y = DummyImproperlyDefined()(x)
    model = Model(x, y)

    with pytest.raises(RuntimeError):
        model.predict(iris.data)


def test_steps_cache(teardown):
    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_data = iris.target

    x1 = Input(name='x1')
    x2 = Input(name='x2')
    y1 = LogisticRegression(name='y1')(x1)
    y2 = PCA(name='y2')(x2)

    model = Model([x1, x2], [y1, y2])
    assert 0 == model._steps_cache.hits and 1 == model._steps_cache.misses

    model.fit([x1_data, x2_data], [y1_data, None])
    assert 1 == model._steps_cache.hits and 1 == model._steps_cache.misses

    model.fit({x1: x1_data, x2: x2_data}, {y1: y1_data, y2: None})
    assert 2 == model._steps_cache.hits and 1 == model._steps_cache.misses

    model.predict({'x1': x1_data, 'x2': x2_data}, ['y2/0', 'y1/0'])
    assert 3 == model._steps_cache.hits and 1 == model._steps_cache.misses

    model.predict([x1_data, x2_data])
    assert 4 == model._steps_cache.hits and 1 == model._steps_cache.misses

    model.predict({x1: x1_data}, 'y1/0')
    assert 4 == model._steps_cache.hits and 2 == model._steps_cache.misses

    model.predict({x1: x1_data}, 'y1/0')
    assert 5 == model._steps_cache.hits and 2 == model._steps_cache.misses


def test_multiedge(teardown):
    x = Input(name='x')
    z1, z2 = DummySIMO()(x)
    y = DummyMISO()([z1, z2])
    model = Model(x, y)

    x_data = np.array([[1], [2]])
    y_out = model.predict(x_data)

    assert_array_equal(y_out, np.array([[2], [4]]))


def test_instantiation_with_wrong_input_type(teardown):
    x = Input(name='x')
    y = DummySISO()(x)

    x_wrong = np.zeros((10,))
    with pytest.raises(ValueError):
        Model(x_wrong, y)


def test_instantiation_with_steps_with_duplicated_names(teardown):
    x = Input(name='x')
    x = DummySISO(name='duplicated-name')(x)
    y = DummySISO(name='duplicated-name')(x)

    with pytest.raises(RuntimeError):
        Model(x, y)


def test_lazy_model(teardown):
    x_data = np.array([[1, 2], [3, 4]])

    x = Input(name='x')
    model = Model(x, x)
    model.fit(x_data)
    x_pred = model.predict(x_data)

    assert_array_equal(x_pred, x_data)


def test_fit_and_predict_model_with_no_fittable_steps(teardown):
    X1_data = np.array([[1, 2], [3, 4]])
    X2_data = np.array([[5, 6], [7, 8]])
    y_expected = np.array([[12, 16], [20, 24]])

    x1 = Input(name='x1')
    x2 = Input(name='x2')
    z = DummyMISO()([x1, x2])
    y = DummySISO()(z)

    model = Model([x1, x2], y)
    model.fit([X1_data, X2_data])  # nothing to fit
    y_pred = model.predict([X1_data, X2_data])

    assert_array_equal(y_pred, y_expected)


def test_predict_with_not_fitted_steps(teardown):
    x_data = iris.data

    x = Input(name='x')
    xt = PCA(n_components=2)(x)
    y = LogisticRegression(multi_class='multinomial', solver='lbfgs')(xt)

    model = Model(x, y)
    with pytest.raises(NotFittedError):
        model.predict(x_data)


def test_predict_using_step_without_transform(teardown):
    x_data = np.array([[1], [2]])

    x = Input(name='x')
    y = DummyWithoutTransform()(x)

    model = Model(x, y)
    with pytest.raises(TypeError):
        # step's function attribute is None
        model.predict(x_data)


def test_fit_predict_pipeline(teardown):
    x_data = iris.data
    y_data = iris.target
    random_state = 123
    n_components = 2

    # baikal way
    x = Input(name='x')
    xt = PCA(n_components=n_components, random_state=random_state, name='pca')(x)
    y = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state, name='logreg')(xt)

    model = Model(x, y)
    y_pred_baikal = model.fit(x_data, y_data).predict(x_data)

    assert xt.step.fitted and y.step.fitted

    # traditional way
    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=random_state)
    logreg = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state)
    x_data_transformed = pca.fit_transform(x_data)
    y_pred_traditional = logreg.fit(x_data_transformed, y_data).predict(x_data_transformed)

    assert_array_equal(y_pred_baikal, y_pred_traditional)


def test_fit_predict_ensemble(teardown):
    mask = iris.target != 2  # Reduce to binary problem to avoid ConvergenceWarning
    x_data = iris.data
    y_data = iris.target
    random_state = 123

    # baikal way
    x = Input(name='x')
    y1 = LogisticRegression(random_state=random_state)(x)
    y2 = RandomForestClassifier(random_state=random_state)(x)
    features = Stack(axis=1)([y1, y2])
    y = LogisticRegression(random_state=random_state)(features)

    model = Model(x, y)
    model.fit(x_data, {y: y_data, y1: y_data, y2: y_data})
    y_pred_baikal = model.predict(x_data)

    # traditional way
    logreg = sklearn.linear_model.LogisticRegression(random_state=random_state)
    logreg.fit(x_data, y_data)
    logreg_pred = logreg.predict(x_data)

    random_forest = sklearn.ensemble.RandomForestClassifier(random_state=random_state)
    random_forest.fit(x_data, y_data)
    random_forest_pred = random_forest.predict(x_data)

    features = np.stack([logreg_pred, random_forest_pred], axis=1)
    ensemble = sklearn.linear_model.LogisticRegression(random_state=random_state)
    ensemble.fit(features, y_data)
    y_pred_traditional = ensemble.predict(features)

    assert_array_equal(y_pred_baikal, y_pred_traditional)


def test_fit_predict_ensemble_with_proba_features(teardown):
    mask = iris.target != 2  # Reduce to binary problem to avoid ConvergenceWarning
    x_data = iris.data[mask]
    y_data = iris.target[mask]
    random_state = 123
    n_estimators = 5

    # baikal way
    x = Input(name='x')
    y1 = LogisticRegression(random_state=random_state, function='predict_proba')(x)
    y2 = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, function='apply')(x)
    features = Concatenate(axis=1)([y1, y2])
    y = LogisticRegression(random_state=random_state)(features)

    model = Model(x, y)
    model.fit(x_data, {y: y_data, y1: y_data, y2: y_data})
    y_pred_baikal = model.predict(x_data)

    # traditional way
    logreg = sklearn.linear_model.LogisticRegression(random_state=random_state)
    logreg.fit(x_data, y_data)
    logreg_proba = logreg.predict_proba(x_data)

    random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    random_forest.fit(x_data, y_data)
    random_forest_leafidx = random_forest.apply(x_data)

    features = np.concatenate([logreg_proba, random_forest_leafidx], axis=1)
    ensemble = sklearn.linear_model.LogisticRegression(random_state=random_state)
    ensemble.fit(features, y_data)
    y_pred_traditional = ensemble.predict(features)

    assert_array_equal(y_pred_baikal, y_pred_traditional)


def test_nested_model(teardown):
    x_data = iris.data
    y_data = iris.target

    # Sub-model
    x = Input()
    h = PCA(n_components=2)(x)
    y = LogisticRegression()(h)
    submodel = Model(x, y)

    # Model
    x = Input()
    y = submodel(x)
    model = Model(x, y)

    with pytest.raises(NotFittedError):
        submodel.predict(x_data)

    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)
    y_pred_sub = submodel.predict(x_data)

    assert_array_equal(y_pred, y_pred_sub)


def test_nested_model_ensemble(teardown):
    x_data = iris.data
    y_data = iris.target
    random_state = 123
    n_components = 2

    # ----------- baikal way
    ensemble_model_baikal = make_ensemble_model(n_components, random_state, x_data, y_data)
    y_pred_baikal = ensemble_model_baikal.predict(x_data)

    # ----------- traditional way
    # Submodel 1
    submodel1 = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state)
    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=random_state)
    pca.fit(x_data)
    pca_trans = pca.transform(x_data)
    submodel1.fit(pca_trans, y_data)
    submodel1_pred = submodel1.predict(pca_trans)

    # Submodel 2 (a nested ensemble model)
    random_forest = sklearn.ensemble.RandomForestClassifier(random_state=random_state)
    random_forest.fit(x_data, y_data)
    random_forest_pred = random_forest.predict(x_data)

    extra_trees = sklearn.ensemble.ExtraTreesClassifier(random_state=random_state)
    extra_trees.fit(x_data, y_data)
    extra_trees_pred = extra_trees.predict(x_data)

    features = np.stack([random_forest_pred, extra_trees_pred], axis=1)
    submodel2 = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state)
    submodel2.fit(features, y_data)
    submodel2_pred = submodel2.predict(features)

    # Ensemble model
    features = np.stack([submodel1_pred, submodel2_pred], axis=1)
    ensemble_model_traditional = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state)
    ensemble_model_traditional.fit(features, y_data)
    y_pred_traditional = ensemble_model_traditional.predict(features)

    assert_array_equal(y_pred_baikal, y_pred_traditional)


def test_model_joblib_serialization(teardown):
    x_data = iris.data
    y_data = iris.target
    random_state = 123
    n_components = 2

    ensemble_model_baikal = make_ensemble_model(n_components, random_state, x_data, y_data)
    y_pred_baikal = ensemble_model_baikal.predict(x_data)

    # Persist model to a file
    f = tempfile.TemporaryFile()
    sklearn.externals.joblib.dump(ensemble_model_baikal, f)
    f.seek(0)
    ensemble_model_baikal_2 = sklearn.externals.joblib.load(f)
    y_pred_baikal_2 = ensemble_model_baikal_2.predict(x_data)

    assert_array_equal(y_pred_baikal_2, y_pred_baikal)


def test_fit_params(teardown):
    x_data = iris.data
    y_data = iris.target
    random_state = 123
    n_components = 2

    sample_weight = y_data + 1  # Just weigh the classes differently
    fit_params = {'logreg__sample_weight': sample_weight}

    # baikal way
    x = Input(name='x')
    xt = PCA(n_components=n_components, random_state=random_state, name='pca')(x)
    y = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state, name='logreg')(xt)

    model = Model(x, y)
    model.fit(x_data, y_data, **fit_params)

    # traditional way
    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=random_state)
    logreg = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state)
    pipe = Pipeline([('pca', pca), ('logreg', logreg)])
    pipe.fit(x_data, y_data, **fit_params)

    # Use assert_allclose instead of all equal due to small numerical differences
    # between fit_transform(...) and fit(...).transform(...)
    assert_allclose(model.get_step('logreg').coef_, pipe.named_steps['logreg'].coef_)


def test_get_params(teardown):
    pca = PCA(name='pca')
    logreg = LogisticRegression(name='logreg')

    x = Input()
    h = pca(x)
    y = logreg(h)
    model = Model(x, y)

    expected = {'pca': pca,
                'logreg': logreg,
                'pca__n_components': None,
                'pca__whiten': False,
                'pca__tol': 0.0,
                'pca__svd_solver': 'auto',
                'pca__copy': True,
                'pca__random_state': None,
                'pca__iterated_power': 'auto',
                'logreg__C': 1.0,
                'logreg__class_weight': None,
                'logreg__dual': False,
                'logreg__fit_intercept': True,
                'logreg__intercept_scaling': 1,
                'logreg__max_iter': 100,
                'logreg__multi_class': 'warn',
                'logreg__n_jobs': None,
                'logreg__penalty': 'l2',
                'logreg__random_state': None,
                'logreg__solver': 'warn',
                'logreg__tol': 0.0001,
                'logreg__verbose': 0,
                'logreg__warm_start': False,
                'logreg__l1_ratio': None}

    params = model.get_params()
    assert expected == params


def test_set_params(teardown):
    pca = PCA(name='pca')
    classifier = RandomForestClassifier(name='classifier')

    x = Input()
    h = pca(x)
    y = classifier(h)
    model = Model(x, y)

    new_params_wrong = {'non_existent_step__param': 42}
    with pytest.raises(ValueError):
        model.set_params(**new_params_wrong)

    new_params_wrong = {'pca__non_existent_param': 42}
    with pytest.raises(ValueError):
        model.set_params(**new_params_wrong)

    new_classifier = LogisticRegression()
    new_params = {'classifier': new_classifier,
                  'pca__n_components': 4,
                  'pca__whiten': True,
                  'classifier__C': 100.0,
                  'classifier__fit_intercept': False,
                  'classifier__penalty': 'l1'}

    model.set_params(**new_params)
    params = model.get_params()

    expected = {'pca': pca,
                'classifier': new_classifier,
                'pca__n_components': 4,
                'pca__whiten': True,
                'pca__tol': 0.0,
                'pca__svd_solver': 'auto',
                'pca__copy': True,
                'pca__random_state': None,
                'pca__iterated_power': 'auto',
                'classifier__C': 100.0,
                'classifier__class_weight': None,
                'classifier__dual': False,
                'classifier__fit_intercept': False,
                'classifier__intercept_scaling': 1,
                'classifier__max_iter': 100,
                'classifier__multi_class': 'warn',
                'classifier__n_jobs': None,
                'classifier__penalty': 'l1',
                'classifier__random_state': None,
                'classifier__solver': 'warn',
                'classifier__tol': 0.0001,
                'classifier__verbose': 0,
                'classifier__warm_start': False,
                'classifier__l1_ratio': None}

    assert expected == params


def test_get_set_params_invariance(teardown):
    pca = PCA(name='pca')
    classifier = RandomForestClassifier(name='classifier')

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
    y_data = iris.target
    random_state = 123
    n_components = 2

    ensemble_model_baikal = make_ensemble_model(n_components, random_state, x_data, y_data)

    # Set sub-model 1's LogisticRegression to untrainable and
    # retrain model on a subset of the data
    np.random.seed(456)
    n_samples = len(x_data) // 2
    idx = np.random.choice(np.arange(len(x_data)), size=n_samples, replace=False)
    x_data_sub, y_data_sub = x_data[idx], y_data[idx]

    logreg_sub1 = ensemble_model_baikal.get_step('submodel1').get_step('logreg_sub1')
    logreg_ensemble = ensemble_model_baikal.get_step('logreg_ensemble')

    logreg_sub1_coef_original = logreg_sub1.coef_.copy()  # This one should not change
    logreg_ensemble_coef_original = logreg_ensemble.coef_.copy()  # This one should change
    logreg_sub1.trainable = False

    fit_params = {'extra_targets': {'submodel1/0': y_data_sub,
                                    'submodel2/0': y_data_sub},
                  'submodel2__extra_targets': {'rforest_sub2/0': y_data_sub,
                                               'extrees_sub2/0': y_data_sub}}
    ensemble_model_baikal.fit(x_data_sub, y_data_sub, **fit_params)
    logreg_sub1_coef_retrained = logreg_sub1.coef_
    logreg_ensemble_coef_retrained = logreg_ensemble.coef_

    assert_array_equal(logreg_sub1_coef_original, logreg_sub1_coef_retrained)
    with pytest.raises(AssertionError):
        assert_array_equal(logreg_ensemble_coef_original, logreg_ensemble_coef_retrained)


def make_ensemble_model(n_components, random_state, x_data, y_data):
    # Am admittedly contrived example of a complex Model

    # Sub-model 1
    x1 = Input(name='x1')
    h1 = PCA(n_components=n_components, random_state=random_state, name='pca_sub1')(x1)
    y1 = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state, name='logreg_sub1')(h1)
    submodel1 = Model(x1, y1, name='submodel1')

    # Sub-model 2 (a nested ensemble model)
    x2 = Input(name='x2')
    y2_1 = RandomForestClassifier(random_state=random_state, name='rforest_sub2')(x2)
    y2_2 = ExtraTreesClassifier(random_state=random_state, name='extrees_sub2')(x2)
    features = Stack(axis=1, name='stack_sub2')([y2_1, y2_2])
    y2 = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state, name='logreg_sub2')(features)
    submodel2 = Model(x2, y2, name='submodel2')

    # Ensemble of submodels
    x = Input(name='x')
    y1 = submodel1(x)
    y2 = submodel2(x)
    features = Stack(axis=1, name='stack')([y1, y2])
    y = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=random_state, name='logreg_ensemble')(features)
    ensemble_model_baikal = Model(x, y, name='ensemble')

    fit_params = {'extra_targets': {y1: y_data, y2: y_data},
                  'submodel2__extra_targets': {y2_1: y_data, y2_2: y_data}}
    ensemble_model_baikal.fit(x_data, y_data, **fit_params)

    return ensemble_model_baikal
