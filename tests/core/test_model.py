from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from sklearn import datasets
import sklearn.decomposition
import sklearn.ensemble
from sklearn.exceptions import NotFittedError
import sklearn.linear_model

from baikal.core.model import Model
from baikal.core.step import Input
from baikal.steps.merge import Stack

from fixtures import teardown
from sklearn_steps import LogisticRegression, RandomForestClassifier, PCA
from dummy_steps import DummySISO, DummySIMO, DummyMISO, DummyMIMO, DummyWithoutTransform

iris = datasets.load_iris()


@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def dummy_model_data_placeholders_and_arrays():
    x1 = Input((1,), name='x1')
    x2 = Input((1,), name='x2')

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
    x1 = Input((2,), name='x1')
    x2 = Input((2,), name='x2')
    y1 = LogisticRegression()(x1)
    y2 = PCA()(x2)
    model = Model([x1, x2], [y1, y2])

    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_target_data = iris.target

    # ------ Correct calls. Should not raise errors.
    # Call with lists
    model.fit([x1_data, x2_data], [y1_target_data, None])

    # Call with dicts (data_placeholder keys)
    model.fit({x1: x1_data, x2: x2_data}, {y1: y1_target_data, y2: None})

    # Call with dicts (name (str) keys)
    model.fit({'x1': x1_data, 'x2': x2_data}, {'LogisticRegression_0/0': y1_target_data, 'PCA_0/0': None})

    # ------ Missing input
    # Call with lists
    with pytest.raises(ValueError):
        model.fit([x1_data], [y1_target_data, None])

    # Call with dicts (data_placeholder keys)
    with pytest.raises(ValueError):
        model.fit({x1: x1_data}, {y1: y1_target_data, y2: None})

    # Call with dicts (name (str) keys)
    with pytest.raises(ValueError):
        model.fit({'x1': x1_data}, {'LogisticRegression_0/0': y1_target_data, 'PCA_0/0': None})

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
        model.fit({'x1': x1_data, 'x3': x2_data}, {'LogisticRegression_0/0': y1_target_data, 'PCA_0/0': None})

    # ------ Non-existing outputs
    with pytest.raises(ValueError):
        model.fit({'x1': x1_data, 'x2': x2_data}, {'non-existing-output': y1_target_data, 'PCA_0/0': None})


def test_predict_call(teardown):
    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_target_data = iris.target

    x1 = Input((2,), name='x1')
    x2 = Input((2,), name='x2')
    y1 = LogisticRegression()(x1)
    y2 = PCA()(x2)
    model = Model([x1, x2], [y1, y2])

    model.fit([x1_data, x2_data], [y1_target_data, None])

    # ------ Correct calls. Should not raise errors.
    # Call with list input. Get all outputs.
    y1_pred, y2_pred = model.predict([x1_data, x2_data])

    # Call with dict input (data_placeholder keys). Get all outputs.
    y1_pred, y2_pred = model.predict({x1: x1_data, x2: x2_data})

    # Call with dict input (name (str) keys). Get all outputs.
    y1_pred, y2_pred = model.predict({'x1': x1_data, 'x2': x2_data})

    # Call with list input. Get an specific output. Call with just the needed input
    y1_pred = model.predict(x1_data, 'LogisticRegression_0/0')

    # ------ Missing input
    # Call with list input. Get all outputs.
    with pytest.raises(RuntimeError):
        y1_pred, y2_pred = model.predict(x1_data)

    # Call with dict input (data_placeholder keys). Get all outputs.
    with pytest.raises(RuntimeError):
        y1_pred, y2_pred = model.predict({x1: x1_data})

    # Call with dict input (name (str) keys). Get all outputs.
    with pytest.raises(RuntimeError):
        y1_pred, y2_pred = model.predict({'x1': x1_data})

    # ------ Non-existing inputs
    with pytest.raises(ValueError):
        y1_pred, y2_pred = model.predict({'x1': x1_data, 'x3': x2_data})

    # ------ Non-existing outputs
    with pytest.raises(ValueError):
        y1_pred, y2_pred = model.predict({'x1': x1_data, 'x2': x2_data}, ['non-existing-output', 'PCA_0/0'])

    # ------ Unnecessary inputs
    with pytest.raises(RuntimeError):
        y1_pred, y2_pred = model.predict({'x1': x1_data, 'x2': x2_data}, 'PCA_0/0')

    # ------ Duplicated outputs
    with pytest.raises(ValueError):
        y1_pred, y2_pred = model.predict([x1_data, x2_data],
                                         ['LogisticRegression_0/0', 'LogisticRegression_0/0', 'PCA_0/0'])


def test_steps_cache(teardown):
    x1_data = iris.data[:, :2]
    x2_data = iris.data[:, 2:]
    y1_target_data = iris.target

    x1 = Input((2,), name='x1')
    x2 = Input((2,), name='x2')
    y1 = LogisticRegression(name='y1')(x1)
    y2 = PCA(name='y2')(x2)

    model = Model([x1, x2], [y1, y2])
    assert 0 == model._steps_cache_info.hits and 1 == model._steps_cache_info.misses

    model.fit([x1_data, x2_data], [y1_target_data, None])
    assert 1 == model._steps_cache_info.hits and 1 == model._steps_cache_info.misses

    model.fit({x1: x1_data, x2: x2_data}, {y1: y1_target_data, y2: None})
    assert 2 == model._steps_cache_info.hits and 1 == model._steps_cache_info.misses

    model.predict({'x1': x1_data, 'x2': x2_data}, ['y2/0', 'y1/0'])
    assert 3 == model._steps_cache_info.hits and 1 == model._steps_cache_info.misses

    model.predict([x1_data, x2_data])
    assert 4 == model._steps_cache_info.hits and 1 == model._steps_cache_info.misses

    model.predict(x1_data, 'y1/0')
    assert 4 == model._steps_cache_info.hits and 2 == model._steps_cache_info.misses

    model.predict(x1_data, 'y1/0')
    assert 5 == model._steps_cache_info.hits and 2 == model._steps_cache_info.misses


def test_multiedge(teardown):
    x = Input((1,), name='x')
    z1, z2 = DummySIMO()(x)
    y = DummyMISO()([z1, z2])
    model = Model(x, y)

    X_data = np.array([[1], [2]])
    y_out = model.predict(X_data)

    assert_array_equal(y_out, np.array([[2], [4]]))


def test_instantiation_with_wrong_input_type(teardown):
    x = Input((10,), name='x')
    y = DummySISO()(x)

    x_wrong = np.zeros((10,))
    with pytest.raises(ValueError):
        Model(x_wrong, y)


def test_instantiation_with_steps_with_duplicated_names(teardown):
    x = Input((10,), name='x')
    x = DummySISO(name='duplicated-name')(x)
    y = DummySISO(name='duplicated-name')(x)

    with pytest.raises(RuntimeError):
        Model(x, y)


def test_lazy_model(teardown):
    X_data = np.array([[1, 2], [3, 4]])

    x = Input((2,), name='x')
    model = Model(x, x)
    model.fit(X_data)
    X_pred = model.predict(X_data)

    assert_array_equal(X_pred, X_data)


def test_fit_and_predict_model_with_no_fittable_steps(teardown):
    X1_data = np.array([[1, 2], [3, 4]])
    X2_data = np.array([[5, 6], [7, 8]])
    y_expected = np.array([[12, 16], [20, 24]])

    x1 = Input((2,), name='x1')
    x2 = Input((2,), name='x2')
    z = DummyMISO()([x1, x2])
    y = DummySISO()(z)

    model = Model([x1, x2], y)
    model.fit([X1_data, X2_data])  # nothing to fit
    y_pred = model.predict([X1_data, X2_data])

    assert_array_equal(y_pred, y_expected)


def test_predict_with_not_fitted_steps(teardown):
    X_data = iris.data

    x = Input((4,), name='x')
    xt = PCA(n_components=2)(x)
    y = LogisticRegression(multi_class='multinomial', solver='lbfgs')(xt)

    model = Model(x, y)
    with pytest.raises(NotFittedError):
        model.predict(X_data)


def test_predict_using_step_without_transform(teardown):
    X_data = np.array([[1], [2]])

    x = Input((1,), name='x')
    y = DummyWithoutTransform()(x)

    model = Model(x, y)
    with pytest.raises(TypeError):
        model.predict(X_data)


def test_fit_pipeline(teardown):
    X_data = iris.data
    y_data = iris.target

    x = Input((4,), name='x')
    xt = PCA(n_components=2)(x)
    y = LogisticRegression(multi_class='multinomial', solver='lbfgs')(xt)

    model = Model(x, y)
    model.fit(X_data, y_data)
    assert xt.step.fitted and y.step.fitted


def test_fit_predict_pipeline(teardown):
    X_data = iris.data
    y_data = iris.target

    # baikal way
    x = Input((4,), name='x')
    xt = PCA(n_components=2)(x)
    y = LogisticRegression(multi_class='multinomial', solver='lbfgs')(xt)

    model = Model(x, y)
    model.fit(X_data, y_data)
    y_pred_baikal = model.predict(X_data)

    # traditional way
    pca = sklearn.decomposition.PCA(n_components=2)
    logreg = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    X_data_transformed = pca.fit_transform(X_data)
    logreg.fit(X_data_transformed, y_data)
    y_pred_traditional = logreg.predict(X_data_transformed)

    assert_array_equal(y_pred_baikal, y_pred_traditional)


def test_fit_predict_ensemble(teardown):
    X_data = iris.data
    y_data = iris.target

    # baikal way
    x = Input((4,), name='x')
    y1 = LogisticRegression(multi_class='multinomial', solver='lbfgs')(x)
    y2 = RandomForestClassifier(random_state=123)(x)
    features = Stack(axis=1)([y1, y2])
    y = LogisticRegression(multi_class='multinomial', solver='lbfgs')(features)

    model = Model(x, y)
    model.fit(X_data, {y: y_data, y1: y_data, y2: y_data})
    y_pred_baikal = model.predict(X_data)

    # traditional way
    logreg = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logreg.fit(X_data, y_data)
    logreg_pred = logreg.predict(X_data)

    random_forest = sklearn.ensemble.RandomForestClassifier(random_state=123)
    random_forest.fit(X_data, y_data)
    random_forest_pred = random_forest.predict(X_data)

    features = np.stack([logreg_pred, random_forest_pred], axis=1)

    ensemble = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    ensemble.fit(features, y_data)
    y_pred_traditional = ensemble.predict(features)

    assert_array_equal(y_pred_baikal, y_pred_traditional)
