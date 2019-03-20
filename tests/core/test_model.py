import numpy as np
from numpy.testing import assert_array_equal
import pytest
from sklearn import datasets
import sklearn.decomposition
from sklearn.exceptions import NotFittedError
import sklearn.linear_model

from baikal.core.model import Model
from baikal.core.step import Input

from fixtures import sklearn_classifier_step, sklearn_transformer_step, teardown
from dummy_steps import DummySISO, DummySIMO, DummyMISO, DummyMIMO, DummyWithoutTransform

iris = datasets.load_iris()


class TestModel:
    @pytest.mark.parametrize('inputs,outputs,error_warning',
                             [(['x1'], ['z1'], None),
                              (['x1'], ['x1'], None),
                              (['x1', 'x2'], ['z5', 'z6'], None),
                              (['x1', 'x2'], ['z1', 'z2'], None),
                              (['z3', 'z4'], ['z5'], None),
                              (['x1'], ['x2'], ValueError),
                              (['z1'], ['z4'], ValueError),
                              (['z1', 'z2'], ['z5'], ValueError),
                              (['x1', 'x2'], ['z1'], RuntimeWarning),
                              (['z1', 'z2', 'x1'], ['z4'], RuntimeWarning),
                              (['z1', 'z2', 'x2'], ['z4'], RuntimeWarning)])
    def test_instantiation(self, inputs, outputs, error_warning, teardown):
        x1 = Input((1,), name='x1')
        x2 = Input((1,), name='x2')

        z1 = DummySISO()(x1)
        z2, z3 = DummySIMO()(x2)
        z4 = DummyMISO()([z1, z2])
        z5, z6 = DummyMIMO()([z4, z3])

        data = {'x1': x1, 'x2': x2,
                'z1': z1, 'z2': z2,
                'z3': z3, 'z4': z4,
                'z5': z5, 'z6': z6}

        inputs = [data[i] for i in inputs]
        outputs = [data[o] for o in outputs]

        if error_warning is ValueError:
            with pytest.raises(error_warning):
                Model(inputs, outputs)

        elif error_warning is RuntimeWarning:
            with pytest.warns(error_warning):
                Model(inputs, outputs)

        else:
            Model(inputs, outputs)

    def test_multiedge(self, teardown):
        x = Input((1,), name='x')
        z1, z2 = DummySIMO()(x)
        y = DummyMISO()([z1, z2])
        model = Model(x, y)

        X_data = np.array([[1], [2]])
        y_out = model.predict(X_data)

        assert_array_equal(y_out, np.array([[2], [4]]))

    def test_instantiation_with_wrong_input_type(self, teardown):
        x = Input((10,), name='x')
        y = DummySISO()(x)

        x_wrong = np.zeros((10,))
        with pytest.raises(ValueError):
            Model(x_wrong, y)

    def test_instantiation_with_steps_with_duplicated_names(self, teardown):
        x = Input((10,), name='x')
        x = DummySISO(name='duplicated-name')(x)
        y = DummySISO(name='duplicated-name')(x)

        with pytest.raises(RuntimeError):
            Model(x, y)

    def test_lazy_model(self, teardown):
        X_data = np.array([[1, 2], [3, 4]])

        x = Input((2,), name='x')
        model = Model(x, x)
        model.fit(X_data)
        X_pred = model.predict(X_data)

        assert_array_equal(X_pred, X_data)

    def test_predict_with_missing_input(self, teardown):
        x1 = Input((1,), name='x1')
        x2 = Input((1,), name='x2')
        y = DummyMISO()([x1, x2])

        model = Model([x1, x2], y)

        x1_data = np.array([[1], [2]])
        with pytest.raises(ValueError):
            model.predict(x1_data)

    def test_predict_with_nonexisting_outputs(self, teardown):
        # Should raise ValueError
        pass

    def test_fit_and_predict_model_with_no_fittable_steps(self, teardown):
        X1_data = np.array([[1, 2], [3, 4]])
        X2_data = np.array([[5, 6], [7, 8]])
        y_expected = np.array([[12, 16], [20, 24]])

        x1 = Input((2,), name='x1')
        x2 = Input((2,), name='x2')
        z = DummyMISO()([x1, x2])
        y = DummySISO()(z)

        model = Model([x1, x2], y)
        model.fit([X1_data, X2_data])
        y_pred = model.predict([X1_data, X2_data])

        assert_array_equal(y_pred, y_expected)

    def test_predict_with_not_fitted_steps(self, sklearn_classifier_step, sklearn_transformer_step, teardown):
        X_data = iris.data

        x = Input((4,), name='x')
        xt = sklearn_transformer_step(n_components=2)(x)
        y = sklearn_classifier_step(multi_class='multinomial', solver='lbfgs')(xt)

        model = Model(x, y)
        with pytest.raises(NotFittedError):
            model.predict(X_data)

    def test_predict_using_step_without_transform(self, teardown):
        X_data = np.array([[1], [2]])

        x = Input((1,), name='x')
        y = DummyWithoutTransform()(x)

        model = Model(x, y)
        with pytest.raises(TypeError):
            model.predict(X_data)

    def test_fit_classifier(self, sklearn_classifier_step, teardown):
        # Based on the example in
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
        X_data = iris.data[:, :2]  # we only take the first two features.
        y_data = iris.target

        x = Input((2,), name='x')
        y = sklearn_classifier_step(multi_class='multinomial', solver='lbfgs')(x)

        model = Model(x, y)

        # Style 1: pass data as in instantiation
        model.fit(X_data, y_data)
        assert y.step.fitted

    def test_fit_transformer(self, sklearn_transformer_step, teardown):
        X_data = iris.data

        x = Input((4,), name='x')
        xt = sklearn_transformer_step(n_components=2)(x)

        model = Model(x, xt)
        model.fit(X_data)
        assert xt.step.fitted

    def test_fit_pipeline(self, sklearn_classifier_step, sklearn_transformer_step, teardown):
        X_data = iris.data
        y_data = iris.target

        x = Input((4,), name='x')
        xt = sklearn_transformer_step(n_components=2)(x)
        y = sklearn_classifier_step(multi_class='multinomial', solver='lbfgs')(xt)

        model = Model(x, y)
        model.fit(X_data, y_data)
        assert xt.step.fitted and y.step.fitted

    def test_fit_predict_pipeline(self, sklearn_classifier_step, sklearn_transformer_step, teardown):
        X_data = iris.data
        y_data = iris.target

        # baikal way
        x = Input((4,), name='x')
        xt = sklearn_transformer_step(n_components=2)(x)
        y = sklearn_classifier_step(multi_class='multinomial', solver='lbfgs')(xt)

        model = Model(x, y)
        model.fit(X_data, y_data)
        y_pred_baikal = model.predict(X_data)

        # traditional way
        pca = sklearn.decomposition.PCA(n_components=2)
        logreg = sklearn.linear_model.logistic.LogisticRegression(multi_class='multinomial', solver='lbfgs')
        X_data_transformed = pca.fit_transform(X_data)
        logreg.fit(X_data_transformed, y_data)
        y_pred_traditional = logreg.predict(X_data_transformed)

        assert_array_equal(y_pred_baikal, y_pred_traditional)
