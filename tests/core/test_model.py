import numpy as np
from numpy.testing import assert_array_equal
import pytest
from sklearn import datasets
import sklearn.decomposition
import sklearn.linear_model

from baikal.core.model import Model
from baikal.core.step import Input

from fixtures import sklearn_classifier_step, sklearn_transformer_step, teardown


class TestModel:
    def test_instantiation(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y = sklearn_classifier_step()(x)
        model = Model(x, y)

    def test_instantiation_with_wrong_input_type(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y = sklearn_classifier_step()(x)

        x_wrong = np.zeros((10,))
        with pytest.raises(ValueError):
            model = Model(x_wrong, y)

    def test_fit_classifier(self, sklearn_classifier_step, teardown):
        # Based on the example in
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
        iris = datasets.load_iris()

        X_data = iris.data[:, :2]  # we only take the first two features.
        y_data = iris.target

        x = Input((2,), name='x')
        y = sklearn_classifier_step(multi_class='multinomial', solver='lbfgs')(x)

        model = Model(x, y)

        # Style 1: pass data as in instantiation
        model.fit(X_data, y_data)
        assert y.step.fitted

    def test_fit_transformer(self, sklearn_transformer_step, teardown):
        iris = datasets.load_iris()
        X_data = iris.data

        x = Input((4,), name='x')
        xt = sklearn_transformer_step(n_components=2)(x)

        model = Model(x, xt)
        model.fit(X_data)
        assert xt.step.fitted

    def test_fit_pipeline(self, sklearn_classifier_step, sklearn_transformer_step, teardown):
        iris = datasets.load_iris()
        X_data = iris.data
        y_data = iris.target

        x = Input((4,), name='x')
        xt = sklearn_transformer_step(n_components=2)(x)
        y = sklearn_classifier_step(multi_class='multinomial', solver='lbfgs')(xt)

        model = Model(x, y)
        model.fit(X_data, y_data)
        assert xt.step.fitted and y.step.fitted

    def test_fit_predict_pipeline(self, sklearn_classifier_step, sklearn_transformer_step, teardown):
        iris = datasets.load_iris()
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

    def test_lazy_model(self, teardown):
        X_data = np.array([[1, 2], [3, 4]])

        x = Input((2,), name='x')
        model = Model(x, x)
        model.fit(X_data)
        X_pred = model.predict(X_data)

        assert_array_equal(X_pred, X_data)

    def test_missing_input(self, teardown):
        x1 = Input((2,), name='x1')
        x2 = Input((2,), name='x2')

        with pytest.raises(ValueError):
            model = Model(x1, x2)
