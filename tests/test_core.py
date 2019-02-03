import pytest
import numpy as np
import sklearn.linear_model.logistic

from baikal.core import (default_graph, Node,
                         Input, ProcessorMixin, Data, Model)


@pytest.fixture
def teardown():
    yield
    Node.clear_names()
    default_graph.clear()


class TestInput:
    def test_instantiation(self, teardown):
        x0 = Input((10,))  # a 10-dimensional feature vector

        assert isinstance(x0, Data)
        assert (10,) == x0.shape
        assert 'InputNode_0/0' == x0.name

    def test_instantiate_two_with_same_name(self, teardown):
        x0 = Input((5,), name='x')
        x1 = Input((2,), name='x')

        assert 'x_0/0' == x0.name
        assert 'x_1/0' == x1.name

    def test_instantiate_two_without_name(self, teardown):
        x0 = Input((5,))
        x1 = Input((2,))

        assert 'InputNode_0/0' == x0.name
        assert 'InputNode_1/0' == x1.name


@pytest.fixture
def extended_sklearn_class():
    class LogisticRegression(ProcessorMixin, sklearn.linear_model.logistic.LogisticRegression):
        def build_output_shapes(self, input_shapes):
            return [(1,)]

        def compute(self, x):
            return self.predict(x)

    return LogisticRegression


class TestProcessorMixin:

    def test_call(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y = extended_sklearn_class()(x)

        assert isinstance(y, Data)
        assert (1,) == y.shape
        assert 'LogisticRegression_0/0' == y.name

    def test_call_with_two_inputs(self, teardown):
        class MIMOProcessor(ProcessorMixin):
            def build_output_shapes(self, input_shapes):
                return [(1,), (1,)]

        x0 = Input((10,), name='x')
        x1 = Input((10,), name='x')
        y0, y1 = MIMOProcessor()([x0, x1])

        assert isinstance(y0, Data)
        assert isinstance(y1, Data)
        assert (1,) == y0.shape
        assert (1,) == y1.shape
        assert 'MIMOProcessor_0/0' == y0.name
        assert 'MIMOProcessor_0/1' == y1.name

    def test_instantiate_two_with_same_name(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y0 = extended_sklearn_class(name='myclassifier')(x)
        y1 = extended_sklearn_class(name='myclassifier')(x)

        assert 'myclassifier_0/0' == y0.name
        assert 'myclassifier_1/0' == y1.name

    def test_instantiate_two_without_name(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y0 = extended_sklearn_class()(x)
        y1 = extended_sklearn_class()(x)

        assert 'LogisticRegression_0/0' == y0.name
        assert 'LogisticRegression_1/0' == y1.name


class TestModel:
    def test_instantiation(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y = extended_sklearn_class()(x)
        model = Model(x, y)

    def test_instantiation_with_wrong_input_type(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y = extended_sklearn_class()(x)

        x_wrong = np.zeros((10,))
        with pytest.raises(ValueError):
            model = Model(x_wrong, y)
