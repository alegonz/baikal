import pytest
import sklearn.linear_model.logistic

from baikal.core import default_graph, ProcessorMixin, Input, InputNode, Data


@pytest.fixture
def teardown():
    yield
    InputNode._names.clear()
    default_graph.clear()


class TestInput:
    def test_instantiation(self, teardown):
        x0 = Input((10,))  # a 10-dimensional feature vector

        node = default_graph.nodes[0]
        assert isinstance(node, InputNode)
        assert 'default/InputNode_0' == node.name
        assert isinstance(x0, Data)
        assert (10,) == x0.shape
        assert 'default/InputNode_0/0' == x0.name

    def test_instantiate_two_with_same_name(self, teardown):
        x0 = Input((5,), name='x')
        x1 = Input((2,), name='x')
        assert 'default/x_0/0' == x0.name
        assert 'default/x_1/0' == x1.name

    def test_instantiate_two_without_name(self, teardown):
        x0 = Input((5,))
        x1 = Input((2,))
        assert 'default/InputNode_0/0' == x0.name
        assert 'default/InputNode_1/0' == x1.name


class LogisticRegression(ProcessorMixin, sklearn.linear_model.logistic.LogisticRegression):
    def build_outputs(self, inputs):
        return Data((1,), self)


class TestProcessorMixin:
    def test_call(self, teardown):
        x = Input((10,), name='x')
        y = LogisticRegression()(x)

        node = default_graph.nodes[0]
        assert isinstance(node, LogisticRegression)
        assert 'default/LogisticRegression_0' == node.name
        assert isinstance(y, Data)
        assert (1,) == y.shape
        assert 'default/LogisticRegression_0/0' == y.name

    def test_instantiate_two_with_same_name(self, teardown):
        x = Input((10,), name='x')
        y0 = LogisticRegression(name='myclassifier')(x)
        y1 = LogisticRegression(name='myclassifier')(x)
        assert 'default/myclassifier_0/0' == y0.name
        assert 'default/myclassifier_1/0' == y1.name

    def test_instantiate_two_without_name(self, teardown):
        x = Input((10,), name='x')
        y0 = LogisticRegression()(x)
        y1 = LogisticRegression()(x)
        assert 'default/LogisticRegression_0/0' == y0.name
        assert 'default/LogisticRegression_1/0' == y1.name
