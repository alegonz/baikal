import pytest
import sklearn.linear_model.logistic

from baikal.core import default_graph, Node, ProcessorMixin, Input, InputNode, Data


@pytest.fixture
def teardown():
    yield
    Node.clear_names()
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


class TestProcessorMixin:

    @classmethod
    @pytest.fixture(scope='class')
    def extended_sklearn_class(cls):
        class LogisticRegression(ProcessorMixin, sklearn.linear_model.logistic.LogisticRegression):
            # TODO: add compute method
            def build_outputs(self, inputs):
                return Data((1,), self)

        return LogisticRegression

    def test_call(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y = extended_sklearn_class()(x)

        assert isinstance(y, Data)
        assert (1,) == y.shape
        assert 'default/LogisticRegression_0/0' == y.name

    def test_call_with_two_inputs(self, teardown):
        class MIMOProcessor(ProcessorMixin):
            # TODO: add compute method
            def build_outputs(self, inputs):
                return Data((1,), self, 0), Data((1,), self, 1)

        x0 = Input((10,), name='x')
        x1 = Input((10,), name='x')
        y0, y1 = MIMOProcessor()([x0, x1])

        assert isinstance(y0, Data)
        assert isinstance(y1, Data)
        assert (1,) == y0.shape
        assert (1,) == y1.shape
        assert 'default/MIMOProcessor_0/0' == y0.name
        assert 'default/MIMOProcessor_0/1' == y1.name

    def test_instantiate_two_with_same_name(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y0 = extended_sklearn_class(name='myclassifier')(x)
        y1 = extended_sklearn_class(name='myclassifier')(x)

        assert 'default/myclassifier_0/0' == y0.name
        assert 'default/myclassifier_1/0' == y1.name

    def test_instantiate_two_without_name(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y0 = extended_sklearn_class()(x)
        y1 = extended_sklearn_class()(x)

        assert 'default/LogisticRegression_0/0' == y0.name
        assert 'default/LogisticRegression_1/0' == y1.name
