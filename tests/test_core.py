import pytest
import sklearn.linear_model.logistic

from baikal.core import default_graph, ProcessorMixin, Input, InputNode, Data


@pytest.fixture
def teardown():
    yield
    InputNode._names.clear()
    default_graph.clear()


class TestInput:
    def test_returns_data_instance(self, teardown):
        x0 = Input()
        assert isinstance(x0, Data)

    def test_input_is_in_default_graph(self, teardown):
        x0 = Input()
        node = default_graph.nodes[0]
        assert isinstance(node, InputNode)
        assert node.name == 'default/InputNode_0'

    def test_instantiate_two_with_same_name(self, teardown):
        x0 = Input(name='x')
        x1 = Input(name='x')
        # TODO: '/0' at the end is cumbersome and unnecessary
        # TODO: Consider removing the graph name from the node name
        assert 'default/x_0/0' == x0.name
        assert 'default/x_1/0' == x1.name

    def test_instantiate_two_without_name(self, teardown):
        x0 = Input()
        x1 = Input()
        assert 'default/InputNode_0/0' == x0.name
        assert 'default/InputNode_1/0' == x1.name


class LogisticRegression(ProcessorMixin, sklearn.linear_model.logistic.LogisticRegression):
    pass


class TestProcessorMixin:
    def test_takes_and_returns_data_instances(self, teardown):
        x = Input(name='x')
        y = LogisticRegression()(x)
        assert isinstance(y, Data)
        assert y.name == 'default/LogisticRegression_0/0'

    def test_processor_is_in_default_graph(self, teardown):
        x0 = LogisticRegression()
        node = default_graph.nodes[0]
        assert isinstance(node, LogisticRegression)
        assert node.name == 'default/LogisticRegression_0'

    def test_instantiate_two_with_same_name(self, teardown):
        x = Input(name='x')
        y0 = LogisticRegression(name='myclassifier')(x)
        y1 = LogisticRegression(name='myclassifier')(x)
        assert 'default/myclassifier_0/0' == y0.name
        assert 'default/myclassifier_1/0' == y1.name

    def test_instantiate_two_without_name(self, teardown):
        x = Input(name='x')
        y0 = LogisticRegression()(x)
        y1 = LogisticRegression()(x)
        assert 'default/LogisticRegression_0/0' == y0.name
        assert 'default/LogisticRegression_1/0' == y1.name
