import pytest

from baikal.core import default_graph, Input, InputNode, Data


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
        assert 'default/x_0/0' == x0.name
        assert 'default/x_1/0' == x1.name

    def test_instantiate_two_without_name(self, teardown):
        x0 = Input()
        x1 = Input()
        assert 'default/InputNode_0/0' == x0.name
        assert 'default/InputNode_1/0' == x1.name
