import pytest

import baikal.core
from baikal.core import create_input, Input, Data


@pytest.fixture
def teardown():
    yield
    Input._names.clear()


class TestInput:
    def test_returns_data_instance(self, teardown):
        x0 = create_input()
        assert isinstance(x0, Data)

    def test_input_is_in_default_graph(self, teardown):
        x0 = create_input()
        for node in baikal.core.default_graph:
            if isinstance(node, Input) and node.name == 'default/Input_0':
                assert True
                return
        assert False

    def test_instantiate_two_with_same_name(self, teardown):
        x0 = create_input(name='x')
        x1 = create_input(name='x')
        assert 'default/x_0/0' == x0.name
        assert 'default/x_1/0' == x1.name

    def test_instantiate_two_without_name(self, teardown):
        x0 = create_input()
        x1 = create_input()
        assert 'default/Input_0/0' == x0.name
        assert 'default/Input_1/0' == x1.name
