import pytest

from baikal.core.digraph import Node, default_graph


@pytest.fixture
def teardown():
    yield
    Node.clear_names()
    default_graph.clear()
