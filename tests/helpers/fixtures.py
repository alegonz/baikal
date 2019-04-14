import pytest

from baikal.core.digraph import default_graph
from baikal.core.step import Step


@pytest.fixture
def teardown():
    yield
    Step.clear_names()
    default_graph.clear()
