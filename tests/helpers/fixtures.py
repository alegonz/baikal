import pytest

from baikal import Step
from baikal._core.digraph import default_graph


@pytest.fixture
def teardown():
    yield
    Step.clear_names()
    default_graph.clear()
