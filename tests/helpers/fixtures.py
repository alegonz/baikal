import pytest

from baikal import Step


@pytest.fixture
def teardown():
    yield
    Step.clear_names()
