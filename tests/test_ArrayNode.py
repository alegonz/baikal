import pytest

from baikal.core import ArrayNode


@pytest.fixture
def teardown_core():
    yield
    ArrayNode.clear_arrays()


class TestArrayNode:
    def test_instantiate_with_name(self, teardown_core):
        ArrayNode(name='arr_0')

    def test_instantiate_two_with_same_name(self, teardown_core):
        ArrayNode(name='arr_0')

        with pytest.raises(ValueError):
            ArrayNode(name='arr_0')

    def test_instantiate_without_name(self, teardown_core):
        arr = ArrayNode()
        assert isinstance(arr.name, str) and len(arr.name) > 0

    def test_instantiate_two_without_name(self, teardown_core):
        arr0 = ArrayNode()
        arr1 = ArrayNode()
        assert arr0.name == 'arr_0' and arr1.name == 'arr_1'
