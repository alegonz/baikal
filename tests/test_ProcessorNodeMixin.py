import pytest
import sklearn.linear_model

from baikal.core import ProcessorNodeMixin


@pytest.fixture
def teardown_core():
    yield
    ProcessorNodeMixin.clear_processors()


class LogisticRegression(ProcessorNodeMixin, sklearn.linear_model.LogisticRegression):
    pass


class TestProcessorNodeMixin:
    def test_instantiate_with_name(self, teardown_core):
        LogisticRegression(name='lr0')

    def test_instantiate_two_with_same_name(self, teardown_core):
        LogisticRegression(name='lr0')

        with pytest.raises(ValueError):
            LogisticRegression(name='lr0')

    def test_instantiate_without_name(self, teardown_core):
        lr = LogisticRegression()
        assert isinstance(lr.name, str) and len(lr.name) > 0

    def test_instantiate_two_without_name(self, teardown_core):
        lr0 = LogisticRegression()
        lr1 = LogisticRegression()
        assert lr0.name == 'LR_0' and lr1.name == 'LR_1'
