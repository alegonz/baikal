from baikal import Step
from baikal._core.data_placeholder import DataPlaceholder


def test_repr():
    class DummyStep(Step):
        def somefunc(self, X):
            pass

    step = DummyStep(name="some-step")
    data_placeholder = DataPlaceholder(step, 1, "some-step/1/0")
    expected_repr = "DataPlaceholder(step=DummyStep(name='some-step', n_outputs=1), port=1, name='some-step/1/0')"
    assert repr(data_placeholder) == expected_repr
