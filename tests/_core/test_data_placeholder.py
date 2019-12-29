from baikal import Step
from baikal._core.data_placeholder import DataPlaceholder


def test_repr():
    class DummyStep(Step):
        def somefunc(self, X):
            pass

    step = DummyStep(name="some-step", function="somefunc")
    data_placeholder = DataPlaceholder(step=step, name="some-step/0")
    expected_repr = (
        "DataPlaceholder(step=DummyStep(name='some-step', "
        "function='somefunc', n_outputs=1, trainable=True), "
        "name='some-step/0')"
    )
    assert repr(data_placeholder) == expected_repr
