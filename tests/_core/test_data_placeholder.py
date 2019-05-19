from baikal import Step
from baikal._core.data_placeholder import DataPlaceholder


def test_repr():
    step = Step(name='some-step')
    data_placeholder = DataPlaceholder(step=step, name='some-step/0')
    assert "DataPlaceholder(step=Step(name='some-step', trainable=True, function=None), name='some-step/0')" == repr(data_placeholder)
