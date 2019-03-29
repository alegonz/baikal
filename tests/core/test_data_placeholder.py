from baikal.core.data_placeholder import DataPlaceholder
from baikal.core.step import Step


def test_repr():
    step = Step(name='some-step')
    data_placeholder = DataPlaceholder(step=step, name='some-step/0')
    assert 'DataPlaceholder(step=Step(name=some-step), name=some-step/0)' == repr(data_placeholder)
