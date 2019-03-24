from baikal.core.data_placeholder import DataPlaceholder
from baikal.core.step import Step


def test_repr():
    step = Step(name='some-step')
    data_placeholder = DataPlaceholder(shape=(2, 4), step=step, name='some-step/0')
    assert 'DataPlaceholder(shape=(2, 4), step=Step(name=some-step), name=some-step/0)' == repr(data_placeholder)
