from baikal.core.data import Data
from baikal.core.step import Step


def test_repr():
    step = Step(name='some-step')
    data = Data(shape=(2, 4), step=step, name='some-step/0')
    assert 'Data(shape=(2, 4), step=Step(name=some-step), name=some-step/0)' == repr(data)
