from baikal.core.data import Data
from baikal.core.step import Input, Step

from dummy_steps import DummyMIMO
from fixtures import teardown
from sklearn_steps import LogisticRegression


class TestInput:
    def test_instantiation(self, teardown):
        x0 = Input((10,))  # a 10-dimensional feature vector

        assert isinstance(x0, Data)
        assert (10,) == x0.shape
        assert 'InputStep_0' == x0.name

    def test_instantiate_two_without_name(self, teardown):
        x0 = Input((5,))
        x1 = Input((2,))

        assert 'InputStep_0' == x0.name
        assert 'InputStep_1' == x1.name


class TestStep:

    def test_call(self, teardown):
        x = Input((10,), name='x')
        y = LogisticRegression()(x)

        assert isinstance(y, Data)
        assert (1,) == y.shape
        assert 'LogisticRegression_0/0' == y.name

    def test_call_with_two_inputs(self, teardown):
        x0 = Input((1,), name='x')
        x1 = Input((1,), name='x')
        y0, y1 = DummyMIMO()([x0, x1])

        assert isinstance(y0, Data)
        assert isinstance(y1, Data)
        assert (1,) == y0.shape
        assert (1,) == y1.shape
        assert 'DummyMIMO_0/0' == y0.name
        assert 'DummyMIMO_0/1' == y1.name

    def test_instantiate_two_without_name(self, teardown):
        x = Input((10,), name='x')
        y0 = LogisticRegression()(x)
        y1 = LogisticRegression()(x)

        assert 'LogisticRegression_0/0' == y0.name
        assert 'LogisticRegression_1/0' == y1.name

    def test_repr(self):
        step = Step(name='some-step')
        assert 'Step(name=some-step)' == repr(step)
