from baikal.core.data import Data
from baikal.core.step import Input, Step

from fixtures import sklearn_classifier_step, sklearn_transformer_step, teardown


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

    def test_call(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y = sklearn_classifier_step()(x)

        assert isinstance(y, Data)
        assert (1,) == y.shape
        assert 'LogisticRegression_0/0' == y.name

    def test_call_with_two_inputs(self, teardown):
        class MIMOStep(Step):
            def build_output_shapes(self, input_shapes):
                return [(1,), (1,)]

        x0 = Input((10,), name='x')
        x1 = Input((10,), name='x')
        y0, y1 = MIMOStep()([x0, x1])

        assert isinstance(y0, Data)
        assert isinstance(y1, Data)
        assert (1,) == y0.shape
        assert (1,) == y1.shape
        assert 'MIMOStep_0/0' == y0.name
        assert 'MIMOStep_0/1' == y1.name

    def test_instantiate_two_without_name(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y0 = sklearn_classifier_step()(x)
        y1 = sklearn_classifier_step()(x)

        assert 'LogisticRegression_0/0' == y0.name
        assert 'LogisticRegression_1/0' == y1.name
