import pytest

from baikal.core.data_placeholder import DataPlaceholder
from baikal.core.step import Input, Step

from tests.helpers.fixtures import teardown
from tests.helpers.dummy_steps import DummyMIMO
from tests.helpers.sklearn_steps import LogisticRegression


class TestInput:
    def test_instantiation(self, teardown):
        x0 = Input()

        assert isinstance(x0, DataPlaceholder)
        assert 'InputStep_0' == x0.name

    def test_instantiate_two_without_name(self, teardown):
        x0 = Input()
        x1 = Input()

        assert 'InputStep_0' == x0.name
        assert 'InputStep_1' == x1.name


class TestStep:

    def test_call(self, teardown):
        x = Input(name='x')
        y = LogisticRegression()(x)

        assert isinstance(y, DataPlaceholder)
        assert 'LogisticRegression_0/0' == y.name

    def test_call_with_two_inputs(self, teardown):
        x0 = Input(name='x')
        x1 = Input(name='x')
        y0, y1 = DummyMIMO()([x0, x1])

        assert isinstance(y0, DataPlaceholder)
        assert isinstance(y1, DataPlaceholder)
        assert 'DummyMIMO_0/0' == y0.name
        assert 'DummyMIMO_0/1' == y1.name

    def test_instantiate_two_without_name(self, teardown):
        x = Input(name='x')
        y0 = LogisticRegression()(x)
        y1 = LogisticRegression()(x)

        assert 'LogisticRegression_0/0' == y0.name
        assert 'LogisticRegression_1/0' == y1.name

    def test_repr(self):
        step = Step(name='some-step')
        assert 'Step(name=some-step, trainable=True)' == repr(step)

    def test_get_params(self, teardown):
        step = LogisticRegression()
        params = step.get_params()

        expected = {'C': 1.0,
                    'class_weight': None,
                    'dual': False,
                    'fit_intercept': True,
                    'intercept_scaling': 1,
                    'max_iter': 100,
                    'multi_class': 'warn',
                    'n_jobs': None,
                    'penalty': 'l2',
                    'random_state': None,
                    'solver': 'warn',
                    'tol': 0.0001,
                    'verbose': 0,
                    'warm_start': False}

        assert expected == params

    def test_set_params(self, teardown):
        step = LogisticRegression()

        new_params_wrong = {'non_existent_param': 42}
        with pytest.raises(ValueError):
            step.set_params(**new_params_wrong)

        new_params = {'C': 100.0,
                      'fit_intercept': False,
                      'penalty': 'l1'}

        step.set_params(**new_params)
        params = step.get_params()

        expected = {'C': 100.0,
                    'class_weight': None,
                    'dual': False,
                    'fit_intercept': False,
                    'intercept_scaling': 1,
                    'max_iter': 100,
                    'multi_class': 'warn',
                    'n_jobs': None,
                    'penalty': 'l1',
                    'random_state': None,
                    'solver': 'warn',
                    'tol': 0.0001,
                    'verbose': 0,
                    'warm_start': False}

        assert expected == params