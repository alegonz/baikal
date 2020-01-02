from contextlib import contextmanager
from functools import partial

import pytest

from baikal import Input, Step
from baikal._core.data_placeholder import DataPlaceholder
from baikal._core.step import InputStep

from tests.helpers.fixtures import teardown
from tests.helpers.dummy_steps import DummyMIMO, DummySISO, DummyEstimator
from tests.helpers.sklearn_steps import LogisticRegression, PCA


@contextmanager
def does_not_raise():
    yield


class TestInput:
    def test_instantiation(self, teardown):
        x0 = Input()

        assert isinstance(x0, DataPlaceholder)
        assert x0.name == "InputStep_0"

    def test_instantiate_two_without_name(self, teardown):
        x0 = Input()
        x1 = Input()

        assert x0.name == "InputStep_0"
        assert x1.name == "InputStep_1"


class TestInputStep:
    def test_repr(self):
        step = InputStep(name="x1")
        assert repr(step) == "InputStep(name='x1')"


class TestStep:
    def test_instantiate_two_without_name(self, teardown):
        lr0 = LogisticRegression()
        lr1 = LogisticRegression()

        assert lr0.name == "LogisticRegression_0"
        assert lr1.name == "LogisticRegression_1"

    def test_instantiate_with_invalid_compute_func(self):
        class DummyStep(Step):
            def somefunc(self, X):
                pass

        class DummyStepWithPredict(Step):
            def predict(self, X):
                pass

        class DummyStepWithTransform(Step):
            def transform(self, X):
                pass

        x = Input()

        with pytest.raises(ValueError):
            step = DummyStep()
            step(x, compute_func=None)

        step = DummyStep()
        step(x, compute_func="somefunc")
        assert step.compute_func == step.somefunc

        def anotherfunc():
            pass

        step = DummyStep()
        step(x, compute_func=anotherfunc)
        assert step.compute_func == anotherfunc

        step = DummyStepWithPredict()
        step(x)
        assert step.compute_func == step.predict

        step = DummyStepWithTransform()
        step(x)
        assert step.compute_func == step.transform

    # Below tests are parametrized to take two kind of fittable steps:
    # - step that requires y (e.g. Logistic Regression)
    # - step that does not require y (e.g. PCA)

    @pytest.mark.parametrize("step_class", [LogisticRegression, PCA])
    @pytest.mark.parametrize("trainable", [True, False])
    def test_call_without_targets(self, step_class, trainable, teardown):
        x = Input()
        step_class()(x, trainable=trainable)

    @pytest.mark.parametrize("step_class", [LogisticRegression, PCA])
    @pytest.mark.parametrize(
        "trainable,expectation",
        [(True, does_not_raise), (False, partial(pytest.warns, UserWarning))],
    )
    def test_call_with_targets(self, step_class, trainable, expectation, teardown):
        x = Input()
        y_t = Input()
        with expectation():
            step_class()(x, y_t, trainable=trainable)

    def test_call_without_targets_without_fit_method(self, teardown):
        x = Input()
        DummySISO()(x)

    def test_call_with_targets_without_fit_method(self, teardown):
        x = Input()
        y_t = Input()
        with pytest.raises(RuntimeError):
            DummySISO()(x, y_t)

    def test_call_with_two_inputs(self, teardown):
        x0 = Input()
        x1 = Input()
        y0, y1 = DummyMIMO()([x0, x1])

        assert isinstance(y0, DataPlaceholder)
        assert isinstance(y1, DataPlaceholder)
        assert y0.name == "DummyMIMO_0/0"
        assert y1.name == "DummyMIMO_0/1"

    def test_repr(self):
        class DummyStep(Step):
            def somefunc(self, X):
                pass

        step = DummyStep(name="some-step")
        assert repr(step) == "DummyStep(name='some-step', n_outputs=1)"

        # TODO: Add test for sklearn step

    def test_get_params(self, teardown):
        step = DummyEstimator()
        params = step.get_params()
        expected = {"x": 123, "y": "abc"}
        assert params == expected

    def test_set_params(self, teardown):
        step = DummyEstimator()

        new_params_wrong = {"non_existent_param": 42}
        with pytest.raises(ValueError):
            step.set_params(**new_params_wrong)

        new_params = {"x": 456}
        step.set_params(**new_params)
        params = step.get_params()
        expected = {"x": 456, "y": "abc"}
        assert params == expected
