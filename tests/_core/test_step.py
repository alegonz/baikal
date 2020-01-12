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
            step(x, compute_func="auto")

        with pytest.raises(ValueError):
            step = DummyStep()
            step(x, compute_func=123)

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

    def test_instantiate_with_invalid_fit_compute_func(self):
        class DummyStepWithoutFit(Step):
            def predict(self, X):
                pass

            def somefunc(self, X):
                pass

        class DummyStepWithFitPredict(Step):
            def predict(self, X):
                pass

            def fit_predict(self, X, y):
                pass

        class DummyStepWithFitTransform(Step):
            def transform(self, X):
                pass

            def fit_transform(self, X, y):
                pass

        x = Input()

        step = DummyStepWithoutFit()
        step(x, fit_compute_func="auto")
        assert step.fit_compute_func is None

        with pytest.raises(ValueError):
            step = DummyStepWithoutFit()
            step(x, fit_compute_func=123)

        step = DummyStepWithoutFit()
        step(x, fit_compute_func="somefunc")
        assert step.fit_compute_func == step.somefunc

        def anotherfunc():
            pass

        step = DummyStepWithoutFit()
        step(x, fit_compute_func=anotherfunc)
        assert step.fit_compute_func == anotherfunc

        step = DummyStepWithFitPredict()
        step(x)
        assert step.fit_compute_func == step.fit_predict

        step = DummyStepWithFitTransform()
        step(x)
        assert step.fit_compute_func == step.fit_transform

        step = DummyStepWithFitTransform()
        step(x, fit_compute_func=None)
        assert step.fit_compute_func is None

    def test_call_with_invalid_input_type(self, teardown):
        with pytest.raises(ValueError):
            LogisticRegression()([[1, 2], [3, 4]])

    def test_call_with_invalid_target_type(self, teardown):
        x = Input()
        with pytest.raises(ValueError):
            LogisticRegression()(x, [0, 1])

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
        assert y0.name == "DummyMIMO_0/0/0"
        assert y1.name == "DummyMIMO_0/0/1"

    def test_call_twice(self, teardown):
        x0 = Input()
        x1 = Input()
        step = DummySISO()
        y0 = step(x0)
        y1 = step(x1)

        assert isinstance(y0, DataPlaceholder)
        assert isinstance(y1, DataPlaceholder)
        assert y0.name == "DummySISO_0/0/0"
        assert y1.name == "DummySISO_0/1/0"

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

    # Below are tests for properties
    @pytest.fixture
    def simple_step(self):
        return DummyEstimator()

    @pytest.fixture
    def shared_step(self):
        return DummyEstimator()

    @pytest.fixture
    def dataplaceholders(self, simple_step, shared_step):
        x1 = Input(name="x1")
        x2 = Input(name="x2")
        y_t = Input(name="y_t")
        y_simple = simple_step(x1, y_t)
        y_shared_1 = shared_step(x1, y_t)
        y_shared_2 = shared_step(
            x2,
            compute_func="predict_proba",
            fit_compute_func="fit_predict_proba",
            trainable=False,
        )
        return x1, x2, y_t, y_simple, y_shared_1, y_shared_2

    def test_inputs(self, simple_step, shared_step, dataplaceholders, teardown):
        x1 = dataplaceholders[0]
        assert simple_step.inputs == [x1]

        with pytest.raises(AttributeError):
            shared_step.inputs

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            LogisticRegression().inputs

    def test_outputs(self, simple_step, shared_step, dataplaceholders, teardown):
        *_, y_simple, y_shared_1, y_shared_2 = dataplaceholders
        assert simple_step.outputs == [y_simple]

        with pytest.raises(AttributeError):
            shared_step.outputs

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            LogisticRegression().outputs

    def test_targets(self, simple_step, shared_step, dataplaceholders, teardown):
        y_t = dataplaceholders[2]
        assert simple_step.targets == [y_t]

        with pytest.raises(AttributeError):
            shared_step.targets

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            LogisticRegression().targets

    def test_compute_func(self, simple_step, shared_step, dataplaceholders, teardown):
        assert simple_step.compute_func == simple_step.predict
        simple_step.compute_func = simple_step.predict_proba
        assert simple_step.compute_func == simple_step.predict_proba

        with pytest.raises(AttributeError):
            shared_step.compute_func

        with pytest.raises(AttributeError):
            shared_step.compute_func = shared_step.predict_proba

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            LogisticRegression().compute_func

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            LogisticRegression().compute_func = lambda x: x

    def test_fit_compute_func(
        self, simple_step, shared_step, dataplaceholders, teardown
    ):
        assert simple_step.fit_compute_func == simple_step.fit_predict
        simple_step.fit_compute_func = simple_step.fit_predict_proba
        assert simple_step.fit_compute_func == simple_step.fit_predict_proba

        with pytest.raises(AttributeError):
            shared_step.fit_compute_func

        with pytest.raises(AttributeError):
            shared_step.fit_compute_func = shared_step.fit_predict_proba

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            DummyEstimator().fit_compute_func

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            DummyEstimator().fit_compute_func = lambda x: x

    def test_trainable(self, simple_step, shared_step, dataplaceholders, teardown):
        assert simple_step.trainable
        simple_step.trainable = False
        assert not simple_step.trainable

        with pytest.raises(AttributeError):
            shared_step.trainable

        with pytest.raises(AttributeError):
            shared_step.trainable = True

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            LogisticRegression().trainable

        with pytest.raises(AttributeError):
            # because the step hasn't been called
            LogisticRegression().trainable = False

    def test_get_inputs_at(self, simple_step, shared_step, dataplaceholders, teardown):
        x1, x2, *_ = dataplaceholders
        assert simple_step.get_inputs_at(0) == [x1]
        assert shared_step.get_inputs_at(0) == [x1]
        assert shared_step.get_inputs_at(1) == [x2]

    def test_get_outputs_at(self, simple_step, shared_step, dataplaceholders, teardown):
        *_, y_simple, y_shared_1, y_shared_2 = dataplaceholders
        assert simple_step.get_outputs_at(0) == [y_simple]
        assert shared_step.get_outputs_at(0) == [y_shared_1]
        assert shared_step.get_outputs_at(1) == [y_shared_2]

    def test_get_targets_at(self, simple_step, shared_step, dataplaceholders, teardown):
        y_t = dataplaceholders[2]
        assert simple_step.get_targets_at(0) == [y_t]
        assert shared_step.get_targets_at(0) == [y_t]
        assert shared_step.get_targets_at(1) == []

    def test_get_compute_func_at(
        self, simple_step, shared_step, dataplaceholders, teardown
    ):
        assert simple_step.get_compute_func_at(0) == simple_step.predict
        assert shared_step.get_compute_func_at(0) == shared_step.predict
        assert shared_step.get_compute_func_at(1) == shared_step.predict_proba

    def test_set_compute_func_at(self, shared_step, dataplaceholders, teardown):
        shared_step.set_compute_func_at(1, shared_step.predict)
        assert shared_step.get_compute_func_at(1) == shared_step.predict

    def test_get_fit_compute_func_at(
        self, simple_step, shared_step, dataplaceholders, teardown
    ):
        assert simple_step.get_fit_compute_func_at(0) == simple_step.fit_predict
        assert shared_step.get_fit_compute_func_at(0) == shared_step.fit_predict
        assert shared_step.get_fit_compute_func_at(1) == shared_step.fit_predict_proba

    def test_set_fit_compute_func_at(self, shared_step, dataplaceholders, teardown):
        shared_step.set_fit_compute_func_at(1, shared_step.fit_predict)
        assert shared_step.get_fit_compute_func_at(1) == shared_step.fit_predict

    def test_get_trainable_at(
        self, simple_step, shared_step, dataplaceholders, teardown
    ):
        assert simple_step.get_trainable_at(0)
        assert shared_step.get_trainable_at(0)
        assert not shared_step.get_trainable_at(1)

    def test_set_trainable_at(self, shared_step, dataplaceholders, teardown):
        shared_step.set_trainable_at(1, True)
        assert shared_step.get_trainable_at(1)
