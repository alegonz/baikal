import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

from baikal._core.data_placeholder import DataPlaceholder, is_data_placeholder_list
from baikal._core.utils import listify, make_name, make_repr, make_args_from_attrs


class _StepBase:
    # used to keep track of number of instances and make unique names
    # a dict-of-dicts with graph and name as keys.
    _names = dict()  # type: Dict[str, int]

    def __init__(self, *args, name: str = None, n_outputs: int = 1, **kwargs):
        # Necessary to use this class as a mixin
        super().__init__(*args, **kwargs)  # type: ignore

        # Use name as is if it was specified by the user, to avoid the user a surprise
        self._name = name if name is not None else self._generate_unique_name()
        # TODO: Add self.n_inputs? Could be used to check inputs in __call__
        self._n_outputs = n_outputs

    def _generate_unique_name(self):
        name = self.__class__.__name__

        n_instances = self._names.get(name, 0)
        unique_name = make_name(name, n_instances, sep="_")

        n_instances += 1
        self._names[name] = n_instances

        return unique_name

    @classmethod
    def _clear_names(cls):
        # For testing purposes only.
        cls._names.clear()

    def _get_param_names(self):
        """This is a workaround to override @classmethod binding of the sklearn
        parent class method so we can feed it the sklearn parent class instead
        of the children class. We assume client code subclassed from this mixin
        and a sklearn class, with the sklearn class being the next base class in
        the mro.
        """
        return super()._get_param_names.__func__(super())

    @property
    def name(self):
        return self._name

    @property
    def n_outputs(self):
        return self._n_outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def targets(self):
        return self._targets


# TODO: Update docstrings
class Step(_StepBase):
    """Mixin class to endow scikit-learn classes with Step capabilities.

    Steps are defined by combining any class we would like to make a step from
    with this mixin class. This mixin, among other things, endows the class of
    interest with a `__call__` method, making the class callable on the outputs
    (`DataPlaceholder` objects) of previous steps and optional targets (also
    `DataPlaceholder` objects). You can make a step from any class you like,
    so long that class implements the scikit-learn API.

    Instructions:
        1. Define a class that inherits from both this mixin and the class you
           wish to make a step of (in that order!).
        2. In the class `__init__`, call `super().__init__(...)` and pass the
           appropriate step parameters.

    Parameters
    ----------
    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.

    function
        Specifies which function must be used when computing the step during
        the model graph execution. If None (default), it will use the predict
        or the transform method (in that order). If a name string is passed,
        it will use the method that matches the given name. If a callable is
        passed, it will use that callable when computing the step.

        The number of inputs and outputs of the function must match those of the
        step (this is not checked, but will raise an error during graph
        execution if there is a mismatch).

        scikit-learn classes typically implement a predict method (Estimators)
        or a transform method (Transformers), but with this argument you can,
        for example, specify `predict_proba` as the compute function.

    n_outputs
        The number of outputs of the step's function (predict, transform, or
        any other callable passed in the `function` argument).

    trainable
        Whether the step is trainable (True) or not (False). This flag is only
        meaningful only for steps with a fit method. Setting `trainable=False`
        allows to skip the step when fitting a Model. This is useful if you
        want to freeze some pre-trained steps.

    Attributes
    ----------
    inputs
        Inputs of the step.

    outputs
        Outputs of the step.

    targets
        Targets of the step.

    n_outputs
        Number of outputs the step must be produce.

    Examples
    --------
    >>> import sklearn.linear_model
    >>> # The order of inheritance is important!
    >>> class LogisticRegression(Step, sklearn.linear_model.LogisticRegression):
    >>>     def __init__(self, name=None, **kwargs):
    >>>         super().__init__(name=name, **kwargs)
    >>>
    >>> logreg = LogisticRegression(C=2.0, function='predict_proba')
    """

    def __init__(
        self,
        *args,
        name: str = None,
        function: Optional[Union[str, Callable[..., Any]]] = None,
        n_outputs: int = 1,
        trainable: bool = True,
        **kwargs
    ):
        # Necessary to use this class as a mixin
        super().__init__(*args, name=name, n_outputs=n_outputs, **kwargs)  # type: ignore

        self.trainable = trainable
        self.function = self._check_function(function)
        self._inputs = []  # type: List[DataPlaceholder]
        self._outputs = []  # type: List[DataPlaceholder]
        self._targets = []  # type: List[DataPlaceholder]

    def _check_function(self, function):
        if function is None:
            if hasattr(self, "predict"):
                function = self.predict
            elif hasattr(self, "transform"):
                function = self.transform
            else:
                raise ValueError(
                    "If `function` is not specified, the class "
                    "must implement a predict or transform method."
                )
        else:
            if isinstance(function, str):
                function = getattr(self, function)
            elif callable(function):
                pass
            else:
                raise ValueError(
                    "If specified, `function` must be either a " "string or a callable."
                )
        return function

    def compute(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __call__(
        self,
        inputs: Union[DataPlaceholder, List[DataPlaceholder]],
        targets: Optional[Union[DataPlaceholder, List[DataPlaceholder]]] = None,
    ) -> Union[DataPlaceholder, List[DataPlaceholder]]:
        """Call the step on input(s) (from previous steps) and generates the
        output(s) to be used in further steps.

        Parameters
        ----------
        inputs
            Input(s) to the step.

        targets
            Target(s) to the step.

        Returns
        -------
        DataPlaceholder
            Output(s) of the step.

        Notes
        -----
        Currently, calling the same step on different inputs and targets to
        reuse the step (similar to the concept of shared layers and nodes in
        Keras) is not supported. Calling a step twice on different inputs will
        override the connectivity from the first call. Support for shareable
        steps might be added in future releases.
        """
        inputs = listify(inputs)
        if not is_data_placeholder_list(inputs):
            raise ValueError("inputs must be of type DataPlaceholder.")

        if targets is not None:
            if not hasattr(self, "fit"):
                raise RuntimeError(
                    "Cannot pass targets to steps that do not have a fit method."
                )

            # TODO: Consider inspecting the fit signature to determine whether the step
            # needs a target (i.e. fit(self, X, y)) or not (i.e. fit(self, X, y=None)).
            # The presence of a default of None for the target might not be reliable
            # though, as there could be estimators (perhaps semi-supervised) that can take
            # both target data and None. Also, sklearn has meta-estimators (e.g. Pipeline)
            # and meta-transformers (e.g. SelectFromModel) that accept both target data
            # and None.
            #
            # Adding this inspection, however, could simplify the API by rejecting early
            # unnecessary targets (e.g. passing targets to PCA) or warning missing targets
            # (e.g. not passing targets to LogisticRegression with trainable=True). This
            # also avoids unintuitive logic to allow superfluous targets during step call,
            # model instantiation and model fit.
            #
            # | requires target |   trainable   | passed target |   result   |
            # ----------------------------------------------------------------
            # |       yes       |      True     |      yes      |     ok     |
            # |       yes       |      True     |      no       |    warn    |
            # |       yes       |      False    |      yes      |    warn    |
            # |       yes       |      False    |      no       |     ok     |
            # |       no        |        -      |      yes      |    error   |
            # |       no        |        -      |      no       |     ok     |

            if not self.trainable:
                warnings.warn(
                    UserWarning("You are passing targets to a non-trainable step.")
                )

            targets = listify(targets)
            if not is_data_placeholder_list(targets):
                raise ValueError(
                    "If specified, targets must be of type DataPlaceholder."
                )

        else:
            targets = []

        self._inputs = inputs
        self._targets = targets
        self._outputs = self._build_outputs()

        if self._n_outputs == 1:
            return self._outputs[0]
        else:
            # Return a shallow copy to avoid modifying self._outputs when
            # using the idiom of passing a variable holding an output to
            # another step and re-writing the variable with the new output:
            #     zs = SomeMultiOutputStep()(...)
            #     zs[i] = SomeStep()(zs[i])
            return list(self.outputs)

    def _build_outputs(self) -> List[DataPlaceholder]:
        outputs = []
        for i in range(self._n_outputs):
            name = make_name(self._name, i)
            outputs.append(DataPlaceholder(self, name))
        return outputs

    def __repr__(self):
        cls_name = self.__class__.__name__
        parent_repr = super().__repr__()
        step_attrs = ["name", "function", "n_outputs", "trainable"]

        # Insert Step attributes into the parent repr
        # if the repr has the sklearn pattern
        sklearn_pattern = r"^" + cls_name + r"\((.*)\)$"
        match = re.search(sklearn_pattern, parent_repr, re.DOTALL)
        if match:
            parent_args = match.group(1)
            indentations = re.findall("[ \t]{2,}", parent_args)
            indent = min(indentations, key=len) if indentations else ""
            step_args = make_args_from_attrs(self, step_attrs)
            repr = "{}({},\n{}{})".format(cls_name, step_args, indent, parent_args)
            return repr

        else:
            return make_repr(self, step_attrs)


class InputStep(_StepBase):
    """Special Step subclass for Model inputs.

    It is characterized by having no inputs (in_degree == 0)
    and exactly one output (out_degree == 1).
    """

    def __init__(self, name=None):
        super().__init__(name=name, n_outputs=1)
        self._inputs = []
        self._outputs = [DataPlaceholder(self, self._name)]
        self._targets = []
        self.trainable = False

    def __repr__(self):
        step_attrs = ["name"]
        return make_repr(self, step_attrs)


def Input(name: Optional[str] = None) -> DataPlaceholder:
    """Helper function that instantiates a data placeholder representing an
    input of the model.

    This function is the starting point when building models.

    Parameters
    ----------
    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.

    Returns
    -------
    DataPlaceholder
        A data placeholder.
    """
    # Maybe this can be implemented in InputStep.__new__
    input = InputStep(name)
    return input._outputs[0]  # Input produces exactly one DataPlaceholder output


# Notes on typing:
# mypy produces false positives with mixins, so we use type: ignore
# See:
# https://github.com/python/mypy/issues/1996
# https://github.com/python/mypy/issues/5887
