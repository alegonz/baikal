import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from baikal._core.data_placeholder import DataPlaceholder, is_data_placeholder_list
from baikal._core.utils import listify, make_name, make_repr


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
        self._nodes = []  # type: List[Node]

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

    @classmethod
    def _get_super_class_with_init(cls):
        """Get super class that defines __init__"""
        mro = cls.mro()
        for super_class in mro[mro.index(_StepBase) + 1 :]:
            if hasattr(super_class, "__init__"):
                # object is the last class in the mro and it defines an __init__ method
                # so this is guaranteed to return before the for loop finishes
                return super_class

    @classmethod
    def _get_param_names(cls):
        """This is a workaround to override @classmethod binding of the sklearn
        parent class method so we can feed it the sklearn parent class instead
        of the children class. We assume client code subclassed from this mixin
        and a sklearn class, with the sklearn class being the next base class in
        the mro.
        """
        super_class = cls._get_super_class_with_init()
        return super()._get_param_names.__func__(super_class)

    def _get_step_attr(self, attr):
        n_nodes = len(self._nodes)
        if n_nodes == 0:
            raise AttributeError("{} has not been connected yet.".format(self.name))
        elif n_nodes == 1:
            return getattr(self._nodes[0], attr)
        else:
            raise AttributeError(
                "{} has been connected {} times (it is a shared step). "
                "Use `get_{}_at(port)` instead.".format(self.name, n_nodes, attr)
            )

    def _set_step_attr(self, attr, value):
        n_nodes = len(self._nodes)
        if n_nodes == 0:
            raise AttributeError("{} has not been connected yet.".format(self.name))
        elif n_nodes == 1:
            setattr(self._nodes[0], attr, value)
        else:
            raise AttributeError(
                "{} has been connected {} times (it is a shared step). "
                "Use `set_{}_at(port)` instead.".format(self.name, n_nodes, attr)
            )

    @property
    def name(self):
        """Get the name of the step."""
        return self._name

    @property
    def n_outputs(self):
        """Get the number of outputs the step produces."""
        return self._n_outputs

    @property
    def inputs(self) -> List[DataPlaceholder]:
        """Get the inputs of the step.

        You can use this only when the step has been called exactly once.

        Returns
        -------
        List of inputs.

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        return self._get_step_attr("inputs")

    @property
    def outputs(self) -> List[DataPlaceholder]:
        """Get the outputs of the step.

        You can use this only when the step has been called exactly once.

        Returns
        -------
        List of outputs.

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        return self._get_step_attr("outputs")

    @property
    def targets(self) -> List[DataPlaceholder]:
        """Get the targets of the step.

        You can use this only when the step has been called exactly once.

        Returns
        -------
        List of targets.

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        return self._get_step_attr("targets")

    @property
    def compute_func(self) -> Callable:
        """Get the compute function of the step.

        You can use this only when the step has been called exactly once.

        Returns
        -------
        Callable

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        return self._get_step_attr("compute_func")

    @compute_func.setter
    def compute_func(self, value: Callable):
        """Set the compute function of the step.

        You can use this only when the step has been called exactly once.

        Parameters
        ----------
        value
            Compute function of the step.

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        self._set_step_attr("compute_func", value)

    @property
    def fit_compute_func(self) -> Optional[Callable]:
        """Get the fit-compute function of the step.

        You can use this only when the step has been called exactly once.

        Returns
        -------
        Callable

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        return self._get_step_attr("fit_compute_func")

    @fit_compute_func.setter
    def fit_compute_func(self, value: Optional[Callable]):
        """Set the fit-compute function of the step.

        You can use this only when the step has been called exactly once.

        Parameters
        ----------
        value
            fit-compute function of the step. Pass ``None`` to disable it.

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        self._set_step_attr("fit_compute_func", value)

    @property
    def trainable(self) -> bool:
        """Get trainable flag of the step.

        You can use this only when the step has been called exactly once.

        Returns
        -------
        bool

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        return self._get_step_attr("trainable")

    @trainable.setter
    def trainable(self, value: bool):
        """Set trainable flag of the step.

        You can use this only when the step has been called exactly once.

        Parameters
        ----------
        value
            Trainable flag.

        Raises
        ------
        AttributeError
            If the step has not been called yet or it is a shared step
            (called several times).
        """
        self._set_step_attr("trainable", value)

    def get_inputs_at(self, port: int) -> List[DataPlaceholder]:
        """Get inputs at the specified port.

        Parameters
        ----------
        port
            Port from which to get the inputs.

        Returns
        -------
        List of inputs.
        """
        return self._nodes[port].inputs

    def get_outputs_at(self, port: int) -> List[DataPlaceholder]:
        """Get outputs at the specified port.

        Parameters
        ----------
        port
            Port from which to get the outputs.

        Returns
        -------
        List of outputs.
        """
        return self._nodes[port].outputs

    def get_targets_at(self, port: int) -> List[DataPlaceholder]:
        """Get targets at the specified port.

        Parameters
        ----------
        port
            Port from which to get the targets.

        Returns
        -------
        List of targets.
        """
        return self._nodes[port].targets

    def get_compute_func_at(self, port: int) -> Callable:
        """Get compute function at the specified port.

        Parameters
        ----------
        port
            Port from which to get the compute function.

        Returns
        -------
        Callable
        """
        return self._nodes[port].compute_func

    def set_compute_func_at(self, port: int, value: Callable):
        """Set compute function at the specified port.

        Parameters
        ----------
        port
            Port on which to set the compute function.

        value
            Compute function of the step.
        """
        self._nodes[port].compute_func = value

    def get_fit_compute_func_at(self, port: int) -> Optional[Callable]:
        """Get fit-compute function at the specified port.

        Parameters
        ----------
        port
            Port from which to get the fit-compute function.

        Returns
        -------
        Callable or None
        """
        return self._nodes[port].fit_compute_func

    def set_fit_compute_func_at(self, port: int, value: Optional[Callable]):
        """Set fit-compute function at the specified port.

        Parameters
        ----------
        port
            Port on which to set the fit-compute function.

        value
            fit-compute function of the step. Pass ``None`` to disable it.
        """
        self._nodes[port].fit_compute_func = value

    def get_trainable_at(self, port: int) -> bool:
        """Get trainable flag at the specified port.

        Parameters
        ----------
        port
            Port from which to get the trainable flag.

        Returns
        -------
        bool
        """
        return self._nodes[port].trainable

    def set_trainable_at(self, port: int, value: bool):
        """Set trainable flag at the specified port.

        Parameters
        ----------
        port
            Port on which to set the trainable flag.

        value
            Trainable flag.
        """
        self._nodes[port].trainable = value


class Step(_StepBase):
    """Mixin class to endow scikit-learn classes with Step capabilities.

    Steps are defined by combining any class we would like to make a step from
    with this mixin class. This mixin, among other things, endows the class of
    interest with a ``__call__`` method, making the class callable on the outputs
    (``DataPlaceholder`` objects) of previous steps and optional targets (also
    ``DataPlaceholder`` objects). You can make a step from any class you like,
    so long that class implements the scikit-learn API.

    Instructions:
        1. Define a class that inherits from both this mixin and the class you
           wish to make a step of (in that order!).
        2. In the class ``__init__``, call ``super().__init__(...)`` and pass the
           appropriate step parameters.

    The base class may implement a predict/transform method (the compute function)
    that take multiple inputs and returns multiple outputs, and a fit method that
    takes multiple inputs and targets. In this case, the input/target arguments are
    expected to be a list of (typically) array-like objects, and the compute function
    is expected to return a list of array-like objects.

    Parameters
    ----------
    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.

    n_outputs
        The number of outputs of the step's function (predict, transform, or
        any other callable passed in the ``compute_func`` argument).

    Examples
    --------
    ::

        import sklearn.linear_model
        # The order of inheritance is important!
        class LogisticRegression(Step, sklearn.linear_model.LogisticRegression):
            def __init__(self, *args, name=None, **kwargs):
                super().__init__(*args, name=name, **kwargs)

        logreg = LogisticRegression(C=2.0)

    """

    if TYPE_CHECKING:  # pragma: no cover

        def fit(self, X, y, **fit_params):
            return self

    def __init__(self, *args, name: str = None, n_outputs: int = 1, **kwargs):
        # Necessary to use this class as a mixin
        super().__init__(*args, name=name, n_outputs=n_outputs, **kwargs)  # type: ignore

        self._nodes = []  # type: List[Node]

    def _check_compute_func(self, compute_func):
        if compute_func == "auto":
            if hasattr(self, "predict"):
                compute_func = self.predict
            elif hasattr(self, "transform"):
                compute_func = self.transform
            else:
                raise ValueError(
                    "If `compute_func` is not specified, the class "
                    "must implement a predict or transform method."
                )
        else:
            if isinstance(compute_func, str):
                compute_func = getattr(self, compute_func)
            elif callable(compute_func):
                pass
            else:
                raise ValueError(
                    "If specified, `compute_func` must be either a "
                    "string or a callable."
                )
        return compute_func

    def _check_fit_compute_func(self, fit_compute_func):
        if fit_compute_func == "auto":
            if hasattr(self, "fit_predict"):
                fit_compute_func = self.fit_predict
            elif hasattr(self, "fit_transform"):
                fit_compute_func = self.fit_transform
            else:
                fit_compute_func = None
        else:
            if isinstance(fit_compute_func, str):
                fit_compute_func = getattr(self, fit_compute_func)
            elif callable(fit_compute_func):
                pass
            elif fit_compute_func is None:
                pass
            else:
                raise ValueError(
                    "If specified, `fit_compute_func` must be either None, a string or"
                    " a callable."
                )
        return fit_compute_func

    def __call__(
        self,
        inputs: Union[DataPlaceholder, List[DataPlaceholder]],
        targets: Optional[Union[DataPlaceholder, List[DataPlaceholder]]] = None,
        *,
        compute_func: Union[str, Callable[..., Any]] = "auto",
        fit_compute_func: Optional[Union[str, Callable[..., Any]]] = "auto",
        trainable: bool = True
    ) -> Union[DataPlaceholder, List[DataPlaceholder]]:
        """Call the step on input(s) (from previous steps) and generates the
        output(s) to be used in further steps.

        You can call the same step on different inputs and targets to reuse the step
        (similar to the concept of shared layers and nodes in Keras), and specify a
        different ``compute_func``/``trainable`` configuration on each call. This is
        achieved via "ports": each call creates a new port and associates the given
        configuration to it. You may access the configuration at each port using the
        ``get_*_at(port)`` methods.

        Parameters
        ----------
        inputs
            Input(s) to the step.

        targets
            Target(s) to the step.

        compute_func
            Specifies which function must be used when computing the step during
            the model graph execution. If ``"auto"`` (default), it will use the ``predict``
            or the ``transform`` method (in that order). If a name string is passed,
            it will use the method that matches the given name. If a callable is
            passed, it will use that callable when computing the step.

            The number of inputs and outputs of the function must match those of the
            step (this is not checked, but will raise an error during graph
            execution if there is a mismatch).

            scikit-learn classes typically implement a ``predict`` method (Estimators)
            or a ``transform`` method (Transformers), but with this argument you can,
            for example, specify ``predict_proba`` as the compute function.

        fit_compute_func
            Specifies which function must be used when fitting AND computing the step
            during the model graph execution.

            If ``"auto"`` (default), it will use the ``fit_predict`` or the ``fit_transform``
            method (in that order) if they are implemented, otherwise it will be
            disabled. If a name string is passed, it will use the method that matches
            the given name. If a callable is passed, it will use that callable when
            fitting the step. If ``None`` is passed it will be ignored during graph
            execution.

            The number of inputs, outputs and targets, of the function must match those
            of the step (this is not checked, but will raise an error during graph
            execution if there is a mismatch).

            By default, when a model is fit, the graph engine will for each step
            1) execute ``fit`` to fit the step, and then 2) execute ``compute_func`` to
            compute the outputs required by successor steps. If a step specifies a
            ``fit_compute_func``, the graph execution will use that instead to fit and
            compute the outputs in a single call. This can be useful for

            1. leveraging implementations of ``fit_transform`` that are more efficient
               than calling ``fit`` and ``transform`` separately,
            2. using transductive estimators,
            3. implementing training protocols such as that of stacked classifiers,
               where the classifier in the first stage might compute out-of-fold
               predictions.

        trainable
            Whether the step is trainable (True) or not (False). This flag is only
            meaningful only for steps with a fit method. Setting ``trainable=False``
            allows to skip the step when fitting a Model. This is useful if you
            want to freeze some pre-trained steps.

        Returns
        -------
        DataPlaceholder
            Output(s) of the step.
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

            if not trainable:
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

        outputs = self._build_outputs()

        self._nodes.append(
            Node(
                self,
                inputs,
                outputs,
                targets,
                getattr(self, "fit", None),
                self._check_compute_func(compute_func),
                self._check_fit_compute_func(fit_compute_func),
                trainable,
            )
        )

        if self._n_outputs == 1:
            return outputs[0]
        else:
            # Return a shallow copy to avoid modifying self._outputs when
            # using the idiom of passing a variable holding an output to
            # another step and re-writing the variable with the new output:
            #     zs = SomeMultiOutputStep()(...)
            #     zs[i] = SomeStep()(zs[i])
            return list(outputs)

    def _build_outputs(self) -> List[DataPlaceholder]:
        port = len(self._nodes)
        outputs = []
        for i in range(self._n_outputs):
            name = make_name(make_name(self._name, port, sep=":"), i)
            outputs.append(DataPlaceholder(self, port, name))
        return outputs

    def __repr__(self):
        return self._repr()

    def _repr(self, n_char_max=700, n_max_elements_to_show=30, depth=None):
        """Adapted from the original in scikit-learn BaseEstimator.__repr__

        Parameters
        ----------
        n_char_max
            (Approximate) maximum number of non-blank characters to render.
        n_max_elements_to_show
            Number of elements to show in sequences.
        depth
            The maximum depth to print out nested structures.

        Returns
        -------
        repr string of the object.
        """
        from baikal._core.pprint import _StepPrettyPrinter, post_process_repr

        pp = _StepPrettyPrinter(
            compact=True,
            indent=1,
            indent_at_name=True,
            n_max_elements_to_show=n_max_elements_to_show,
            depth=depth,
        )
        repr_ = pp.pformat(self)
        return post_process_repr(repr_, n_char_max)


class InputStep(_StepBase):
    """Special Step subclass for Model inputs.

    It is characterized by having no inputs (in_degree == 0)
    and exactly one output (out_degree == 1).
    """

    def __init__(self, name=None):
        super().__init__(name=name, n_outputs=1)
        self._nodes = [
            Node(
                self,
                [],
                [DataPlaceholder(self, 0, self._name)],
                [],
                None,
                None,
                None,
                False,
            )
        ]

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
    return input.outputs[0]  # Input produces exactly one DataPlaceholder output


class Node:
    def __init__(
        self,
        step: Step,
        inputs: List[DataPlaceholder],
        outputs: List[DataPlaceholder],
        targets: List[DataPlaceholder],
        fit_func: Optional[Callable],
        compute_func: Callable,
        fit_compute_func: Optional[Callable],
        trainable: bool,
    ):
        self._step = step
        self._inputs = inputs
        self._outputs = outputs
        self._targets = targets
        self.fit_func = fit_func  # at present, fit is the same for all step calls
        self.compute_func = compute_func
        self.fit_compute_func = fit_compute_func
        self.trainable = trainable

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, new_step):
        old_step = self._step
        self._step = new_step

        # Update outputs of old step to point to the new step
        # Note that the dataplaceholders keep the name from the old step
        # TODO: Maybe the output dataplaceholders should be replaced too
        for output in self._outputs:
            output._step = new_step

        # Special process to transfer functions:
        # if it is a bound method get the corresponding method bound to the
        # new step otherwise leave it as is.
        # Note that Step._check_[fit_]compute_func guarantees step.[fit_]compute_func
        # is a callable (i.e: assert callable(old_step.[fit_]compute_func) passes)
        for attr_name in ("fit_func", "compute_func", "fit_compute_func"):
            old_attr = getattr(self, attr_name)
            if inspect.ismethod(old_attr):
                assert old_attr.__self__ is old_step
                setattr(self, attr_name, getattr(new_step, old_attr.__name__))

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def targets(self):
        return self._targets

    @property
    def port(self):
        return self.step._nodes.index(self)

    @property
    def name(self):
        return make_name(self.step.name, self.port, sep=":")


# Notes on typing:
# mypy produces false positives with mixins, so we use type: ignore
# See:
# https://github.com/python/mypy/issues/5887
