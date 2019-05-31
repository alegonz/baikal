import re
from typing import List, Optional, Union, Callable, Any

from baikal._core.data_placeholder import DataPlaceholder, is_data_placeholder_list
from baikal._core.utils import listify, make_name, make_repr, make_args_from_attrs


class Step:
    """Mixin class to endow scikit-learn classes with Step capabilities.

    Steps are defined by combining any class we would like to make a step from
    with this mixin class. This mixin, among other things, endows the class of
    interest with a `__call__` method, making the class callable on the outputs
    (`DataPlaceholder` objects) of previous steps. You can make a step from any
    class you like, so long that class implements the scikit-learn API.

    Instructions:
        1. Define a class that inherits from both this mixin and the class you
           wish to make a step of (in that order!).
        2. Set the `self.n_outputs` variable to the number of outputs the step
           should output at predict/transform time.

    Parameters
    ----------
    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.

    trainable
        Whether the step is trainable (True) or not (False). This flag is only
        meaningful only for steps with a fit method. Setting `trainable=False`
        allows to skip the step when fitting a Model. This is useful if you
        want to freeze some pre-trained steps.

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

    Attributes
    ----------
    inputs
        Inputs of the step.

    outputs
        Outputs of the step.

    n_outputs
        Number of outputs the step must be produce.

    Examples
    --------
    >>> import sklearn.linear_model
    >>> # The order of inheritance is important!
    >>> class LogisticRegression(Step, sklearn.linear_model.LogisticRegression):
    >>>     def __init__(self, name=None, **kwargs):
    >>>         super(LogisticRegression, self).__init__(name=name, **kwargs)
    >>>         self.n_outputs = 1  # Make sure to set this value!
    >>>
    >>> logreg = LogisticRegression(C=2.0, function='predict_proba')
    """

    # used to keep track of number of instances and make unique names
    # a dict-of-dicts with graph and name as keys.
    _names = dict()

    def __init__(self,
                 *args,
                 name:str = None,
                 trainable: bool = True,
                 function: Optional[Union[str, Callable[..., Any]]] = None,
                 **kwargs):
        super(Step, self).__init__(*args, **kwargs)  # Necessary to use this class as a mixin

        # Use name as is if it was specified by the user, to avoid the user a surprise
        self.name = name if name is not None else self._generate_unique_name()

        self.inputs = None
        self.outputs = None
        # TODO: Add self.n_inputs? Could be used to check inputs in __call__
        self.n_outputs = None  # Client code must override this value when subclassing from Step.
        self.trainable = trainable
        self.function = self._check_function(function)

    def _check_function(self, function):
        if function is None:
            if hasattr(self, 'predict'):
                function = self.predict
            elif hasattr(self, 'transform'):
                function = self.transform
            else:
                raise ValueError('If `function` is not specified, the class must '
                                 'implement a predict or transform method.')
        else:
            if isinstance(function, str):
                function = getattr(self, function)
            elif callable(function):
                pass
            else:
                raise ValueError('`function` must be either None, a string or a callable.')
        return function

    def compute(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __call__(self, inputs: Union[DataPlaceholder, List[DataPlaceholder]]) \
            -> Union[DataPlaceholder, List[DataPlaceholder]]:
        """Call the step on input(s) (from previous steps) and generates the
        output(s) to be used in further steps.

        Parameters
        ----------
        inputs
            Input(s) to the step.

        Returns
        -------
        DataPlaceholder
            Output(s) of the step.

        Notes
        -----
        Currently, calling the same step on different inputs to reuse the step
        (similar to the concept of shared layers and nodes in Keras) is not
        supported. Calling a step twice on different inputs will override the
        connectivity from the first call. Support for shareable steps might be
        added in future releases.
        """
        inputs = listify(inputs)

        if not is_data_placeholder_list(inputs):
            raise ValueError('Steps must be called with DataPlaceholder inputs.')

        self.inputs = inputs
        self.outputs = self._build_outputs()

        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def _build_outputs(self) -> List[DataPlaceholder]:
        outputs = []
        for i in range(self.n_outputs):
            name = make_name(self.name, i)
            outputs.append(DataPlaceholder(self, name))
        return outputs

    def _generate_unique_name(self):
        name = self.__class__.__name__

        n_instances = self._names.get(name, 0)
        unique_name = make_name(name, n_instances, sep='_')

        n_instances += 1
        self._names[name] = n_instances

        return unique_name

    @classmethod
    def _clear_names(cls):
        # For testing purposes only.
        cls._names.clear()

    def __repr__(self):
        cls_name = self.__class__.__name__
        parent_repr = super(Step, self).__repr__()
        step_attrs = ['name', 'trainable', 'function']

        # Insert Step attributes into the parent repr
        # if the repr has the sklearn pattern
        sklearn_pattern = '^' + cls_name + '\((.*)\)$'
        match = re.search(sklearn_pattern, parent_repr, re.DOTALL)
        if match:
            parent_args = match.group(1)
            indentations = re.findall('[ \t]{2,}', parent_args)
            indent = min(indentations, key=len) if indentations else ''
            step_args = make_args_from_attrs(self, step_attrs)
            repr = '{}({},\n{}{})'.format(cls_name, step_args, indent, parent_args)
            return repr

        else:
            return make_repr(self, step_attrs)

    def _get_param_names(self):
        """This is a workaround to override @classmethod binding of the sklearn
        parent class method so we can feed it the sklearn parent class instead
        of the children class. We assume client code subclassed from this mixin
        and a sklearn class, with the sklearn class being the next base class in
        the mro.
        """
        return super(Step, self)._get_param_names.__func__(super(Step, self))


class InputStep(Step):
    """Special Step subclass for Model inputs.

    It is characterized by having no inputs and exactly one output.
    """
    def __init__(self, name=None):
        super(InputStep, self).__init__(name=name, trainable=False, function=None)
        self.inputs = []
        self.outputs = [DataPlaceholder(self, self.name)]

    def _check_function(self, function):
        pass


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
