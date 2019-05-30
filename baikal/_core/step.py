import re
from typing import List

from baikal._core.data_placeholder import DataPlaceholder, is_data_placeholder_list
from baikal._core.utils import listify, make_name, make_repr, make_args_from_attrs


class Step:
    # used to keep track of number of instances and make unique names
    # a dict-of-dicts with graph and name as keys.
    _names = dict()

    def __init__(self, *args, name=None, trainable=True, function=None, **kwargs):
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

    def __call__(self, inputs):
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
    def clear_names(cls):
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
        # Workaround to override @classmethod binding of the sklearn parent class method
        # so we can feed it the sklearn parent class instead of the children class.
        # We assume client code subclassed from this mixin and a sklearn class, with
        # the sklearn class being the next base class in the mro.
        return super(Step, self)._get_param_names.__func__(super(Step, self))


class InputStep(Step):
    def __init__(self, name=None):
        super(InputStep, self).__init__(name=name, trainable=False, function=None)
        self.inputs = []
        self.outputs = [DataPlaceholder(self, self.name)]

    def _check_function(self, function):
        pass


def Input(name=None):
    # Maybe this can be implemented in InputStep.__new__
    input = InputStep(name)
    return input.outputs[0]  # Input produces exactly one DataPlaceholder output
