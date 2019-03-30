from typing import List, Tuple

from baikal.core.data_placeholder import DataPlaceholder, is_data_placeholder_list
from baikal.core.digraph import Node
from baikal.core.utils import listify, make_name


class Step(Node):
    def __init__(self, *args, name=None, trainable=True, **kwargs):
        super(Step, self).__init__(*args, name=name, **kwargs)
        self.inputs = None
        self.outputs = None
        self.n_outputs = None  # Client code must override this value when subclassing from Step.
        self.trainable = trainable

    def __call__(self, inputs):
        # TODO: Add a target keyword argument to specify inputs that are only required at fit time
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


class InputStep(Node):
    def __init__(self, name=None):
        super(InputStep, self).__init__(name=name)
        self.inputs = []
        self.outputs = [DataPlaceholder(self, self.name)]


def Input(name=None):
    # Maybe this can be implemented in InputStep.__new__
    input = InputStep(name)
    return input.outputs[0]  # Input produces exactly one DataPlaceholder output
