from typing import List, Union, Tuple

from baikal.core.data import Data, is_data_list
from baikal.core.digraph import Node
from baikal.core.utils import listify


class Step(Node):
    def __init__(self, *args, name=None, **kwargs):
        super(Step, self).__init__(*args, name=name, **kwargs)
        self.inputs = None
        self.outputs = None

    def __call__(self, inputs):
        # TODO: Add a target keyword argument to specify inputs that are only required at fit time
        inputs = listify(inputs)

        if not is_data_list(inputs):
            raise ValueError('Steps must be called with Data inputs.')

        self.inputs = inputs
        self.outputs = self._build_outputs(inputs)

        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    # TODO: We might need a check_inputs method as well (Concatenate, Split, Merge, etc will need it).
    # Also, sklearn-based Steps can accept only shapes of length 1
    # (ignoring the samples, the dimensionality of the feature vector)

    def _build_outputs(self, inputs: List[Data]) -> List[Data]:
        input_shapes = [input.shape for input in inputs]
        output_shapes = self.build_output_shapes(input_shapes)

        outputs = [Data(shape, self, i) for i, shape in enumerate(output_shapes)]
        return outputs

    def build_output_shapes(self, input_shapes: List[Tuple]) -> List[Tuple]:
        raise NotImplementedError


class InputStep(Node):
    def __init__(self, shape, name=None):
        super(InputStep, self).__init__(name=name)
        # TODO: Maybe '/0' at the end is cumbersome and unnecessary in InputStep's
        self.inputs = []
        self.outputs = [Data(shape, self)]


def Input(shape, name=None):
    # Maybe this can be implemented in InputStep.__new__
    input = InputStep(shape, name)
    return input.outputs[0]  # Input produces exactly one Data output
