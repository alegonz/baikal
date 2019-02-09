from baikal.core.data import is_data_list
from baikal.core.utils import listify


class Model:
    def __init__(self, inputs, outputs):
        inputs = listify(inputs)
        outputs = listify(outputs)

        if not is_data_list(inputs) or not is_data_list(outputs):
            raise ValueError('inputs and outputs must be of type Data.')

        self.inputs = inputs
        self.outputs = outputs
        # TODO: Get list of necessary steps (topological sort)

    def fit(self, inputs, outputs):
        pass
