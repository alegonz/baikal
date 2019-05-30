from baikal._core.utils import make_repr


def is_data_placeholder_list(l):
    return all([isinstance(item, DataPlaceholder) for item in l])


class DataPlaceholder:
    """Auxiliary class that represents the inputs and outputs of Steps.

    DataPlaceholders are just minimal, low-weight auxiliary objects whose main
    purpose is to keep track of the input/output connectivity between steps, and
    serve as the keys to map the actual input data to their appropriate Step.
    They are not arrays/tensors, nor contain any shape/type information whatsoever.

    Steps are called on and output DataPlaceHolders. DataPlaceholders are
    produced and consumed exclusively by Steps, so you shouldn't need to
    instantiate these yourself.
    """
    def __init__(self, step, name):
        self.step = step
        self.name = name

    def __repr__(self):
        attrs = ['step', 'name']
        return make_repr(self, attrs)

    # Make it sortable to aid cache hits in Model._get_required_steps
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.name < other.name
        return NotImplemented
