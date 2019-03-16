from baikal.core.utils import make_name


def is_data_list(l):
    return all([isinstance(d, Data) for d in l])


class Data:
    def __init__(self, shape, step, index=0):
        self.shape = shape
        self.step = step
        self.index = index
        self.name = make_name(step.name, index)
    # TODO: Implement __str__ and __repr__
