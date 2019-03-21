from baikal.core.utils import make_repr


def is_data_list(l):
    return all([isinstance(d, Data) for d in l])


class Data:
    def __init__(self, shape, step, name):
        self.shape = shape
        self.step = step
        self.name = name

    def __repr__(self):
        attrs = ['shape', 'step', 'name']
        return make_repr(self, attrs)
