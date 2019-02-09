from baikal.core.utils import make_name


def is_data_list(l):
    return all([isinstance(d, Data) for d in l])


class Data:
    def __init__(self, shape, node, index=0):
        self.shape = shape
        self.node = node
        self.index = index
        self.name = make_name(node.name, index)
