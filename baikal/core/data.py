def is_data_list(l):
    return all([isinstance(d, Data) for d in l])


class Data:
    def __init__(self, shape, step, name):
        self.shape = shape
        self.step = step
        self.name = name
    # TODO: Implement __str__ and __repr__
