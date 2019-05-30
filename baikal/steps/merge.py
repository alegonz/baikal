import numpy as np

from baikal._core.step import Step


class Concatenate(Step):
    """Step for numpy's concatenate function."""
    def __init__(self, axis=-1, name=None):
        super(Concatenate, self).__init__(name=name)
        self.axis = axis
        self.n_outputs = 1

    def transform(self, *Xs):
        return np.concatenate(Xs, axis=self.axis)


class Stack(Step):
    """Step for numpy's stack function."""
    def __init__(self, axis=-1, name=None):
        super(Stack, self).__init__(name=name)
        self.axis = axis
        self.n_outputs = 1

    def transform(self, *Xs):
        return np.stack(Xs, axis=self.axis)


class ColumnStack(Step):
    """Step for numpy's column_stack function."""
    def __init__(self, name=None):
        super(ColumnStack, self).__init__(name=name, trainable=False)
        self.n_outputs = 1

    def transform(self, *Xs):
        return np.column_stack(Xs)
