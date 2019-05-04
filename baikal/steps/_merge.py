import numpy as np

from baikal.core.step import Step


class Concatenate(Step):
    def __init__(self, axis=-1, name=None):
        super(Concatenate, self).__init__(name=name)
        self.axis = axis
        self.n_outputs = 1

    def transform(self, *Xs):
        return np.concatenate(Xs, axis=self.axis)


class Stack(Step):
    def __init__(self, axis=-1, name=None):
        super(Stack, self).__init__(name=name)
        self.axis = axis
        self.n_outputs = 1

    def transform(self, *Xs):
        return np.stack(Xs, axis=self.axis)
