__all__ = ["Concatenate", "Stack", "ColumnStack", "Split"]

import numpy as np

from baikal.steps import Step
from baikal._core.utils import listify as _listify


class Concatenate(Step):
    """Step for numpy's concatenate function."""

    def __init__(self, axis=-1, name=None):
        super().__init__(name=name)
        self.axis = axis

    def transform(self, Xs):
        return np.concatenate(_listify(Xs), axis=self.axis)


class Stack(Step):
    """Step for numpy's stack function."""

    def __init__(self, axis=-1, name=None):
        super().__init__(name=name)
        self.axis = axis

    def transform(self, Xs):
        return np.stack(_listify(Xs), axis=self.axis)


class ColumnStack(Step):
    """Step for numpy's column_stack function."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def transform(self, Xs):
        return np.column_stack(_listify(Xs))


class Split(Step):
    """Step for numpy's concatenate function."""

    def __init__(self, indices_or_sections, axis=-1, name=None):
        try:
            n_outputs = len(indices_or_sections) + 1
        except TypeError:
            n_outputs = indices_or_sections
        super().__init__(name=name, n_outputs=n_outputs)
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def transform(self, X):
        return np.split(X, self.indices_or_sections, axis=self.axis)
