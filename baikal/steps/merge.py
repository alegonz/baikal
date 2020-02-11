__all__ = ["Concatenate", "Stack", "ColumnStack", "Split"]

import numpy as np

from baikal.steps import Step
from baikal._core.utils import listify as _listify


class Concatenate(Step):
    """Step for concatenating arrays.

    Parameters
    ----------
    axis
        The axis of concatenation (default is -1, the last axis).

    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.

    """

    def __init__(self, axis=-1, name=None):
        super().__init__(name=name)
        self.axis = axis

    def transform(self, Xs):
        return np.concatenate(_listify(Xs), axis=self.axis)


class Stack(Step):
    """Step for stacking arrays.

    Parameters
    ----------
    axis
        The axis parameter specifies the index of the new axis in the dimensions of
        the result (default is -1).

    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.
    """

    def __init__(self, axis=-1, name=None):
        super().__init__(name=name)
        self.axis = axis

    def transform(self, Xs):
        return np.stack(_listify(Xs), axis=self.axis)


class ColumnStack(Step):
    """Step for stacking arrays along the columns.

    Parameters
    ----------
    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def transform(self, Xs):
        return np.column_stack(_listify(Xs))


class Split(Step):
    """Step for splitting arrays.

    Parameters
    ----------
    indices_or_sections
        If an integer (N) is passed, the array will be divided into N equal arrays along
        axis. If an 1-D array of sorted integers is passed, the entries indicate where
        along axis the array is split.

    axis
        The axis on where to split the array (default is -1, the last axis).

    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.
    """

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
