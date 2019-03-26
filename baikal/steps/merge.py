import numpy as np

from baikal.core.step import Step


# TODO: Think carefully about sample axis
class Concatenate(Step):
    def __init__(self, axis=-1, name=None):
        super(Concatenate, self).__init__(name=name)
        self.axis = axis

    def transform(self, *Xs):
        return np.concatenate(Xs, axis=self.axis)

    def build_output_shapes(self, input_shapes):
        # Check input_shapes
        ndims = [len(shape) for shape in input_shapes]
        if len(set(ndims)) != 1:
            raise ValueError('All inputs must have the same number of dimensions.')
        ndim = ndims[0]

        axis = self.axis % ndim
        if axis >= ndim:
            raise ValueError('The specified axis is out of range. Got shapes of {} dimensions '
                             'but axis={}'.format(ndim, self.axis))

        common_shapes = [tuple(ax for k, ax in enumerate(shape) if k != axis)
                         for shape in input_shapes]
        if len(set(common_shapes)) != 1:
            raise ValueError('All inputs must have matching shapes, except for the concatenation axis.')
        common_shape = common_shapes[0]

        concat_ax = sum(shape[axis] for shape in input_shapes)
        output_shape = common_shape[:axis] + (concat_ax,) + common_shape[axis:]

        return [output_shape]


class Stack(Step):
    def __init__(self, axis=-1, name=None):
        super(Stack, self).__init__(name=name)
        self.axis = axis

    def transform(self, *Xs):
        return np.stack(Xs, axis=self.axis)

    def build_output_shapes(self, input_shapes):
        # Check input_shapes
        if len(set(input_shapes)) != 1:
            raise ValueError('All inputs must have the same shape.')
        shape = input_shapes[0]
        ndim = len(shape)

        axis = self.axis % ndim
        if axis >= ndim:
            raise ValueError('The specified axis is out of range. Got shapes of {} dimensions '
                             'but axis={}'.format(ndim, self.axis))

        output_shape = shape[:axis] + (len(input_shapes),) + shape[axis:]

        return [output_shape]
