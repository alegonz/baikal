# Unfortunately we cannot name this module lambda.py so
# we are stuck with this unintuitive module name.

from baikal._core.step import Step


class Lambda(Step):
    def __init__(self, function, n_outputs=1, name=None):
        super(Lambda, self).__init__(name=name, function=function)
        self.n_outputs = n_outputs
