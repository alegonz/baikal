from baikal.core.step import Step


class Lambda(Step):
    def __init__(self, function, n_outputs=1, name=None):
        super(Lambda, self).__init__(name=name, function=function)
        self.n_outputs = n_outputs
