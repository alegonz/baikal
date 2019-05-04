from baikal.core.step import Step


class Lambda(Step):
    def __init__(self, function, n_outputs=1, name=None):
        super(Lambda, self).__init__(name=name)
        self.function = function
        self.n_outputs = n_outputs

    def transform(self, *Xs):
        return self.function(*Xs)
