from baikal import Step


class DummySISO(Step):
    """Dummy step that takes a single input and produces a single output.
    """
    def __init__(self, name=None):
        super(DummySISO, self).__init__(name=name)

    def transform(self, x):
        return 2 * x


class DummySIMO(Step):
    """Dummy step that takes a single input and produces multiple outputs.
    """
    def __init__(self, name=None):
        super(DummySIMO, self).__init__(name=name, n_outputs=2)

    def transform(self, x):
        return x + 1.0, x - 1.0


class DummyMISO(Step):
    """Dummy step that takes multiple inputs and produces a single output.
    """
    def __init__(self, name=None):
        super(DummyMISO, self).__init__(name=name)

    def transform(self, x1, x2):
        return x1 + x2


class DummyMIMO(Step):
    """Dummy step that takes multiple inputs and produces multiple outputs.
    """
    def __init__(self, name=None):
        super(DummyMIMO, self).__init__(name=name, n_outputs=2)

    def transform(self, x1, x2):
        return x1 * 10.0, x2 / 10.0


class DummyImproperlyDefined(Step):
    """Dummy step that returns two outputs but defines only one.
    """
    def __init__(self, name=None):
        super(DummyImproperlyDefined, self).__init__(name=name)

    def transform(self, x):
        return x + 1.0, x - 1.0
