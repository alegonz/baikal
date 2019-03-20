from baikal.core.step import Step


class DummySISO(Step):
    """Dummy step that takes a single input and produces a single output.
    """
    def __init__(self, name=None):
        super(DummySISO, self).__init__(name=name)

    def transform(self, x):
        return 2 * x

    def build_output_shapes(self, input_shapes):
        return input_shapes


class DummySIMO(Step):
    """Dummy step that takes a single input and produces multiple outputs.
    """
    def __init__(self, name=None):
        super(DummySIMO, self).__init__(name=name)

    def transform(self, x):
        return x + 1.0, x - 1.0

    def build_output_shapes(self, input_shapes):
        return input_shapes * 2


class DummyMISO(Step):
    """Dummy step that takes multiple inputs and produces a single output.
    """
    def __init__(self, name=None):
        super(DummyMISO, self).__init__(name=name)

    def transform(self, x1, x2):
        return x1 + x2

    def build_output_shapes(self, input_shapes):
        return input_shapes[0:1]


class DummyMIMO(Step):
    """Dummy step that takes multiple inputs and produces multiple outputs.
    """
    def __init__(self, name=None):
        super(DummyMIMO, self).__init__(name=name)

    def transform(self, x1, x2):
        return x1 * 10.0, x2 / 10.0

    def build_output_shapes(self, input_shapes):
        return input_shapes


class DummyWithoutTransform(Step):
    """Dummy step that lacks transform method.
    """
    def __init__(self, name=None):
        super(DummyWithoutTransform, self).__init__(name=name)

    def build_output_shapes(self, input_shapes):
        return input_shapes
