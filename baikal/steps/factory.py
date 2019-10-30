import inspect

from baikal import Step


def make_step(base_class):
    """Creates a step subclass from the given base class.

    For example, calling

        PCA = make_step(sklearn.decomposition.PCA)

    is equivalent to

        class PCA(Step, sklearn.decomposition.PCA):
            def __init__(self, name=None, function=None,
                         n_outputs=1, trainable=True, **kwargs):
                super().__init__(name=name, function=function,
                                 n_outputs=n_outputs, trainable=trainable,
                                 **kwargs)

    Parameters
    ----------
    base_class : type
        The base class to inherit from. It must implement the scikit-learn API.

    Returns
    -------
    step_subclass: type
        A new class that inherits from both Step and the given base class.

    """

    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super(self.__class__, self).__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs,
        )

    metaclass = type(base_class)
    name = base_class.__name__
    bases = (Step, base_class)
    caller_module = inspect.currentframe().f_back.f_globals["__name__"]
    dict = {"__init__": __init__, "__module__": caller_module}
    step_subclass = metaclass(name, bases, dict)

    return step_subclass
