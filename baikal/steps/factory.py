import inspect

from baikal import Step


def make_step(base_class, n_outputs=1):
    """Creates a step subclass from the given base class.

    For example, calling

        PCA = make_step(sklearn.decomposition.PCA)

    is equivalent to

        class PCA(Step, sklearn.decomposition.PCA):
            def __init__(self, name=None, trainable=True, function=None, **kwargs):
                super(PCA, self).__init__(name=name, trainable=trainable,
                                          function=function, **kwargs)
                self.n_outputs = 1

    Parameters
    ----------
    base_class : type
        The base class to inherit from. It must implement the scikit-learn API.

    n_outputs : int, optional (default=1)
        Number of outputs the step instances return when called (default=1)

    Returns
    -------
    step_subclass: type
        A new class that inherits from both Step and the given base class.

    """

    def __init__(self, name=None, trainable=True, function=None, **kwargs):
        super(self.__class__, self).__init__(name=name, trainable=trainable,
                                             function=function, **kwargs)
        self.n_outputs = n_outputs

    metaclass = type(base_class)
    name = base_class.__name__
    bases = (Step, base_class)
    caller_module = inspect.currentframe().f_back.f_globals['__name__']
    dict = {'__init__': __init__,
            '__module__': caller_module}
    step_subclass = metaclass(name, bases, dict)

    return step_subclass
