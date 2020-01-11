import inspect

from baikal import Step


def make_step(base_class, attr_dict=None):
    """Creates a step subclass from the given base class.

    For example, calling

        PCA = make_step(sklearn.decomposition.PCA)

    is equivalent to

        class PCA(Step, sklearn.decomposition.PCA):
            def __init__(self, name=None, n_outputs=1, **kwargs):
                super().__init__(name=name, n_outputs=n_outputs, **kwargs)

    Parameters
    ----------
    base_class : type
        The base class to inherit from. It must implement the scikit-learn API.

    attr_dict : dict
        Dictionary of additional attributes for the class. You can use this to add
        methods such as `fit_compute` to the class. (keys: name of attribute (str),
        values: attributes).

    Returns
    -------
    step_subclass: type
        A new class that inherits from both Step and the given base class and has the
        the specified attributes.

    """

    def __init__(self, name=None, n_outputs=1, **kwargs):
        super(self.__class__, self).__init__(
            name=name, n_outputs=n_outputs, **kwargs,
        )

    metaclass = type(base_class)
    name = base_class.__name__
    bases = (Step, base_class)
    caller_module = inspect.currentframe().f_back.f_globals["__name__"]

    dict = {"__init__": __init__, "__module__": caller_module}
    if attr_dict is not None:
        dict.update(attr_dict)

    step_subclass = metaclass(name, bases, dict)

    return step_subclass
