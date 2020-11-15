import warnings
from types import FrameType
from typing import Optional, Dict, Any, cast

__all__ = ["make_step"]

import inspect

from baikal.steps import Step


def make_step(
    base_class: type, attr_dict: Dict[str, Any] = None, class_name: Optional[str] = None
) -> type:
    """Creates a step subclass from the given base class.

    For example, calling::

        PCA = make_step(sklearn.decomposition.PCA, class_name="PCA")

    is equivalent to::

        class PCA(Step, sklearn.decomposition.PCA):
            def __init__(self, *args, name=None, n_outputs=1, **kwargs):
                super().__init__(*args, name=name, n_outputs=n_outputs, **kwargs)

    Parameters
    ----------
    base_class
        The base class to inherit from. It must implement the scikit-learn API.

    attr_dict
        Dictionary of additional attributes for the class. You can use this to add
        methods such as ``fit_compute`` to the class. (keys: name of attribute (``str``),
        values: attributes).

    class_name
        Name of the step class. If None, the name will be the name of the given
        base class. For instances made from the generated class to be pickle-able,
        you must pass a name that matches the name of the variable the generated
        class is being assigned to (the variable must also be declared at the top
        level of the module). **Deprecation notice**: This argument will be required
        from version 0.5.0.

    Returns
    -------
    step_subclass
        A new class that inherits from both Step and the given base class and has the
        the specified attributes.

    """

    def __init__(self, *args, name=None, n_outputs=1, **kwargs):
        super(self.__class__, self).__init__(
            *args, name=name, n_outputs=n_outputs, **kwargs,
        )

    metaclass = type(base_class)
    if class_name is None:
        warnings.warn(
            "Pass a string to `class_name`. From version 0.5.0 this argument will be"
            " required.",
            FutureWarning,
        )
        name = base_class.__name__
    else:
        name = class_name
    bases = (Step, base_class)
    caller_frame = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back)
    caller_module = caller_frame.f_globals["__name__"]

    dict = {"__init__": __init__, "__module__": caller_module}
    if attr_dict is not None:
        dict.update(attr_dict)

    step_subclass = metaclass(name, bases, dict)

    return step_subclass
