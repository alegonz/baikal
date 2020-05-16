"""Global configuration parameters

Follows the same interface of sklearn's config API.
"""
from typing import Dict, Any, Optional

_config = {"print_changed_only": True}


def get_config() -> Dict[str, Any]:
    """Get global configuration parameters.

    Returns
    -------
    Global configuration parameters.
    """
    return _config.copy()


def set_config(print_changed_only: Optional[bool] = None):
    """Set global configuration parameters.

    Parameters
    ----------
    print_changed_only
        If ``True``, only the parameters (of the step's parent estimator) that were set
        to non-default values will be printed when printing a step. The ``name`` and
        ``n_outputs`` parameters are always printed. For example, if there is a ``SomeStep``
        step class that inherited from an estimator (or any class that implements the
        scikit-learn API) with signature ``(x=123, y='foo')``, setting the flag to ``True``
        would print ``SomeStep(name='SomeStep_0', n_outputs=1)`` and setting to ``False``
        would print ``SomeStep(x=123, y='foo', name='SomeStep_0', n_outputs=1)``.
    """
    if print_changed_only is not None:
        _config["print_changed_only"] = print_changed_only
