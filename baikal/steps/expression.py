# Unfortunately we cannot name this module lambda.py so
# we are stuck with this unintuitive module name.
from typing import Optional, Any, Callable

from baikal._core.step import Step


class Lambda(Step):
    """Step for arbitrary functions.

    Parameters
    ----------
    function
        The function to make the step from.

    n_outputs
        Number of outputs of function.

    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.

    Examples
    --------
    >>> def function(x1, x2):
    >>>     return 2 * x1, x2 / 2
    >>>
    >>> x = Input()
    >>> y1, y2 = Lambda(function, n_outputs=2)([x, x])
    >>> model = Model(x, [y1, y2])
    >>>
    >>> x_data = np.array([[1.0, 2.0],
    >>>                    [3.0, 4.0]])
    >>>
    >>> y1_pred, y2_pred = model.predict(x_data)
    >>> y1_pred
    [[2. 4.]
     [6. 8.]]
    >>> y2_pred
    [[0.5 1. ]
     [1.5 2. ]]
    """
    def __init__(self,
                 function: Callable[..., Any],
                 n_outputs: int = 1,
                 name: Optional[str] = None):
        super(Lambda, self).__init__(name=name, trainable=False, function=function)
        self.n_outputs = n_outputs

    # TODO: Consider adding get_params/set_params to tune function
