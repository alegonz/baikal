# Unfortunately we cannot name this module lambda.py so
# we are stuck with this unintuitive module name.
__all__ = ["Lambda"]

from functools import partial
from typing import Optional, Any, Callable, Union, List

from baikal.steps import Step
from baikal._core.data_placeholder import DataPlaceholder


class Lambda(Step):
    """Step for arbitrary functions.

    Parameters
    ----------
    compute_func
        The function to make the step from. This function a single array-like object
        (in the case of a single input) or a list of array-like objects (in the case of
        multiple inputs) If compute_func takes additional arguments you may either pass
        them as keyword arguments or use a functools.partial object.

    n_outputs
        Number of outputs of the function.

    name
        Name of the step (optional). If no name is passed, a name will be
        automatically generated.

    **kwargs
        Additional arguments to compute_func.

    Examples
    --------
    ::

        def function(Xs):
            x1, x2 = Xs
            return 2 * x1, x2 / 2

        x = Input()
        y1, y2 = Lambda(function, n_outputs=2)([x, x])
        model = Model(x, [y1, y2])

        x_data = np.array([[1.0, 2.0],
                           [3.0, 4.0]])

        y1_pred, y2_pred = model.predict(x_data)

        print(y1_pred)
        # [[2. 4.]
        # [6. 8.]]

        print(y2_pred)
        # [[0.5 1. ]
        # [1.5 2. ]]

    """

    def __init__(
        self,
        compute_func: Callable[..., Any],
        n_outputs: int = 1,
        name: Optional[str] = None,
        **kwargs
    ):
        self._compute_func = partial(compute_func, **kwargs)
        super().__init__(name=name, n_outputs=n_outputs)

    def __call__(
        self,
        inputs: Union[DataPlaceholder, List[DataPlaceholder]],
        targets: Optional[Union[DataPlaceholder, List[DataPlaceholder]]] = None,
        *,
        compute_func: Union[str, Callable[..., Any]] = "auto",
        fit_compute_func: Optional[Union[str, Callable[..., Any]]] = "auto",
        trainable: bool = True
    ) -> Union[DataPlaceholder, List[DataPlaceholder]]:
        """Call the step on input(s) (from previous steps) and generates the
        output(s) to be used in further steps.

        Parameters
        ----------
        inputs
            Input(s) to the step.

        targets
            Target(s) to the step.

        compute_func
            Ignored. This step will use the compute function passed at instantation.
            Kept for signature compatibility purposes.

        trainable
            Ignored. This step is always non-trainable.
            Kept for signature compatibility purposes.


        Returns
        -------
        DataPlaceholder
            Output(s) of the step.
        """
        # compute_func and trainable are ignored and kept for signature compatibility purposes
        return super().__call__(
            inputs, targets, compute_func=self._compute_func, trainable=False
        )

    # TODO: Consider adding get_params/set_params to tune function
