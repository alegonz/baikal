__all__ = ["SKLearnWrapper"]

from typing import Dict, Any, Callable

from baikal import Model


class SKLearnWrapper:
    """Wrapper utility class that allows models to used in scikit-learn's
    ``GridSearchCV`` API. It follows the style of Keras' own wrapper.

    A future release of **baikal** plans to remove this class and instead
    include a custom ``GridSearchCV`` API, based on the original scikit-learn
    implementation, that can handle baikal models natively.

    Parameters
    ----------
    build_fn
        A function that takes no arguments and builds and returns a baikal Model.

        Note that, in order to specify which parameters of which steps to tune
        using a dictionary keyed by ``<step>__<parameter>``, you *must* pass a
        name to the appropriate steps when building the model in this function.

    params
        Dictionary mapping parameter names to their values. Valid parameter
        names are 'build_fn' and any parameter the wrapped model can take
        (in the form ``<step>__<parameter>``).
    """

    def __init__(self, build_fn: Callable[[], Model], **params):
        # build_fn must be a function that builds and returns a baikal Model
        self.build_fn = build_fn
        self._model = build_fn()
        self.set_params(**params)

    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep
            Unused. Kept for API compatibility purposes. It will always include
            any nested params.

        Returns
        -------
        params
            Parameter names mapped to their values.
        """
        params = self._model.get_params(deep=True)
        params["build_fn"] = self.build_fn
        return params

    def set_params(self, **params) -> "SKLearnWrapper":
        """Set the parameters of this estimator.

        Parameters
        ----------
        params
            Dictionary mapping parameter names to their values. Valid parameter
            names are 'build_fn' and any parameter the wrapped model can take
            (in the form ``<step-name>__<parameter-name>``).

        Returns
        -------
        self
        """
        self.build_fn = params.pop("build_fn", self.build_fn)
        self._model.set_params(**params)
        return self

    def fit(self, X, y=None, **fit_params):
        """Fit wrapped model.

        Parameters
        ----------
        X
            Input data to the model.
        y
            Target data to the model.
        fit_params
            Parameters passed to the fit method of each model step, where each
            parameter name has the form ``<step-name>__<parameter-name>``.

        Returns
        -------
        self
        """
        self._model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        """Predict with the wrapped model.

        Parameters
        ----------
        X
            Input data to the model.

        Returns
        -------
        Model predictions.

        Notes
        -----
        outputs argument is currently unsupported.
        """
        return self._model.predict(X)

    @property
    def model(self):
        """Get the wrapped model."""
        return self._model
