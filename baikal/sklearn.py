class SKLearnWrapper:
    def __init__(self, build_fn, **params):
        # build_fn must be a function that builds and returns a baikal Model
        self.build_fn = build_fn
        self._model = build_fn()
        self.set_params(**params)

    def get_params(self, deep=True):
        # deep=True is left for API compatibility purposes.
        # We will always include any nested params.
        params = self._model.get_params(deep=True)
        params['build_fn'] = self.build_fn
        return params

    def set_params(self, **params):
        self.build_fn = params.pop('build_fn', self.build_fn)
        self._model.set_params(**params)
        return self

    def fit(self, X, y=None, **fit_params):
        self._model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        # outputs argument currently unsupported
        return self._model.predict(X)

    @property
    def model(self):
        return self._model
