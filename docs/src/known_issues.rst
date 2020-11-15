Known issues
============

Pickle serialization/deserialization in models using CatBoost steps
-------------------------------------------------------------------

When trying to use a model loaded from a pickle file and that contains
`CatBoost <https://catboost.ai/>`__ steps, you might see the following error:

.. code-block::

    >>> model = joblib.load("model.pkl")
    >>> model.predict(data)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/venv/lib/python3.7/site-packages/baikal/_core/model.py", line 470, in predict
        X_norm, [], outputs, allow_unused_inputs=True, follow_targets=False
      File "/venv/lib/python3.7/site-packages/baikal/_core/model.py", line 191, in _get_required_nodes
        required_nodes |= backtrack(output)
      File "/venv/lib/python3.7/site-packages/baikal/_core/model.py", line 176, in backtrack
        parent_node = output.node
      File "/venv/lib/python3.7/site-packages/baikal/_core/data_placeholder.py", line 44, in node
        return self.step._nodes[self.port]
    AttributeError: 'CatBoostClassifierStep' object has no attribute '_nodes'

This is because CatBoost estimators (``CatBoostClassifier``, ``CatBoostRegressor``)
implement their own ``__getstate__`` and ``__setstate__`` methods and, if they are
not overridden appropriately, they won't include ``Step``-specific attributes in the
pickled result. The solution to this problem is to override the ``__getstate__`` and
``__setstate__`` methods to include the missing attributes as follows:

.. code-block:: python

    class CatBoostClassifierStep(Step, CatBoostClassifier):
        def __init__(self, *args, name=None, n_outputs=1, **kwargs):
            super().__init__(*args, name=name, n_outputs=n_outputs, **kwargs)

        def __getstate__(self):
            state = super().__getstate__()
            state["_name"] = self._name
            state["_nodes"] = self._nodes
            state["_n_outputs"] = self._n_outputs
            return state

        def __setstate__(self, state):
            self._name = state.pop("_name")
            self._nodes = state.pop("_nodes")
            self._n_outputs = state.pop("_n_outputs")
            super().__setstate__(state)
