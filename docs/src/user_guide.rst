User guide
==========

Key concepts
------------

.. TODO: Render here the docstrings of each class instead of copy-pasting

The baikal API introduces three basic elements:

* **Step**: Steps are the building blocks of the API. Conceptually similar to TensorFlow's
  operations and Keras layers, each Step is a unit of computation (e.g. PCA, Logistic Regression)
  that take the data from preceding Steps and produce data to be used by other Steps further
  in the pipeline. Steps are defined by combining the ``Step`` mixin class with a base class
  that implements the scikit-learn API. This is explained in more detail below.

* **DataPlaceholder**: The inputs and outputs of Steps. If Steps are like TensorFlow
  operations or Keras layers, then DataPlaceHolders are akin to tensors. Don't be misled
  though, DataPlaceholders are just minimal, low-weight auxiliary objects whose main
  purpose is to keep track of the input/output connectivity between steps, and serve as
  the keys to map the actual input data to their appropriate Step. They are not arrays/tensors,
  nor contain any shape/type information whatsoever.

* **Model**: A Model is a network (more precisely, a directed acyclic graph) of Steps,
  and it is defined from the input/output specification of the pipeline. Models have fit
  and predict routines that, together with graph-based engine, allow the automatic (feed-forward)
  computation of each of the pipeline steps when fed with data.

Quick-start guide
-----------------

Without further ado, here's a short example of a simple SVC model built with **baikal**:

.. code-block:: python

    import sklearn.svm
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    from baikal import make_step, Input, Model


    # 1. Define a step
    SVC = make_step(sklearn.svm.SVC)

    # 2. Build the model
    x = Input()
    y_t = Input()
    y = SVC(C=1.0, kernel="rbf", gamma=0.5)(x, y_t)
    model = Model(x, y, y_t)

    # 3. Train the model
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, random_state=0
    )

    model.fit(X_train, y_train)

    # 4. Use the model
    y_test_pred = model.predict(X_test)


API walkthrough
---------------

.. module:: baikal

As shown in the short example above, the **baikal** API consists of four basic steps:

.. contents:: :local:

Let's take a look at each of them in detail. Full examples can be found in the project's
`examples <examples>`__ folder.

1. Define the steps
~~~~~~~~~~~~~~~~~~~

A step is defined very easily, just feed the provided ``make_step`` function with the
class you want to make a step from:

.. code-block:: python

    import sklearn.linear_model
    from baikal import make_step

    LogisticRegression = make_step(sklearn.linear_model.LogisticRegression)

You can make a step from any class you like, so long that class implements the
`scikit-learn API <https://scikit-learn.org/stable/developers/contributing.html#different-objects>`__.

What this function is doing under the hood, is to combine the given class with the ``Step``
mixin class. The ``Step`` mixin, among other things, endows the given class with a
``__call__`` method, making the class callable on the outputs (``DataPlaceholder`` objects)
of previous steps. If you prefer to do this manually, you only have to:

1. Define a class that inherits from both the ``Step`` mixin and the class you wish to
   make a step of (in that order!).
2. In the class ``__init__``, call ``super().__init__(...)`` and pass the appropriate
   step parameters.

For example, to make a step for ``sklearn.linear_model.LogisticRegression`` we do:

.. code-block:: python

    import sklearn.linear_model
    from baikal import Step

    # The order of inheritance is important!
    class LogisticRegression(Step, sklearn.linear_model.LogisticRegression):
        def __init__(self, name=None, n_outputs=1, **kwargs):
            super().__init__(name=name,n_outputs=n_outputs,**kwargs)

Other steps are defined similarly (omitted here for brevity).

**baikal** can also handle steps with multiple input/outputs/targets. The base class may
implement a ``predict``/``transform`` method (the compute function) that take multiple
inputs and returns multiple outputs, and a fit method that takes multiple inputs and targets
(native scikit-learn classes at present take one input, return one output, and take at
most one target). In this case, the input/target arguments are expected to be a list of
(typically) array-like objects, and the compute function is expected to return a list of
array-like objects. For example, the base class may implement the methods like this:

.. code-block:: python

    class SomeClass(BaseEstimator):
        ...
        def predict(self, Xs):
            X1, X2 = Xs
            # use X1, X2 to calculate y1, y2
            return y1, y2

        def fit(self, Xs, ys):
            (X1, X2), (y1, y2) = Xs, ys
            # use X1, X2, y1, y2 to fit the model
            return self

2. Build the model
~~~~~~~~~~~~~~~~~~

Once we have defined the steps, we can make a model like shown below. First, you create
the initial step, that serves as the entry-point to the model, by calling the ``Input``
helper function. This outputs a DataPlaceholder representing one of the inputs to the
model. Then, all you have to do is to instantiate the steps and call them on the outputs
(DataPlaceholders from previous steps) as you deem appropriate. Finally, you instantiate
the model with the inputs, outputs and targets (also DataPlaceholders) that specify your
pipeline.

This style should feel familiar to users of Keras.

Note that steps that require target data (like ``ExtraTreesClassifier``, ``RandomForestClassifier``,
``LogisticRegression`` and ``SVC``) are called with two arguments. These arguments correspond
to the inputs (e.g. ``x1``, ``x2``) and targets (e.g. ``y_t``) of the step. These targets
are specified to the Model at instantiation via the third argument. **baikal** pipelines
are made of complex, heterogenous, non-differentiable steps (e.g. a whole ``RandomForestClassifier``,
with its own internal learning algorithm), so there's no some magic automatic differentiation
that backpropagates the target information from the outputs to the appropriate steps, so
we must specify which step needs which targets directly.

.. code-block:: python

    from baikal import Input, Model
    from baikal.steps import Stack

    # Assume the steps below were already defined
    x1 = Input()
    x2 = Input()
    y_t = Input()

    y1 = ExtraTreesClassifier()(x1, y_t)
    y2 = RandomForestClassifier()(x2, y_t)
    z = PowerTransformer()(x2)
    z = PCA()(z)
    y3 = LogisticRegression()(z, y_t)

    ensemble_features = Stack()([y1, y2, y3])
    y = SVC()(ensemble_features, y_t)

    model = Model([x1, x2], y, y_t)

You can call the same step on different inputs and targets to reuse the step (similar to
the concept of shared layers and nodes in Keras), and specify a different ``compute_func``/``trainable``
configuration on each call. This is achieved via "ports": each call creates a new port
and associates the given configuration to it. You may access the configuration at each
port using the ``get_*_at(port)`` methods.

(*) Steps are called on and output DataPlaceholders. DataPlaceholders are produced and
consumed exclusively by Steps, so you do not need to instantiate these yourself.

3. Train the model
~~~~~~~~~~~~~~~~~~

Now that we have built a model, we are ready to train it. The model also follows the
scikit-learn API, as it has a ``fit`` method:

.. code-block:: python

    model.fit(X=[X1_train, X2_train], y=y_train)

.. autofunction:: baikal.Model.fit
    :noindex:

4. Use the model
~~~~~~~~~~~~~~~~

To predict with the model, use the ``predict`` method and pass it the input data like you
would for the ``fit`` method. The model will automatically propagate the inputs through
all the steps and produce the outputs specified at instantiation.

.. code-block:: python

    y_test_pred = model.predict([X1_test, X2_test])

    # This also works:
    y_test_pred = model.predict({x1: X1_test, x2: X2_test})

.. autofunction:: baikal.Model.predict
    :noindex:

**Models are query-able**. That is, you can request other outputs other than those specified
at model instantiation. This allows querying intermediate outputs and ease debugging.
For example, to get both the output from ``PCA`` and the ``ExtraTreesClassifier``:

.. code-block:: python

    outs = model.predict(
        [X1_test, X2_test], output_names=["ExtraTreesClassifier_0:0/0", "PCA_0:0/0"]
    )

You don't need to pass inputs that are not required to compute the queried output.
For example, if we just want the output of ``PowerTransformer``:

.. code-block:: python

    outs = model.predict({x2: X2_data}, output_names="PowerTransformer_0:0/0")

**Models are also nestable**. In fact, Models are steps, too. This allows composing smaller
models into bigger ones, like so:

.. code-block:: python

    # Assume we have two previously built complex
    # classifier models, perhaps loaded from a file.
    submodel1 = ...
    submodel2 = ...

    # Now we make an stacked classifier from both submodels
    x = Input()
    y_t = Input()
    y1 = submodel1(x)
    y2 = submodel2(x, y_t)
    z = Stack()([y1, y2])
    y = SVC()(z, y_t)
    bigmodel = Model(x, y, y_t)

..  Additional concepts
    -------------------

    TODO

Utilities
---------

.. _SecMaintLimsURL: https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
.. _GridSearchCVURL: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn-model-selection-gridsearchcv
.. _pydotURL: https://pypi.org/project/pydot
.. _graphvizURL: https://graphviz.gitlab.io

Persisting the model
~~~~~~~~~~~~~~~~~~~~

Like native scikit-learn objects, models can be serialized with pickle or joblib without
any extra setup:

.. code-block:: python

    import joblib
    joblib.dump(model, "model.pkl")
    model_reloaded = joblib.load("model.pkl")

Keep in mind, however, the `security and maintainability limitations <SecMaintLimsURL_>`__
of these formats.

scikit-learn wrapper for ``GridSearchCV``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, **baikal** also provides a wrapper utility class that allows models to used
in scikit-learn's `GridSearchCV API <GridSearchCVURL_>`__. Below there's a code snippet
showing its usage. It follows the style of Keras' own wrapper.

See :ref:`Tune a model with ``GridSearchCV``` for an example script of this utility.

A future release of **baikal** plans to include a custom ``GridSearchCV`` API, based on
the original scikit-learn implementation, that can handle baikal models natively, avoiding
a couple of gotchas with the current wrapper implementation (mentioned below).

.. code-block:: python

    # 1. Define a function that returns your baikal model
    def build_fn():
        x = Input()
        y_t = Input()
        h = PCA(random_state=random_state, name="pca")(x)
        y = LogisticRegression(random_state=random_state, name="classifier")(h, y_t)
        model = Model(x, y, y_t)
        return model

    # 2. Define a parameter grid
    # - keys have the [step-name]__[parameter-name] format, similar to sklearn Pipelines
    # - You can also search over the steps themselves using [step-name] keys
    param_grid = [
        {
            "classifier": [LogisticRegression()],
            "classifier__C": [0.01, 0.1, 1],
            "pca__n_components": [1, 2, 3, 4],
        },
        {
            "classifier": [RandomForestClassifier()],
            "classifier__n_estimators": [10, 50, 100],
        },
    ]

    # 3. Instantiate the wrapper
    sk_model = SKLearnWrapper(build_fn)

    # 4. Use GridSearchCV as usual
    gscv_baikal = GridSearchCV(sk_model, param_grid)
    gscv_baikal.fit(x_data, y_data)
    best_model = gscv_baikal.best_estimator_.model

Currently there are a couple of gotchas:

* The ``cv`` argument of ``GridSearchCV`` will default to KFold if the estimator is a
  baikal Model, so you have to specify an appropriate splitter directly if you need another
  splitting scheme.
* ``GridSearchCV`` cannot handle models with multiple inputs/outputs. A way to work around
  this is to split the input data and merge the outputs within the model.

Plotting your model
~~~~~~~~~~~~~~~~~~~

The baikal package includes a plot utility.

.. code-block:: python

    from baikal.plot import plot_model
    plot_model(model, filename="model.png")

In order to use the plot utility, you need to install `pydot <pydotURL_>`__ and
`graphviz <graphvizURL_>`__.

For the example above, it produces this:

.. image:: ../../illustrations/multiple_input_nonlinear_pipeline_example_plot.png
    :alt: "An example of a multiple-input, nonlinear pipeline rendered with the plot utility"
    :height: 600
