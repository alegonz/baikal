Introduction
============

What is baikal?
---------------

.. _sklearnAPIURL: https://scikit-learn.org/stable/developers/contributing.html#different-objects
.. _KerasURL: https://keras.io
.. _TensorFlowURL: https://www.tensorflow.org
.. _graphkitURL: https://github.com/yahoo/graphkit

**baikal is a graph-based, functional API for building complex machine learning pipelines
of objects that implement the** `scikit-learn API <sklearnAPIURL_>`__. It is mostly
inspired on the excellent `Keras <KerasURL_>`__ API for Deep Learning, and borrows a few
concepts from the `TensorFlow <TensorFlowURL_>`__ framework and the (perhaps lesser known)
`graphkit <graphkitURL_>`__ package.

**baikal** aims to provide an API that allows to build complex, non-linear machine learning
pipelines that looks like this:

.. image:: ../../illustrations/multiple_input_nonlinear_pipeline_example_diagram.png
    :alt: "An example of a multiple-input, nonlinear pipeline"

with code that looks like this:

.. code-block:: python

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

What can I do with it?
----------------------

With **baikal** you can

- build non-linear pipelines effortlessly
- handle multiple inputs and outputs
- add steps that operate on targets as part of the pipeline
- nest pipelines
- use prediction probabilities (or any other kind of output) as inputs to other steps in the pipeline
- query intermediate outputs, easing debugging
- freeze steps that do not require fitting
- define and add custom steps easily
- plot pipelines

All with boilerplate-free, readable code.

Why baikal?
-----------

.. _ComposeAPIURL: https://scikit-learn.org/stable/modules/compose.html#pipelines-and-composite-estimators
.. _ColumnTransformerURL: https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data
.. _StackingClassifierURL: http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/#stackingclassifier


The pipeline above (to the best of the author's knowledge) cannot be easily built using
`scikit-learn's composite estimators API <ComposeAPIURL_>`__ as you encounter some limitations:

1. It is aimed at linear pipelines

    * You could add some step parallelism with the `ColumnTransformer <ColumnTransformerURL_>`__ API,
      but this is limited to transformer objects.

2. Classifiers/Regressors can only be used at the end of the pipeline.

    * This means we cannot use the predicted labels (or their probabilities) as features
      to other classifiers/regressors.

    * You could leverage mlxtend's `StackingClassifier <StackingClassifierURL_>`__ and come
      up with some clever combination of the above composite estimators (``Pipeline``\ s,
      ``ColumnTransformer``\ s, and ``StackingClassifier``\ s, etc), but you might end up
      with code that feels hard-to-follow and verbose.

3. Cannot handle multiple input/multiple output models.

Perhaps you could instead define a big, composite estimator class that integrates each of
the pipeline steps through composition. This, however, most likely will require

* writing big ``__init__`` methods to control each of the internal steps' knobs;
* being careful with ``get_params`` and ``set_params`` if you want to use, say, ``GridSearchCV``;
* and adding some boilerplate code if you want to access the outputs of intermediate steps
  for debugging.

By using **baikal** as shown in the example above, code can be more readable, less verbose
and closer to our mental representation of the pipeline. **baikal** also provides an API
to fit, predict with, and query the entire pipeline with single commands, as we will see next.
