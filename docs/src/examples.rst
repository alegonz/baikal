Examples
========

Stacked classifiers (naive protocol)
------------------------------------

Similar to the the example in the quick-start guide, (a naive) stacks of classifiers
(or regressors) can be built like shown below. Note that you can specify the function
the step should use for computation, in this case ``compute_func='predict_proba'`` to
use the label probabilities as the features of the meta-classifier.

.. code-block:: python

    x = Input()
    y_t = Input()
    y_p1 = LogisticRegression()(x, y_t, compute_func="predict_proba")
    y_p2 = RandomForestClassifier()(x, y_t, compute_func="predict_proba")
    # predict_proba returns arrays whose columns sum to one, so we drop one column
    drop_first_col = Lambda(lambda array: array[:, 1:])
    y_p1 = drop_first_col(y_p1)
    y_p2 = drop_first_col(y_p2)
    ensemble_features = ColumnStack()([y_p1, y_p2])
    y_p = ExtraTreesClassifier()(ensemble_features, y_t)

    model = Model(x, y_p, y_t)

.. container:: toggle

    .. literalinclude:: ../../examples/stacked_classifiers_naive.py

Stacked classifiers (standard protocol)
---------------------------------------

In the naive stack above, each classifier in the 1st level will calculate the predictions
for the 2nd level using the same data it used for fitting its parameters. This is prone
to overfitting as the 2nd level classifier will tend to give more weight to an overfit
classifier in the 1st level. To avoid this, the standard protocol recommends that, during
fit, the 1st level classifiers are still trained on the original data, but instead they
provide out-of-fold (OOF) predictions to the 2nd level classifier. To achieve this special
behavior, we leverage the ``fit_compute_func`` API: we define a ``fit_predict`` method
that does the fitting and the OOF predictions, and add it as a method of the 1st level
classifiers (``LogisticRegression`` and ``RandomForestClassifier``, in the example) when
making the steps. **baikal** will then detect and use this method during fit.

.. code-block:: python

    from sklearn.model_selection import cross_val_predict


    def fit_predict(self, X, y):
        self.fit(X, y)
        return cross_val_predict(self, X, y, method="predict_proba")


    attr_dict = {"fit_predict": fit_predict}

    # 1st level classifiers
    LogisticRegression = make_step(sklearn.linear_model.LogisticRegression, attr_dict)
    RandomForestClassifier = make_step(sklearn.ensemble.RandomForestClassifier, attr_dict)

    # 2nd level classifier
    ExtraTreesClassifier = make_step(sklearn.ensemble.ExtraTreesClassifier)

The rest of the stack is build exactly the same as in the naive example.

.. container:: toggle

    .. literalinclude:: ../../examples/stacked_classifiers_standard.py

Classifier chain
----------------

.. _ClassifierChainWikiURL: https://en.wikipedia.org/wiki/Classifier_chains
.. _ClassifierChainURL: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html#sklearn.multioutput.ClassifierChain
.. _RegressorChainURL: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html#sklearn.multioutput.RegressorChain

The API also lends itself for more interesting configurations, such as that of
`classifier chains <ClassifierChainWikiURL_>`__. By leveraging the API and Python's own
control flow, a classifier chain model can be built as follows:

.. code-block:: python

    x = Input()
    y_t = Input()
    order = list(range(n_targets))
    random.shuffle(order)

    squeeze = Lambda(np.squeeze, axis=1)

    ys_t = Split(n_targets, axis=1)(y_t)
    ys_p = []
    for j, k in enumerate(order):
        x_stacked = ColumnStack()([x, *ys_p[:j]])
        ys_t[k] = squeeze(ys_t[k])
        ys_p.append(LogisticRegression()(x_stacked, ys_t[k]))

    ys_p = [ys_p[order.index(j)] for j in range(n_targets)]
    y_p = ColumnStack()(ys_p)

    model = Model(x, y_p, y_t)

Sure, scikit-learn already does have `ClassifierChain <ClassifierChainURL_>`__ and
`RegressorChain <RegressorChainURL_>`__ classes for this. But with **baikal** you could,
for example, mix classifiers and regressors to predict multilabels that include both
categorical and continuous labels.

.. container:: toggle

    .. literalinclude:: ../../examples/classifier_chain.py

Transformed target
------------------

You can also call steps on the targets to apply transformations on them. Note that by
making the transformer a shared step, you can re-use learned parameters to apply the
inverse transform later in the pipeline.

.. code-block:: python

    transformer = QuantileTransformer(n_quantiles=300, output_distribution="normal")

    x = Input()
    y_t = Input()
    # QuantileTransformer requires an explicit feature dimension, hence the Lambda step
    y_t_trans = Lambda(np.reshape, newshape=(-1, 1))(y_t)
    y_t_trans = transformer(y_t_trans)
    y_p_trans = RidgeCV()(x, y_t_trans)
    y_p = transformer(y_p_trans, compute_func="inverse_transform", trainable=False)
    # Note that transformer is a shared step since it was called twice

    model = Model(x, y_p, y_t)

.. container:: toggle

    .. literalinclude:: ../../examples/transformed_target.py
