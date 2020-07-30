import numpy as np
from numpy.testing import assert_array_equal
from baikal.steps import ColumnStack, Concatenate, Stack, Split

import joblib
from tests.helpers.fixtures import teardown # NOQA
from tests.steps.classifier import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from baikal import Input, Model, make_step


def test_columnstack(teardown):
    x1 = Input()
    x2 = Input()
    y = ColumnStack()([x1, x2])
    model = Model([x1, x2], y)

    x1_data = np.array([1, 10, 100])
    x2_data = np.array([2, 20, 200])

    y_expected = np.column_stack([x1_data, x2_data])

    model.fit([x1_data, x2_data])

    f = joblib.dump(model, 't.pkl')
    l = joblib.load('t.pkl')
    y_pred = l.predict([x1_data, x2_data])

    assert_array_equal(y_pred, y_expected)


def test_make_step(teardown):
    # 1. Define a step


    # 2. Build the model
    x = Input()
    y_t = Input()
    y_p = SVC(C=1.0, kernel="rbf", gamma=0.5)(x, y_t)

    model = Model(x, y_p, y_t)

    # 3. Train the model
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, random_state=0
    )

    model.fit(X_train, y_train)
    f = joblib.dump(model, 't.pkl')

    model = joblib.load('t.pkl')
    # 4. Use the model
    y_test_pred = model.predict(X_test)
    print(y_test_pred)
