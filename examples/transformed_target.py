# Adapted from the scikit-learn example in:
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py

import numpy as np
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from baikal import make_step, Input, Model
from baikal.steps import Lambda


# ------- Define steps
RidgeCV = make_step(sklearn.linear_model.RidgeCV)
QuantileTransformer = make_step(sklearn.preprocessing.QuantileTransformer)

# ------- Load dataset
dataset = load_boston()
target = np.array(dataset.feature_names) == "DIS"
X = dataset.data[:, np.logical_not(target)]
y = dataset.data[:, target].squeeze()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# ------- Build model
transformer = QuantileTransformer(n_quantiles=300, output_distribution="normal")

x = Input()
y_t = Input()
# QuantileTransformer requires an explicit feature dimension, hence the Lambda step
y_t_trans = Lambda(np.reshape, newshape=(-1, 1))(y_t)
y_t_trans = transformer(y_t_trans)
y_p_trans = RidgeCV()(x, y_t_trans)
y_p = transformer(y_p_trans, compute_func="inverse_transform", trainable=False)
model = Model(x, y_p, y_t)

# ------- Train model
model.fit(X_train, y_train)

# ------- Evaluate model
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = median_absolute_error(y_test, y_pred)
print("R^2={}\nMAE={}".format(r2, mae))
