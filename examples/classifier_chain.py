import numpy as np
import random

import sklearn.linear_model
from sklearn.datasets import fetch_openml
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split

from baikal import Input, Model, make_step
from baikal.plot import plot_model
from baikal.steps import ColumnStack, Split, Lambda

# ------- Define steps
LogisticRegression = make_step(sklearn.linear_model.LogisticRegression)

# ------- Load a multi-label dataset
# (from https://www.openml.org/d/40597)
X, Y = fetch_openml("yeast", version=4, return_X_y=True)
Y = Y == "TRUE"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

n_targets = Y.shape[1]
random.seed(87)
order = list(range(n_targets))
random.shuffle(order)

# ------- Build model
x = Input()
y_t = Input()

squeeze = Lambda(np.squeeze, axis=1)

ys_t = Split(n_targets, axis=1)(y_t)
ys_p = []
for j, k in enumerate(order):
    x_stacked = ColumnStack()(inputs=[x, *ys_p[:j]])
    ys_t[k] = squeeze(ys_t[k])
    ys_p.append(LogisticRegression(solver="lbfgs")(x_stacked, ys_t[k]))

ys_p = [ys_p[order.index(j)] for j in range(n_targets)]
y_p = ColumnStack()(ys_p)

model = Model(inputs=x, outputs=y_p, targets=y_t)
# This might take a few seconds
plot_model(model, filename="classifier_chain.png", dpi=96)

# ------- Train model
model.fit(X_train, Y_train)

# ------- Evaluate model
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

print(
    "Jaccard score on train data:",
    jaccard_score(Y_train, Y_train_pred, average="samples"),
)
print(
    "Jaccard score on test data:",
    jaccard_score(Y_test, Y_test_pred, average="samples"),
)
