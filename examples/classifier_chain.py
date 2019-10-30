import numpy as np
import sklearn.linear_model
from sklearn.datasets import fetch_openml
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split

from baikal import Input, Model, make_step
from baikal.plot import plot_model
from baikal.steps import ColumnStack


# ------- Define steps
LogisticRegression = make_step(sklearn.linear_model.LogisticRegression)

# ------- Load a multi-label dataset
# (from https://www.openml.org/d/40597)
X, Y = fetch_openml("yeast", version=4, return_X_y=True)
Y = Y == "TRUE"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


n_targets = Y.shape[1]

# ------- Build model
x = Input()
ys_t, ys_p = [], []
for j in range(n_targets):
    x_stacked = ColumnStack()(inputs=[x, *ys_p[:j]])
    yj_t = Input()
    yj_p = LogisticRegression(solver="lbfgs")(inputs=x_stacked, targets=yj_t)
    ys_t.append(yj_t)
    ys_p.append(yj_p)

y_pred = ColumnStack()(ys_p)

model = Model(inputs=x, outputs=y_pred, targets=ys_t)
plot_model(
    model, filename="classifier_chain.png", dpi=96
)  # This might take a few seconds

# ------- Train model
np.random.seed(87)
order = np.arange(n_targets)
np.random.shuffle(order)

ys_train = {ys_t[j]: Y_train[:, order[j]] for j in range(n_targets)}
model.fit(X_train, ys_train)

# ------- Evaluate model
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

print(
    "Jaccard score on train data:",
    jaccard_score(Y_train[:, order], Y_train_pred, average="samples"),
)
print(
    "Jaccard score on test data:",
    jaccard_score(Y_test[:, order], Y_test_pred, average="samples"),
)
