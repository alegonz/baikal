import numpy as np
import sklearn.linear_model
from sklearn.datasets import fetch_openml
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split

from baikal import Input, Model, Step
from baikal.plot import plot_model
from baikal.steps import ColumnStack


# ------- Define steps
class LogisticRegression(Step, sklearn.linear_model.LogisticRegression):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outputs = 1


# ------- Load a multi-label dataset
# (from https://www.openml.org/d/40597)
X, Y = fetch_openml('yeast', version=4, return_X_y=True)
Y = Y == 'TRUE'
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)


n_targets = Y.shape[1]

# ------- Build model
x = Input()

ys = []
for j in range(n_targets):
    x_stacked = ColumnStack()([x, *ys[:j]])
    yj = LogisticRegression(solver='lbfgs')(x_stacked)
    ys.append(yj)

y = ColumnStack()(ys)

model = Model(x, y)
plot_model(model, filename='classifier_chain.png', dpi=96)  # This might take a few seconds

# ------- Train model
np.random.seed(87)
order = np.arange(n_targets)
np.random.shuffle(order)

# The model output is the output of the Stack step, which is
# not a trainable step, so we pass y=None.
# The preceding steps (each classifier of the chain) take
# their target data via the extra_targets argument.
extra_targets = {ys[j]: Y_train[:, order[j]] for j in range(n_targets)}
model.fit(X_train, y=None, extra_targets=extra_targets)

# ------- Evaluate model
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

print('Jaccard score on train data:', jaccard_score(Y_train[:, order], Y_train_pred, average='samples'))
print('Jaccard score on test data:', jaccard_score(Y_test[:, order], Y_test_pred, average='samples'))
