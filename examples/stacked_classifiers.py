import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from baikal import Input, Model, Step
from baikal.plot import plot_model
from baikal.steps import Concatenate


# ------- Define steps
class LogisticRegression(Step, sklearn.linear_model.LogisticRegression):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outputs = 1


class RandomForestClassifier(Step, sklearn.ensemble.RandomForestClassifier):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outputs = 1


class ExtraTreesClassifier(Step, sklearn.ensemble.ExtraTreesClassifier):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outputs = 1


# ------- Load dataset
data = sklearn.datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# ------- Build model
x = Input()
y1 = LogisticRegression(function='predict_proba')(x)
y2 = RandomForestClassifier(function='predict_proba')(x)
ensemble_features = Concatenate()([y1, y2])
y = ExtraTreesClassifier()(ensemble_features)

model = Model(x, y)
plot_model(model, filename='stacked_classifiers.png', dpi=96)

# ------- Train model
# The model output is the output of the ExtraTreesClassifier
# step, and it requires target data to fit, so we pass y=y_train.
# The preceding steps (each classifier in the stack) take
# their target data via the extra_targets argument.
model.fit(X_train, y=y_train, extra_targets={y1: y_train, y2: y_train})
# This is also valid, however:
# model.fit(X_train, {y: y_train, y1: y_train, y2: y_train})

# ------- Evaluate model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('F1 score on train data:', f1_score(y_train, y_train_pred))
print('F1 score on test data:', f1_score(y_test, y_test_pred))
