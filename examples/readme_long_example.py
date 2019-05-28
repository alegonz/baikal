import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from baikal import Input, Step, Model
from baikal.plot import plot_model
from baikal.steps import Stack


# 1. Define the steps
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


class PCA(Step, sklearn.decomposition.PCA):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outputs = 1


class SVC(Step, sklearn.svm.SVC):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outputs = 1


class PowerTransformer(Step, sklearn.preprocessing.PowerTransformer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outputs = 1


# 2. Build the model
x1 = Input()
x2 = Input()

y1 = ExtraTreesClassifier()(x1)
y2 = RandomForestClassifier()(x2)
z = PowerTransformer()(x2)
z = PCA()(z)
y3 = LogisticRegression()(z)

ensemble_features = Stack()([y1, y2, y3])
y = SVC()(ensemble_features)

model = Model([x1, x2], y)
plot_model(model, filename='multiple_input_nonlinear_pipeline_example_plot.png')

# 3. Train the model
dataset = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=0)

# Let's suppose the dataset is originally split in two
X1_train, X2_train = X_train[:, :15], X_train[:, 15:]
X1_test, X2_test = X_test[:, :15], X_test[:, 15:]

model.fit(X=[X1_train, X2_train], y=y_train,
          extra_targets={y1: y_train, y2: y_train, y3: y_train})

# 4. Use the model
y_test_pred = model.predict([X1_test, X2_test])

# This also works:
# y_test_pred = model.predict({x1: X1_test, x2: X2_test})

# We can also query any intermediate outputs:
outs = model.predict([X1_test, X2_test], outputs=['ExtraTreesClassifier_0/0', 'PCA_0/0'])
