import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from baikal import Input, Model, make_step
from baikal.plot import plot_model
from baikal.steps import Stack


# 1. Define the steps
LogisticRegression = make_step(sklearn.linear_model.LogisticRegression)
RandomForestClassifier = make_step(sklearn.ensemble.RandomForestClassifier)
ExtraTreesClassifier = make_step(sklearn.ensemble.ExtraTreesClassifier)
PCA = make_step(sklearn.decomposition.PCA)
SVC = make_step(sklearn.svm.SVC)
PowerTransformer = make_step(sklearn.preprocessing.PowerTransformer)

# 2. Build the model
x1 = Input()
x2 = Input()
y_t = Input()

y1 = ExtraTreesClassifier()(x1, y_t)
y2 = RandomForestClassifier()(x2, y_t)
z = PowerTransformer()(x2)
z = PCA()(z)
y3 = LogisticRegression()(z, y_t)

stacked_features = Stack()([y1, y2, y3])
y_p = SVC()(stacked_features, y_t)

model = Model([x1, x2], y_p, y_t)
plot_model(model, filename="multiple_input_nonlinear_pipeline_example_plot.png")

# 3. Train the model
dataset = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, random_state=0
)

# Let's suppose the dataset is originally split in two
X1_train, X2_train = X_train[:, :15], X_train[:, 15:]
X1_test, X2_test = X_test[:, :15], X_test[:, 15:]

model.fit([X1_train, X2_train], y_train)

# 4. Use the model
y_test_pred = model.predict([X1_test, X2_test])

# This also works:
# y_test_pred = model.predict({x1: X1_test, x2: X2_test})

# We can also query any intermediate outputs:
outs = model.predict(
    [X1_test, X2_test], output_names=["ExtraTreesClassifier_0/0", "PCA_0/0"]
)
