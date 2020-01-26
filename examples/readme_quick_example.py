import sklearn.svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from baikal import Input, Model, make_step
from baikal.plot import plot_model


# 1. Define a step
SVC = make_step(sklearn.svm.SVC)

# 2. Build the model
x = Input()
y_t = Input()
y_p = SVC(C=1.0, kernel="rbf", gamma=0.5)(x, y_t)

model = Model(x, y_p, y_t)
plot_model(model, filename="readme_quick_example.png")

# 3. Train the model
dataset = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, random_state=0
)

model.fit(X_train, y_train)

# 4. Use the model
y_test_pred = model.predict(X_test)
