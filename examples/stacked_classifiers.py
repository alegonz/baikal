import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from baikal import Input, Model, make_step
from baikal.plot import plot_model
from baikal.steps import Concatenate


# ------- Define steps
LogisticRegression = make_step(sklearn.linear_model.LogisticRegression)
RandomForestClassifier = make_step(sklearn.ensemble.RandomForestClassifier)
ExtraTreesClassifier = make_step(sklearn.ensemble.ExtraTreesClassifier)

# ------- Load dataset
data = sklearn.datasets.load_breast_cancer()
X, y_p = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y_p, test_size=.2, random_state=0)

# ------- Build model
x = Input()
y_t = Input()
y_p1 = LogisticRegression(function='predict_proba')(x, y_t)
y_p2 = RandomForestClassifier(function='predict_proba')(x, y_t)
ensemble_features = Concatenate()([y_p1, y_p2])
y_p = ExtraTreesClassifier()(ensemble_features, y_t)

model = Model(x, y_p, y_t)
plot_model(model, filename='stacked_classifiers.png', dpi=96)

# ------- Train model
model.fit(X_train, y_train)

# ------- Evaluate model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('F1 score on train data:', f1_score(y_train, y_train_pred))
print('F1 score on test data:', f1_score(y_test, y_test_pred))
