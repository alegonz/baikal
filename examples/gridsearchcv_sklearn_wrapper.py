import sklearn.decomposition
import sklearn.ensemble
import sklearn.decomposition
import sklearn.linear_model
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from baikal import Input, Model, make_step
from baikal.sklearn import SKLearnWrapper


LogisticRegression = make_step(sklearn.linear_model.LogisticRegression)
RandomForestClassifier = make_step(sklearn.ensemble.RandomForestClassifier)
PCA = make_step(sklearn.decomposition.PCA)


def build_fn():
    x = Input()
    y_t = Input()
    h = PCA(random_state=random_state, name='pca')(x)
    y_p = LogisticRegression(random_state=random_state, name='classifier')(h, y_t)
    model = Model(x, y_p, y_t)
    return model


iris = datasets.load_iris()
x_data = iris.data
y_data = iris.target
random_state = 123
verbose = 0

# cv will default to KFold if the estimator is a baikal Model
# so we have to pass StratifiedKFold directly
cv = StratifiedKFold(n_splits=3, random_state=random_state)

param_grid = [
    {'classifier': [LogisticRegression(random_state=random_state, solver='lbfgs', multi_class='multinomial')],
     'classifier__C': [0.01, 0.1, 1],
     'pca__n_components': [1, 2, 3, 4]},
    {'classifier': [RandomForestClassifier(random_state=random_state)],
     'classifier__n_estimators': [10, 50, 100],
     'pca__n_components': [1, 2, 3, 4]}
]

sk_model = SKLearnWrapper(build_fn)
gscv_baikal = GridSearchCV(sk_model, param_grid, cv=cv, scoring='accuracy',
                           return_train_score=True, verbose=verbose)
gscv_baikal.fit(x_data, y_data)
print('Best score:', gscv_baikal.best_score_)
print('Best parameters', gscv_baikal.best_params_)
# The model with the best parameters can be accessed via:
# gscv_baikal.best_estimator_.model
