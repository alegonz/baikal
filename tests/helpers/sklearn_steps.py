import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing

from baikal import make_step

LogisticRegression = make_step(sklearn.linear_model.LogisticRegression)
RandomForestClassifier = make_step(sklearn.ensemble.RandomForestClassifier)
ExtraTreesClassifier = make_step(sklearn.ensemble.ExtraTreesClassifier)
PCA = make_step(sklearn.decomposition.PCA)
StandardScaler = make_step(sklearn.preprocessing.StandardScaler)
