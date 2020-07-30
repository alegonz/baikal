import sklearn.svm
from baikal import Step

from baikal import make_step

class SVC(Step, sklearn.svm.SVC):
    def __init__(self, *args, name=None, n_outputs=1, **kwargs):
        super().__init__(*args, name=name, n_outputs=n_outputs, **kwargs)