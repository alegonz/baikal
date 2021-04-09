![baikal](illustrations/baikal1_blue.png)

# A graph-based functional API for building complex scikit-learn pipelines

[![docs](https://img.shields.io/badge/docs-read%20now-blue.svg)](https://baikal.readthedocs.io)
[![build status](https://circleci.com/gh/alegonz/baikal/tree/master.svg?style=svg&circle-token=fb67eeed2067c361989d2091b9d4d03e6899010b)](https://circleci.com/gh/alegonz/baikal/tree/master)
[![coverage](https://codecov.io/gh/alegonz/baikal/branch/master/graph/badge.svg?token=SSoeQETNh6)](https://codecov.io/gh/alegonz/baikal)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/alegonz/baikal.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/alegonz/baikal/context:python)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![latest release](https://img.shields.io/pypi/v/baikal.svg)](https://pypi.org/project/baikal)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/baikal/badges/version.svg)](https://anaconda.org/conda-forge/baikal)
[![license](https://img.shields.io/pypi/l/baikal.svg)](https://github.com/alegonz/baikal/blob/master/LICENSE)

**baikal** is written in pure Python. It supports Python 3.5 and above.

Note: **baikal** is still a young project and there might be backward incompatible changes. 
The next development steps and backwards-incompatible changes are announced and discussed 
in [this issue](https://github.com/alegonz/baikal/issues/16). Please subscribe to it if 
you use **baikal**.

### What is baikal?

**baikal is a graph-based, functional API for building complex machine learning pipelines 
of objects that implement the** [scikit-learn API](https://scikit-learn.org/stable/developers/contributing.html#different-objects). 
It is mostly inspired on the excellent [Keras](https://keras.io) API for Deep Learning, 
and borrows a few concepts from the [TensorFlow](https://www.tensorflow.org) framework 
and the (perhaps lesser known) [graphkit](https://github.com/yahoo/graphkit) package.

**baikal** aims to provide an API that allows to build complex, non-linear machine learning 
pipelines that look like this: 

![multiple_input_nonlinear_pipeline_example_diagram](illustrations/multiple_input_nonlinear_pipeline_example_diagram.png "An example of a multiple-input, nonlinear pipeline")


with code that looks like this:

```python
x1 = Input()
x2 = Input()
y_t = Input()

y1 = ExtraTreesClassifier()(x1, y_t)
y2 = RandomForestClassifier()(x2, y_t)
z = PowerTransformer()(x2)
z = PCA()(z)
y3 = LogisticRegression()(z, y_t)

ensemble_features = Stack()([y1, y2, y3])
y = SVC()(ensemble_features, y_t)

model = Model([x1, x2], y, y_t)
```

### What can I do with it?

With **baikal** you can

- build non-linear pipelines effortlessly
- handle multiple inputs and outputs
- add steps that operate on targets as part of the pipeline
- nest pipelines
- use prediction probabilities (or any other kind of output) as inputs to other steps in the pipeline
- query intermediate outputs, easing debugging
- freeze steps that do not require fitting
- define and add custom steps easily
- plot pipelines

All with boilerplate-free, readable code.

### Why baikal?	

The pipeline above (to the best of the author's knowledge) cannot be easily built using 
[scikit-learn's composite estimators API](https://scikit-learn.org/stable/modules/compose.html#pipelines-and-composite-estimators) 
as you encounter some limitations:	

1. It is aimed at linear pipelines	
    - You could add some step parallelism with the [ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data) 
      API, but this is limited to transformer objects.	
2. Classifiers/Regressors can only be used at the end of the pipeline.	
    - This means we cannot use the predicted labels (or their probabilities) as features 
      to other classifiers/regressors.	
    - You could leverage mlxtend's [StackingClassifier](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/#stackingclassifier) 
      and come up with some clever combination of the above composite estimators 
      (`Pipeline`s, `ColumnTransformer`s, and `StackingClassifier`s, etc), but you might 
      end up with code that feels hard-to-follow and verbose.	
3. Cannot handle multiple input/multiple output models.	

Perhaps you could instead define a big, composite estimator class that integrates each of 
the pipeline steps through composition. This, however, most likely will require 	
* writing big `__init__` methods to control each of the internal steps' knobs;	
* being careful with `get_params` and `set_params` if you want to use, say, `GridSearchCV`;	
* and adding some boilerplate code if you want to access the outputs of intermediate 
  steps for debugging.	

By using **baikal** as shown in the example above, code can be more readable, less verbose 
and closer to our mental representation of the pipeline. **baikal** also provides an API 
to fit, predict with, and query the entire pipeline with single commands. 
