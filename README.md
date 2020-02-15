![baikal](illustrations/baikal1_blue.png)

# A graph-based functional API for building complex scikit-learn pipelines

[![build status](https://circleci.com/gh/alegonz/baikal/tree/master.svg?style=svg&circle-token=fb67eeed2067c361989d2091b9d4d03e6899010b)](https://circleci.com/gh/alegonz/baikal/tree/master)
[![coverage](https://codecov.io/gh/alegonz/baikal/branch/master/graph/badge.svg?token=SSoeQETNh6)](https://codecov.io/gh/alegonz/baikal)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/alegonz/baikal.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/alegonz/baikal/context:python)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![latest release](https://img.shields.io/pypi/v/baikal.svg)](https://pypi.org/project/baikal)
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
