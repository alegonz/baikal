#! /usr/bin/env python
#
# Copyright (c) 2018 Alejandro González Tineo <alejandrojgt@gmail.com>
# License: New 3-clause BSD

from setuptools import find_packages, setup

PACKAGE_NAME = "baikal"
DESCRIPTION = (
    "A graph-based functional API for building complex scikit-learn pipelines."
)
LONG_DESCRIPTION = """
**baikal is a graph-based, functional API for building complex machine learning 
pipelines of objects that implement the scikit-learn API**. It is mostly inspired 
on the excellent `Keras <https://keras.io>`__ API for Deep Learning, and borrows 
a  few concepts from the `TensorFlow <https://www.tensorflow.org>`__ framework 
and the (perhaps lesser known) `graphkit <https://github.com/yahoo/graphkit>`__
package.

**baikal** aims to provide an API that allows to build complex, non-linear 
machine learning pipelines that looks like this: 

.. image:: https://raw.githubusercontent.com/alegonz/baikal/master/illustrations/multiple_input_nonlinear_pipeline_example_diagram.png

with code that looks like this:

.. code-block:: python

    x1 = Input()
    x2 = Input()
    y_t = Input()
    
    y1 = ExtraTreesClassifier()(x1, y_t)
    y2 = RandomForestClassifier()(x2, y_t)
    z = PowerTransformer()(x2)
    z = PCA()(z)
    y3 = LogisticRegression()(z, y_t)
    
    stacked_features = Stack()([y1, y2, y3])
    y = SVC()(stacked_features, y_t)
    
    model = Model([x1, x2], y, y_t)

**baikal** is compatible with Python >=3.5 and is distributed under the 
BSD 3-clause license.
"""
PROJECT_URL = "https://github.com/alegonz/baikal"
LICENSE = "new BSD"
AUTHOR = "Alejandro González Tineo"
AUTHOR_EMAIL = "alejandrojgt@gmail.com"
PYTHON_REQUIRES = ">=3.5"
INSTALL_REQUIRES = ["numpy"]
EXTRAS_REQUIRE = {
    "dev": ["codecov", "joblib", "mypy", "pytest", "pytest-cov", "scikit-learn"],
    "viz": ["pydot"],
}
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]

# Execute _version.py to get __version__ variable in context
exec(open("baikal/_version.py", encoding="utf-8").read())

setup(
    name=PACKAGE_NAME,
    version=__version__,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    url=PROJECT_URL,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=["tests"]),
)
