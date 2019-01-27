#! /usr/bin/env python
#
# Copyright (c) 2018 Alejandro González Tineo <alejandrojgt@gmail.com>
# License: New 3-clause BSD

from setuptools import setup

version = '0.1.0'

setup(name='baikal',
      version=version,
      description='Graph-based functional API for building machine learning pipelines',
      url='https://gitlab.com/alegonz/baikal',
      license='new BSD',
      author='Alejandro González Tineo',
      author_email='alejandrojgt@gmail.com',
      python_requires='>=3.5',
      extras_require={
          'dev': [
              'pytest',
              'sklearn'
          ]
      },
      packages=['baikal'])
