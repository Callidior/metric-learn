#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
from distutils.extension import Extension
import numpy
from Cython.Build import cythonize

version = "0.5.0"
setup(name='metric-learn',
      version=version,
      description='Python implementations of metric learning algorithms',
      author=['CJ Carey', 'Yuan Tang'],
      author_email='ccarey@cs.umass.edu',
      url='http://github.com/all-umass/metric-learn',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      packages=['metric_learn'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'six',
          'cython'
      ],
      extras_require=dict(
          docs=['sphinx', 'shinx_rtd_theme', 'numpydoc'],
          demo=['matplotlib'],
      ),
      test_suite='test',
      ext_modules = cythonize([Extension("metric_learn.itml", ["metric_learn/itml.pyx"], include_dirs = [numpy.get_include()])]),
      keywords=[
          'Metric Learning',
          'Large Margin Nearest Neighbor',
          'Information Theoretic Metric Learning',
          'Sparse Determinant Metric Learning',
          'Least Squares Metric Learning',
          'Neighborhood Components Analysis'
      ])
