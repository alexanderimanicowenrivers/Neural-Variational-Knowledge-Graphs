# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
      requirements = f.readlines()

setup(name='vkge',
      version='0.1.0',
      description='Variational Embeddings of Knowledge Graphs',
      author='Pasquale Minervini',
      author_email='p.minervini@cs.ucl.ac.uk',
      url='https://github.com/pminervini/vkge',
      test_suite='tests',
      license='MIT',
      install_requires=requirements,
      setup_requires=['pytest-runner'] + requirements,
      tests_require=requirements,
      packages=find_packages())
