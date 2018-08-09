# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
      requirements = f.readlines()

setup(name='VKGE',
      version='0.1.0',
      description='Variational Embeddings of Knowledge Graphs',
      author='Pasquale Minervini',
      author_email='mc_rivers@icloud.com',
      url='https://github.com/acr42/vkge',
      test_suite='tests',
      license='MIT',
      install_requires=requirements,
      setup_requires=['pytest-runner'] + requirements,
      tests_require=requirements,
      packages=find_packages())
