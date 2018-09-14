
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='hyperspherical_vae',
    version='0.1.1',
    author='Nicola De Cao, Tim R. Davidson, Luca Falorsi',
    author_email='nicola.decao@gmail.com',
    description='Tensorflow implementation of Hyperspherical Variational Auto-Encoders',
    license='MIT',
    keywords='tensorflow vae variational-auto-encoder von-mises-fisher  machine-learning deep-learning manifold-learning',
    url='https://nicola-decao.github.io/s-vae-tf/',
    download_url='https://github.com/nicola-decao/SVAE',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    install_requires=['numpy', 'tensorflow>=1.7.0', 'scipy'],
    packages=find_packages()
)

# -*- coding: utf-8 -*-

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
