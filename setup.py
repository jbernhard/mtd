#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

import mtd

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='mtd',
    version=mtd.__version__,
    description='Helper classes for Bayesian model-to-data comparison.',
    long_description=long_description,
    author='Jonah Bernhard',
    author_email='jonah.bernhard@gmail.com',
    url='https://github.com/jbernhard/mtd',
    license='MIT',
    packages=['mtd', 'mtd.test'],
    install_requires=['scipy', 'emcee', 'george'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
