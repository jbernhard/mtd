===
mtd
===

Helper classes for Bayesian model-to-data comparison.

.. image:: http://img.shields.io/travis/jbernhard/mtd.svg?style=flat-square
  :target: https://travis-ci.org/jbernhard/mtd

.. image:: http://img.shields.io/coveralls/jbernhard/mtd.svg?style=flat-square
  :target: https://coveralls.io/r/jbernhard/mtd

Based on
`Computer Model Calibration Using High-Dimensional Output <http://www.jstor.org/stable/27640080>`_ (2008)
and
`A Bayesian Approach for Parameter Estimation and Prediction using a Computationally Intensive Model <http://inspirehep.net/record/1305921>`_ (2014)
by Dave Higdon *et al*.

Features
--------
- Emulation of multivariate models via Gaussian process regression and principal component analysis.
- Automated emulator training and calibration to experimental data.
- Intuitive system for specifying priors.
- Fully multithreaded.
- Built on Numpy+Scipy and Dan Foreman-Mackey's excellent MCMC toolkit
  `emcee <https://github.com/dfm/emcee>`_
  and Gaussian process library 
  `george <https://github.com/dfm/george>`_.

To do
-----
- Documentation...
- More rigorous error handling.
- Full Bayesian calibration (currently the kernel hyperparameters are fixed at the
  maximum a posteriori point for the calibration phase).
- Discrepancy model.
- Performance optimizations.
