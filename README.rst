===
mtd
===
Helper classes for Bayesian model-to-data comparison.

Based on
`Computer Model Calibration Using High-Dimensional Output <http://www.jstor.org/stable/27640080>`_ (2008)
and
`A Bayesian Approach for Parameter Estimation and Prediction using a Computationally Intensive Model <http://inspirehep.net/record/1305921>`_ (2014)
by Dave Higdon *et al*.

WARNING
-------
I made this for my personal use, in particular for my papers http://inspirehep.net/record/1342465 and http://inspirehep.net/record/1458287.
Some example scripts using this library: https://github.com/jbernhard/mtd-paper/blob/master/mcmc/train-and-calibrate and https://github.com/jbernhard/qm2015/blob/master/calibration/calibrate.

While ``mtd`` certainly served its purpose, it lacks features and flexibility, and I don't plan to develop it further.

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
