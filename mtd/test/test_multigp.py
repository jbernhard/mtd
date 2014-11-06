# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_allclose
from nose.tools import assert_raises
import george

from .. import MultiGP, kernels, priors
from ..multigp import _GPProcess


def test_gp_process():
    """GP multiprocessing"""

    x = np.array((-1, 0, 1), dtype=float)
    y = np.array((-.5, 0, 1), dtype=float)
    kernel = kernels.ExpSquaredKernel(1.)

    proc = _GPProcess(x, y, kernel)
    proc.start()

    gp = george.GP(kernel)
    gp.compute(x)

    mu = proc.send_cmd('predict', x, mean_only=True).get_result()
    mu_ = gp.predict(y, x, mean_only=True)

    err_msg = 'Inconsistent predictions from multiprocessing and normal GPs.'
    assert_array_equal(mu, mu_, err_msg=err_msg)

    t = -.75, .5
    proc.send_cmd('predict', t)
    mu, cov = proc.get_result()
    mu_, cov_ = gp.predict(y, t)

    assert_array_equal(mu, mu_, err_msg=err_msg)
    assert_array_equal(cov, cov_, err_msg=err_msg)

    proc.send_cmd('hello')
    assert_raises(ValueError, proc.get_result)

    proc.stop()


def test_multigp():
    """multivariate gaussian process"""

    # create random data
    nsamples, ndim, nfeatures = 5, 2, 2
    x = np.random.rand(nsamples, ndim)
    y = np.random.rand(nsamples, nfeatures)

    # create kernel with standard hyperparameters and conjugate prior
    kernel = (
        1.*kernels.ExpSquaredKernel(np.ones(ndim), ndim=ndim) +
        kernels.WhiteKernel(1e-8, ndim=ndim)
    )
    prior = (
        priors.VariancePrior() +
        priors.LengthScalePrior()*ndim +
        priors.NoisePrior()
    )

    # instantiate MultiGP
    mgp = MultiGP(x, y, kernel, npc=nfeatures)

    # test [de]standardization
    xmin = x.min(axis=0)
    assert_equal(mgp._standardize(xmin), np.zeros(ndim),
                 err_msg='x minimum must standardize to zero.')
    assert_equal(mgp._destandardize(np.zeros(ndim, dtype=int)), xmin,
                 err_msg='Zero must destandardize to x minimum.')
    xmax = x.max(axis=0)
    assert_equal(mgp._standardize(xmax), np.ones(ndim, dtype=int),
                 err_msg='x maximum must standardize to one.')
    assert_equal(mgp._destandardize(np.ones(ndim)), xmax,
                 err_msg='One must destandardize to x maximum.')

    assert_allclose(
        y, mgp.predict(x),
        err_msg='MultiGP does not predict noise-free training points exactly.'
    )

    # can't get the chain before training
    assert_raises(RuntimeError, mgp.get_training_sampler_attr, 0, 'chain')

    # test training by verifying the MCMC chain has the expected shape
    nwalkers, nsteps = 8, 5
    mgp.train(prior, nwalkers, nsteps)

    for i in range(nfeatures):
        chain = mgp.get_training_sampler_attr(i, 'chain')
        assert_equal(
            chain.shape,
            (nwalkers, nsteps, len(prior)),
            err_msg='Training chain {} has incorrect shape.'.format(i)
        )

    # bad attribute
    assert_raises(AttributeError, mgp.get_training_sampler_attr, 0, 'hello')

    yexp = np.random.rand(nfeatures)
    yerr = .1
    prior = priors.FlatPrior() * ndim

    mgp.calibrate(yexp, yerr, prior, nwalkers, nsteps)

    flatchain = mgp.get_calibration_chain(flat=True)
    assert_equal(flatchain.shape, (nwalkers*nsteps, ndim),
        err_msg='Calibration flatchain has incorrect shape.')

    chain = mgp.get_calibration_chain(flat=False)
    assert_equal(chain.shape, (nwalkers, nsteps, ndim),
        err_msg='Calibration chain has incorrect shape.')
