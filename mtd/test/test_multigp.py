# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_allclose
from nose.tools import assert_raises
import george

from .. import MultiGP, PCA, kernels, priors
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
    cov_ = cov_.diagonal()

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
        priors.InvGammaPrior() +
        priors.BetaPrior()*ndim +
        priors.LogPrior()
    )

    # dimension mismatch
    assert_raises(ValueError, MultiGP, x, y[:3], kernel)

    # instantiate MultiGP
    mgp = MultiGP(x, y, kernel, npc=nfeatures)

    assert len(mgp) == nfeatures, \
        'MultiGP has incorrect length (number of individual GPs).'

    assert isinstance(mgp.pca, PCA), \
        'MultiGP.pca property does not return a PCA object.'

    # test [de]standardization
    xmin = x.min(axis=0)
    assert_equal(mgp._standardize(xmin), np.zeros(ndim),
                 err_msg='x minimum must standardize to zero.')
    assert_equal(mgp._destandardize(np.zeros(ndim)), xmin,
                 err_msg='Zero must destandardize to x minimum.')
    xmax = x.max(axis=0)
    assert_equal(mgp._standardize(xmax), np.ones(ndim),
                 err_msg='x maximum must standardize to one.')
    assert_equal(mgp._destandardize(np.ones(ndim)), xmax,
                 err_msg='One must destandardize to x maximum.')

    # get/set kernel hyperparameters
    for n in range(len(mgp)):
        assert_array_equal(mgp.get_kernel_pars(n), kernel.pars,
                           err_msg='Incorrect retrieved hyperparameters.')

    new_pars = kernel.pars * 1.1
    for n in range(len(mgp)):
        mgp.set_kernel_pars(n, new_pars)
        assert_array_equal(mgp.get_kernel_pars(n), new_pars,
                           err_msg='Incorrect manually set hyperparameters.')

    pc1 = mgp._predict_pc(x)
    pc2 = np.empty((nsamples, nfeatures))
    mgp._predict_pc(x, out=pc2)
    assert_array_equal(
        pc1, pc2,
        err_msg='In-place PC prediction gives inconsistent results.'
    )

    mean = mgp.predict(x)
    assert_allclose(
        y, mean,
        err_msg='MultiGP does not predict noise-free training points exactly.'
    )

    mean2, var = mgp.predict(x, mean_only=False)
    assert_array_equal(mean2, mean, err_msg='Means are inconsistent.')
    assert_allclose(
        var, 0, atol=1e-15,
        err_msg='MultiGP variance must vanish at noise-free training points.'
    )
    assert mean.shape == var.shape == (nsamples, nfeatures), \
        'Mean and variance arrays must have same shape.'

    mean, var = mgp.predict(np.random.rand(3, ndim), mean_only=False)
    assert np.all(var > 1e-10), \
        'Variance must be nonzero away from training points.\n{}'.format(var)

    # can't get results before training
    assert_raises(RuntimeError, lambda: mgp.training_results)

    # test training
    nstarts = 3
    mgp.train(prior, nstarts, verbose=True)
    res = mgp.training_results
    assert len(res) == len(mgp), \
        'Incorrect number of results lists.'

    for r in res:
        assert len(r) == nstarts, \
            'Incorrect number of training results.'

    yexp = np.random.rand(nfeatures)
    yerr = .1
    prior = priors.UniformPrior() * ndim

    # can't get the sampler before calibration
    assert_raises(RuntimeError, lambda: mgp.cal_sampler)

    # test calibration
    nwalkers, nsteps = 8, 10
    mgp.calibrate(yexp, yerr, nwalkers, nsteps, verbose=True)
    mgp.calibrate(yexp, yerr, nwalkers, nsteps, prior=prior, verbose=True)

    assert mgp.cal_sampler is mgp._cal_sampler

    flatchain = mgp.cal_flatchain
    assert_equal(flatchain.shape, (nwalkers*nsteps, ndim),
                 err_msg='Calibration flatchain has incorrect shape.')

    chain = mgp.cal_chain
    assert_equal(chain.shape, (nwalkers, nsteps, ndim),
                 err_msg='Calibration chain has incorrect shape.')

    samples = mgp.cal_samples
    assert_equal(samples.shape, (nwalkers*nsteps, ndim),
                 err_msg='Calibration samples have incorrect shape.')

    # wrong number of dimensions
    assert_raises(ValueError, mgp.predict, [[.5, .5, .5]])

    # small sample size
    nsamples, ndim, nfeatures = 5, 2, 7
    x = np.random.rand(nsamples, ndim)
    y = np.random.rand(nsamples, nfeatures)

    mgp = MultiGP(x, y, kernel)

    assert len(mgp) == nsamples, \
        'MultiGP has incorrect length.'

    assert mgp.predict([[.5, .5]]).shape == (1, nfeatures), \
        'Incorrect number of predicted features.'
