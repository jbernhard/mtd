# -*- coding: utf-8 -*-

from __future__ import division

import multiprocessing

import numpy as np
import emcee
from george import GP

from .pca import PCA

__all__ = 'MultiGP',


def _train_gp(gp, y, prior, nwalkers, nsteps, **kwargs):
    def log_posterior(pars):
        log_prior = prior.logpdf(pars)
        if not np.isfinite(log_prior):
            return -np.inf
        gp.kernel.pars = pars
        return log_prior + gp.lnlikelihood(y, quiet=True)

    sampler = emcee.EnsembleSampler(nwalkers, len(prior), log_posterior)

    # set kernel.pars to MAP point of chain

    return sampler


class _GPProcess(multiprocessing.Process):
    """
    Run a GP in a separate process.

    """
    def __init__(self, x, y, kernel):
        self._x = x
        self._y = y
        self._kernel = kernel
        self._in_pipe, self._out_pipe = multiprocessing.Pipe()
        super(_GPProcess, self).__init__()

    def run(self):
        gp = GP(self._kernel)
        gp.compute(self._x)

        for cmd, args, kwargs in iter(self._out_pipe.recv, None):
            if cmd == 'predict':
                result = gp.predict(self._y, *args, **kwargs)
            elif cmd == 'train':
                result = _train_gp(gp, self._y, *args, **kwargs)
            else:
                result = ValueError('Unknown command: {}.'.format(cmd))

            self._out_pipe.send(result)

    def send_cmd(self, cmd, *args, **kwargs):
        self._in_pipe.send((cmd, args, kwargs))
        return self

    def stop(self):
        self._in_pipe.send(None)

    def get_result(self):
        res = self._in_pipe.recv()
        if isinstance(res, Exception):
            raise res
        return res


class MultiGP(object):
    """
    Multidimensional Gaussian process using principal component analysis.

    x: (nsamples, ndim)
        Design matrix.
    y: (nsamples, nfeatures)
        Model outputs.
    kernel: george kernel
        Template kernel for each GP.
    npc: number of principal components to use
        Passed directly to the PCA constructor.  A separate process will be
        created for each PC.

    """
    def __init__(self, x, y, kernel, npc=None):
        x = np.atleast_2d(x)
        self._x_min = x.min(axis=0)
        self._x_range = x.ptp(axis=0)
        x = self._standardize(x)

        self._pca = PCA(y, npc=npc, normalize=True)

        self._procs = tuple(
            _GPProcess(x, y, kernel)
            for y in self._pca.transform().T
        )
        for p in self._procs:
            p.start()

    def __del__(self):
        for p in self._procs:
            p.stop()

    def _standardize(self, x):
        """Scale x to the unit hypercube [0, 1]^ndim."""
        return (x - self._x_min) / self._x_range

    def train(self, prior, nwalkers, nsteps):
        """
        Train the GPs, i.e. estimate the optimal hyperparameters via MCMC.

        prior: Prior object
            Priors for the kernel hyperparameters.
        nwalkers: number of MCMC walkers
        nsteps: number of MCMC steps per walker
            Both the burn-in and production chains will have nsteps.

        """
        pass
