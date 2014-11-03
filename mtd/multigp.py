# -*- coding: utf-8 -*-

from __future__ import division

import copy
import multiprocessing
import itertools

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

    """
    def __init__(self, x, y, kernel, npc=None):
        x = np.atleast_2d(x)
        self._x_min = x.min(axis=0)
        self._x_range = x.ptp(axis=0)

        self._pca = PCA(y, npc=npc, normalize=True)
        self._y_pc = self._pca.transform()

        template = GP(kernel)
        template.compute(self._standardize(x))
        self._GPs = tuple(copy.deepcopy(template)
                          for _ in range(self._pca.npc))

    def _standardize(self, x):
        """Scale x to the unit hypercube [0, 1]^ndim."""
        return (x - self._x_min) / self._x_range

    def __iter__(self):
        """
        Iterable of the individual GPs.

        """
        return iter(self._GPs)

    def __len__(self):
        """
        Number of individual GPs (i.e. the number of PC).

        """
        return len(self._GPs)

    def train(self, prior, nwalkers, nsteps, nproc=None):
        """
        Train the GPs, i.e. estimate the optimal hyperparameters via MCMC.

        prior: Prior object
            Priors for the kernel hyperparameters.
        nwalkers: number of MCMC walkers
        nsteps: number of MCMC steps per walker
            Both the burn-in and production chains will have nsteps.
        nproc: number of GPs to train in parallel
            Default is to use all available CPUs.

        """
        jobs = zip(
            self._GPs,
            self._y_pc.T,
            itertools.repeat(prior),
            itertools.repeat(nwalkers),
            itertools.repeat(nsteps)
        )

        # pool = multiprocessing.Pool(processes=nproc)
        # samplers = pool.map(_train_gp, jobs)
