# -*- coding: utf-8 -*-

from __future__ import division

import copy
import multiprocessing

import numpy as np
import emcee
from george import GP

from .pca import PCA

__all__ = 'MultiGP',


def _train_gp(args):
    gp, y, prior, nwalkers = args

    def log_posterior(pars):
        log_prior = prior.logpdf(pars)
        if not np.isfinite(log_prior):
            return -np.inf
        gp.kernel.pars = pars
        return log_prior + gp.lnlikelihood(y, quiet=True)

    sampler = emcee.EnsembleSampler(nwalkers, len(prior), log_posterior)
    return sampler


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
        self._y_pc = self._pca.transform(y)

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
        # pool = multiprocessing.Pool(processes=nproc)
        # samplers = pool.map(_train_gp, zip(self._GPs, self._y_pc.T))
        pass
