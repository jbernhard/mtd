# -*- coding: utf-8 -*-

from __future__ import division

import functools

import numpy as np
from scipy import stats

__all__ = ('Prior', 'FlatPrior', 'VariancePrior', 'LengthScalePrior',
           'NoisePrior')


class Prior(list):
    """
    Prior distribution(s).  Subclass of builtins.list; addition and
    multiplication work exactly like a list.

    distributions: iterable of frozen scipy.stats distribution objects

    """
    def __add__(self, other):
        return Prior(super(Prior, self).__add__(other))

    def __mul__(self, other):
        return Prior(super(Prior, self).__mul__(other))

    def rvs(self, size=1):
        """
        Random sample.

        """
        if len(self) == 1:
            return self[0].rvs(size)
        else:
            return np.column_stack([dist.rvs(size) for dist in self])

    def logpdf(self, x):
        """
        Log PDF.

        """
        if len(self) == 1:
            return self[0].logpdf(x)
        else:
            return sum(dist.logpdf(i) for dist, i in zip(self, x))


def FlatPrior(lower=0., upper=1.):
    """
    Constant prior over a finite range.

    """
    return Prior([stats.uniform(loc=lower, scale=upper-lower)])


def VariancePrior(a=5., b=5.):
    """
    Inverse gamma prior for GP variance.

    """
    return Prior([stats.invgamma(a, scale=b)])


def LengthScalePrior(a=1., b=0.1):
    """
    Beta prior for GP length scales (correlation lengths).

    """
    beta = stats.beta(a, b)

    # The scipy beta distribution with b < 1 will generate random samples == 1,
    # but then will also evaluate
    #   beta.pdf(1) == inf,
    # which is probably formally correct but will definitely not work as a
    # prior for MCMC.  It also doesn't make sense for a GP length scale to be
    # exactly zero.  As a workaround, hack the distribution to support
    # (0, 1) exclusive instead of [0, 1] inclusive.

    # small epsilon > 0
    eps = 1e-4

    # change distribution argument limits
    beta.dist.a = eps
    beta.dist.b = 1 - eps

    # set location and scale for beta.rvs() only
    beta.dist.rvs = functools.partial(beta.dist.rvs, loc=eps, scale=(1-2*eps))

    return Prior([beta])


class _log_gen(stats.rv_continuous):
    def _rvs(self, a):
        return np.exp(np.random.uniform(np.log(a), 0, self._size))

    def _pdf(self, x, a):
        return 1/x  # not normalized!

    def _logpdf(self, x, a):
        return -np.log(x)  # not normalized!


def NoisePrior(lower=1e-8):
    """
    Logarithmic (Jeffreys) prior for the noise term (nugget).

    """
    return Prior([_log_gen(a=1e-16, name='log')(lower)])
