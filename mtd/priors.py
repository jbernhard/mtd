# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy.stats import distributions

__all__ = ('Prior', 'UniformPrior', 'ExpPrior', 'InvGammaPrior', 'BetaPrior',
           'LogPrior')


class Prior(object):
    """
    Convenience class for handling prior distributions.  Prior objects can be
    added together and multiplied like lists.

    dists: frozen scipy.stats distribution object(s)

    """
    def __init__(self, *dists):
        self._dists = list(dists)
        self._update()

    def __len__(self):
        return len(self._dists)

    def __iter__(self):
        return iter(self._dists)

    def __getitem__(self, key):
        return self._dists[key]

    def __add__(self, other):
        return Prior(*(self._dists + other._dists))

    def __mul__(self, other):
        return Prior(*(self._dists * other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __iadd__(self, other):
        self._dists += other._dists
        self._update()
        return self

    def __imul__(self, other):
        self._dists *= other
        self._update()
        return self

    def _update(self):
        """
        Cache prior ranges.

        """
        self._min, self._max = np.array([d.interval(1.) for d in self]).T

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
            if np.any((x < self._min) | (x > self._max)):
                return -np.inf
            return sum(dist.logpdf(i) for dist, i in zip(self, x))

    def __getstate__(self):
        return [(dist.dist.name, dist.args, dist.kwds) for dist in self]

    def __setstate__(self, state):
        self._dists = [getattr(distributions, name)(*args, **kwds)
                       for name, args, kwds in state]
        self._update()


def UniformPrior(low=0., high=1.):
    """
    Constant prior over a finite range.

    low, high : min, max of range

    """
    return Prior(distributions.uniform(loc=low, scale=(high-low)))


def ExpPrior(rate=1.):
    """
    Exponential prior.

    rate : exponential rate parameter (inverse scale)

    """
    return Prior(distributions.expon(scale=1./rate))


def InvGammaPrior(a=5., b=5.):
    """
    Inverse gamma prior, e.g. for GP variance.

    a : gamma shape parameter
    b : scale parameter

    """
    return Prior(distributions.invgamma(a, scale=b))


class _beta_mod_gen(distributions.beta.__class__):
    """
    A beta distribution modified to work as a Bayesian prior.

    The scipy beta dist with b < 1 will generate random samples == 1, but then
    will also evaluate beta.pdf(1) == inf, which is probably formally correct
    but will definitely not work as a prior.  It also doesn't make sense for a
    GP length scale to be exactly zero.  As a workaround, hack the distribution
    to support (0, 1) exclusive instead of [0, 1] inclusive.

    """
    # small epsilon > 0
    _eps = 1e-4

    def _rvs(self, *args):
        # coerce random samples to (0, 1) exclusive
        rvs = super(distributions.beta.__class__, self)._rvs(*args)
        rvs *= 1 - 2*self._eps
        rvs += self._eps
        return rvs

distributions.beta_mod = _beta_mod_gen(
    a=_beta_mod_gen._eps,
    b=(1.-_beta_mod_gen._eps),
    name='beta_mod'
)


def BetaPrior(a=1., b=0.1):
    """
    Beta prior, e.g. for GP length scales (correlation lengths).

    a, b : beta shape parameters

    """
    return Prior(distributions.beta_mod(a, b))


class _log_gen(distributions.rv_continuous):
    def _rvs(self, a):
        return np.exp(np.random.uniform(np.log(a), 0, self._size))

    def _logpdf(self, x, a):
        return -np.log(x)  # not normalized!

distributions.logarithmic = _log_gen(a=1e-16, name='logarithmic', shapes='a')


def LogPrior(low=1e-8, high=1.):
    """
    Logarithmic (Jeffreys) prior, e.g. for the noise term (nugget).

    low, high : range of random samples
        Does not affect log probability.

    """
    return Prior(distributions.logarithmic(low/high, scale=high))
