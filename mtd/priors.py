# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import stats

__all__ = ('Prior', 'FlatPrior', 'VariancePrior', 'LengthScalePrior',
           'NoisePrior')


class Prior(object):
    """
    Convenience class for handling prior distributions.  Prior objects can be
    added together and multiplied like lists.

    dists: frozen scipy.stats distribution object(s)

    """
    def __init__(self, *dists):
        self._dists = list(dists)

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

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __iadd__(self, other):
        self._dists += other._dists
        return self

    def __imul__(self, other):
        self._dists *= other
        return self

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

    def __getstate__(self):
        return [(dist.dist.name, dist.args, dist.kwds) for dist in self]

    def __setstate__(self, state):
        self._dists = [self._recreate_dist(*s) for s in state]

    @staticmethod
    def _recreate_dist(name, args, kwds):
        try:
            dist = globals()[name]
        except KeyError:
            dist = getattr(stats, name)
        return dist(*args, **kwds)


def FlatPrior(lower=0., upper=1.):
    """
    Constant prior over a finite range.

    """
    return Prior(stats.uniform(loc=lower, scale=upper-lower))


def VariancePrior(a=5., b=5.):
    """
    Inverse gamma prior for GP variance.

    """
    return Prior(stats.invgamma(a, scale=b))


class _beta_mod_gen(stats.beta.__class__):
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
        rvs = super(stats.beta.__class__, self)._rvs(*args)
        rvs *= 1 - 2*self._eps
        rvs += self._eps
        return rvs

_beta_mod = _beta_mod_gen(a=_beta_mod_gen._eps, b=(1.-_beta_mod_gen._eps),
                          name='_beta_mod')


def LengthScalePrior(a=1., b=0.1):
    """
    Beta prior for GP length scales (correlation lengths).

    """
    return Prior(_beta_mod(a, b))


class _log_gen(stats.rv_continuous):
    def _rvs(self, a):
        return np.exp(np.random.uniform(np.log(a), 0, self._size))

    def _pdf(self, x, a):
        return 1/x  # not normalized!

    def _logpdf(self, x, a):
        return -np.log(x)  # not normalized!

_log = _log_gen(a=1e-16, name='_log')


def NoisePrior(lower=1e-8):
    """
    Logarithmic (Jeffreys) prior for the noise term (nugget).

    """
    return Prior(_log(lower))
