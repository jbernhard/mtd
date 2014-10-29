# -*- coding: utf-8 -*-

from __future__ import division

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
        return np.squeeze(np.column_stack([dist.rvs(size) for dist in self]))

    def logpdf(self, x):
        """
        Log PDF.

        """
        x = np.asarray(x)

        if len(self) == 1:
            return self[0].logpdf(x)
        else:
            x = np.squeeze(x.T)
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
    return Prior([stats.beta(a, b)])


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
