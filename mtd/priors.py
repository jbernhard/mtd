# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import stats

__all__ = 'FlatPrior', 'VariancePrior', 'LengthScalePrior', 'LogPrior'


def FlatPrior(lower=0., upper=1.):
    """
    Constant prior over a finite range.

    """
    return stats.uniform(loc=lower, scale=upper-lower)


def VariancePrior(a=5., b=5.):
    """
    Inverse gamma prior for GP variance.

    """
    return stats.invgamma(a, scale=b)


def LengthScalePrior(a=1., b=0.1):
    """
    Beta prior for GP length scales (correlation lengths).

    """
    return stats.beta(a, b)


class _log_gen(stats.rv_continuous):
    def _rvs(self, a):
        return np.exp(np.random.uniform(np.log(a), 0, self._size))

    def _pdf(self, x, a):
        return 1/x  # not normalized!

    def _logpdf(self, x, a):
        return -np.log(x)  # not normalized!


def LogPrior(lower=1e-8):
    """
    Logarithmic (Jeffreys) prior.

    """
    return _log_gen(a=1e-16, name='log')(lower)
