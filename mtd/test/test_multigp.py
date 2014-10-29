# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import george

from .. import MultiGP, kernels, priors


def test_multigp():
    """multivariate gaussian process"""

    # create random data
    nsamples, ndim, nfeatures = 5, 2, 2
    x = np.random.rand(nsamples, ndim)
    y = np.random.rand(nsamples, nfeatures)

    kernel = (
        1.*kernels.ExpSquaredKernel(np.ones(ndim), ndim=ndim) +
        kernels.WhiteKernel(1e-8, ndim=ndim)
    )
    mgp = MultiGP(x, y, kernel, npc=nfeatures)

    assert all(isinstance(g, george.GP) for g in mgp), \
        'MultiGP does not iterate over GPs.'
    assert len(mgp) == nfeatures, \
        'Incorrect number of GPs.\n{} != {}'.format(len(mgp), nfeatures)

    prior = (
        [priors.VariancePrior()] +
        [priors.LengthScalePrior()]*ndim +
        [priors.NoisePrior()]
    )

    mgp.train(prior, 10, 10)
