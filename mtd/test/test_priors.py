# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .. import priors


def test_priors():
    """prior distributions"""

    # This is a scipy.stats distribution so does not need much testing.
    flat = priors.FlatPrior()
    sample = flat.rvs(5)
    assert np.all((sample >= 0) & (sample <= 1)), \
        'FlatPrior:  must produce values in [0, 1].'
    assert np.all(flat.logpdf(sample) == 0), \
        'FlatPrior:  log PDF must be zero in [0, 1].'
    assert np.all(np.isinf(flat.logpdf((-.5, 1.1)))), \
        'FlatPrior:  log PDF must be -inf outside [0, 1].'

    # This is a scipy.stats distribution so does not need much testing.
    var = priors.VariancePrior()
    sample = var.rvs(5)
    assert np.all(sample > 0), \
        'VariancePrior:  must produce values > 0.'
    assert np.all(np.isinf(var.logpdf((-.5, -.01)))), \
        'VariancePrior:  log PDF must be -inf for x < 0.'

    # This is a scipy.stats distribution so does not need much testing.
    length = priors.LengthScalePrior()
    sample = length.rvs(5)
    assert np.all((sample >= 0) & (sample <= 1)), \
        'LengthScalePrior:  must produce values in [0, 1].'
    assert np.all(np.isinf(length.logpdf((-.5, 1.1)))), \
        'LengthScalePrior:  log PDF must be -inf outside [0, 1].'

    # This one is implemented in mtd so it's tested a bit more rigorously.
    noise = priors.NoisePrior()
    sample = noise.rvs(5)
    assert np.all((sample >= 0) & (sample <= 1)), \
        'NoisePrior:  must produce values in [0, 1].'
    assert np.all(noise.logpdf(sample) == -np.log(sample)), \
        'NoisePrior:  log PDF must be -log(x).'
    assert np.all(noise.pdf(sample) == 1./sample), \
        'NoisePrior:  log PDF must be 1/x.'
    assert np.all(np.isinf(noise.logpdf((-.5, 0.)))), \
        'NoisePrior:  log PDF must be -inf for x <= 0.'
