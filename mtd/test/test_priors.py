# -*- coding: utf-8 -*-

from __future__ import division

import pickle

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
    assert isinstance(pickle.loads(pickle.dumps(flat)), priors.Prior), \
        'FlatPrior:  not pickleable.'

    # This is a scipy.stats distribution so does not need much testing.
    var = priors.VariancePrior()
    sample = var.rvs(5)
    assert np.all(sample > 0), \
        'VariancePrior:  must produce values > 0.'
    assert np.all(np.isfinite(var.logpdf(sample))), \
        'VariancePrior:  sample log PDF is not finite.'
    assert np.all(np.isinf(var.logpdf((-.5, -.01)))), \
        'VariancePrior:  log PDF must be -inf for x < 0.'

    # This is a scipy.stats distribution so does not need much testing.
    length = priors.LengthScalePrior()
    sample = length.rvs(5)
    assert np.all((sample >= 0) & (sample <= 1)), \
        'LengthScalePrior:  must produce values in [0, 1].'
    assert np.all(np.isfinite(length.logpdf(sample))), \
        'LengthScalePrior:  sample log PDF is not finite.'
    assert np.all(np.isinf(length.logpdf((-.5, 1.1)))), \
        'LengthScalePrior:  log PDF must be -inf outside [0, 1].'

    # This one is implemented in mtd so it's tested a bit more rigorously.
    noise = priors.NoisePrior()
    sample = noise.rvs(5)
    assert np.all((sample >= 0) & (sample <= 1)), \
        'NoisePrior:  must produce values in [0, 1].'
    assert np.all(noise.logpdf(sample) == -np.log(sample)), \
        'NoisePrior:  log PDF must be -log(x).'
    assert np.all(noise[0].pdf(sample) == 1./sample), \
        'NoisePrior:  log PDF must be 1/x.'
    assert np.all(np.isinf(noise.logpdf((-.5, 0.)))), \
        'NoisePrior:  log PDF must be -inf for x <= 0.'

    # Test combining priors.
    prior = var + 2*length + noise
    assert len(prior) == 4, 'Combined prior has incorrect length.'
    assert all((
        prior[0] is var[0],
        prior[1] is length[0],
        prior[2] is length[0],
        prior[3] is noise[0]
    )), 'Combined prior does not reduce to its components.'

    sample = prior.rvs(5)
    assert sample.shape == (5, 4), \
        'Combined prior sample has incorrect shape.'
    lp = tuple(prior.logpdf(s) for s in sample)
    assert len(lp) == 5, 'Sample log PDF does not match sample size.'
    assert np.all(np.isfinite(lp)), 'Sample log PDF is not finite.'
