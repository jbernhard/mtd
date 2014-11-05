# -*- coding: utf-8 -*-

from __future__ import division

import pickle

import numpy as np

from .. import priors


def test_priors():
    """prior distributions"""

    for name in priors.__all__:
        if name == 'Prior':
            continue

        p = getattr(priors, name)()
        sample = p.rvs(10)
        a, b = {
            'FlatPrior': (0, 1),
            'VariancePrior': (0, np.inf),
            'LengthScalePrior': (0, 1),
            'NoisePrior': (0, np.inf),
        }[name]

        assert np.all((sample >= a) & (sample <= b)), \
            '{}: must produce values in [{}, {}].'.format(name, a, b)
        assert np.all(np.isfinite(p.logpdf(sample))), \
            '{}: sample log PDF is not finite.'.format(name)
        assert np.all(np.isinf(p.logpdf((a - 1., b + 1.)))), \
            '{}: log PDF must be -inf outside [{}, {}].'.format(name, a, b)
        assert isinstance(pickle.loads(pickle.dumps(p)), priors.Prior), \
            '{}: not pickleable.'.format(name)

    # Test combining priors.
    var = priors.VariancePrior()
    length = priors.LengthScalePrior()
    noise = priors.NoisePrior()

    prior = length
    prior *= 2
    prior = var + prior
    prior += noise

    assert len(prior) == 4, 'Combined prior has incorrect length.'
    assert all((
        prior[0] is var[0],
        prior[1] is length[0],
        prior[2] is length[0],
        prior[3] is noise[0]
    )), 'Combined prior does not reduce to its components.'

    sample = prior.rvs(5)
    assert sample.shape == (5, 4), \
        'Combined prior: sample has incorrect shape.'

    lp = tuple(prior.logpdf(s) for s in sample)
    assert len(lp) == 5, \
        'Combined prior: log PDF does not match sample size.'
    assert np.all(np.isfinite(lp)), \
        'Combined prior: sample log PDF is not finite.'
