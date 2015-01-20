# -*- coding: utf-8 -*-

from __future__ import division

import pickle

import numpy as np

from .. import priors


def _check_pickle(prior, name):
    prior2 = pickle.loads(pickle.dumps(prior))
    conds = (
        isinstance(prior2, priors.Prior),
        len(prior2) == len(prior),
        prior2.__getstate__() == prior.__getstate__()
    )
    assert all(conds), '{}: not pickleable.'.format(name)


def test_priors():
    """prior distributions"""

    for name in priors.__all__:
        if name == 'Prior':
            continue

        p = getattr(priors, name)()
        sample = p.rvs(10)
        a, b = {
            'UniformPrior': (0, 1),
            'ExpPrior': (0, np.inf),
            'InvGammaPrior': (0, np.inf),
            'BetaPrior': (0, 1),
            'LogPrior': (0, np.inf),
        }[name]

        assert np.all((sample >= a) & (sample <= b)), \
            '{}: must produce values in [{}, {}].'.format(name, a, b)
        assert np.all(np.isfinite(p.logpdf(sample))), \
            '{}: sample log PDF is not finite.'.format(name)
        assert np.all(np.isinf(p.logpdf((a - 1., b + 1.)))), \
            '{}: log PDF must be -inf outside [{}, {}].'.format(name, a, b)
        _check_pickle(p, name)

    # Test combining priors.
    var = priors.InvGammaPrior()
    length = priors.BetaPrior()
    noise = priors.LogPrior()

    # build up a compound prior in a funny way to test all operations
    prior = var
    prior *= 2
    prior += 2*length + noise

    print(len(prior))
    assert len(prior) == 5, 'Combined prior has incorrect length.'
    assert all((
        prior[0] is var[0],
        prior[1] is var[0],
        prior[2] is length[0],
        prior[3] is length[0],
        prior[4] is noise[0]
    )), 'Combined prior does not reduce to its components.'

    sample = prior.rvs(4)
    assert sample.shape == (4, 5), \
        'Combined prior: sample has incorrect shape.'

    lp = tuple(prior.logpdf(s) for s in sample)
    assert len(lp) == 4, \
        'Combined prior: log PDF does not match sample size.'
    assert np.all(np.isfinite(lp)), \
        'Combined prior: sample log PDF is not finite.'
    _check_pickle(prior, 'Combined prior')
