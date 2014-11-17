# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from nose.tools import assert_raises

from .. import PCA


def test_pca():
    """principal component analysis"""

    # trivial test data which can be fully described by one PC
    nsamples, nfeatures, npc = 5, 2, 1
    y = np.linspace(0, 1, nsamples*nfeatures).reshape(nsamples, nfeatures)

    pca = PCA(y, npc=npc)

    assert np.allclose(pca.weights, (1, 0)), \
        'Incorrect PC weights.\n{} != (1, 0)'.format(pca.weights)
    assert np.allclose(np.abs(pca.pc), np.sqrt(2)/2), \
        'Incorrect PCs.\n{} != +/-sqrt(2)/2'.format(pca.pc)
    ratio = np.divide(*pca.pc)
    assert np.allclose(ratio, (1, -1)) or np.allclose(ratio, (-1, 1)), \
        'Incorrect PC ratio.\n{} != (+/-1, -/+1)'.format(ratio)

    z1 = pca.transform()
    z2 = pca.transform(y)
    assert np.allclose(z1, z2), \
        'Transformations are inconsistent.\n{} != {}'.format(z1, z2)
    expected_shape = nsamples, npc
    assert z1.shape == expected_shape, \
        ('Transformed y has wrong shape.\n{} != {}'
         .format(expected_shape, y.shape))
    y2 = pca.inverse(z2)
    assert np.allclose(y2, y), \
        ('Inverse transformation does not recover original data.\n'
         '{} != {}'.format(y2, y))

    mean = z1.mean(axis=0)
    assert np.allclose(mean, 0), \
        'Transformed y must have mean zero.\n{} != 0'.format(mean)
    var = z1.var(axis=0)
    assert np.allclose(var, 1), \
        'Transformed y must have unit variance.\n{} != 0'.format(var)

    mu, cov = pca.inverse(0, var=1)
    assert np.allclose(mu, y.mean(axis=0)), \
        'Inverse transform of zero must equal sample mean.'
    assert np.allclose(cov, y.var(axis=0)), \
        'Inverse transform of unit variance must equal sample variance.'

    assert_raises(ValueError, pca.inverse, [0, 1], 0)

    # random data
    nsamples, nfeatures = 5, 3
    y = np.random.rand(nsamples, nfeatures)
    pca = PCA(y)

    var_sum = y.var(axis=0).sum()
    explained_var = pca.var.sum()
    assert np.allclose(explained_var, var_sum), \
        ('All PC must explain full variance.\n{} != {}'
         .format(explained_var, var_sum))

    explained_var_frac = pca.weights.sum()
    assert np.allclose(explained_var_frac, 1), \
        ('All PC must explain full variance fraction.\n{} != 1'
         .format(explained_var_frac))

    z1 = pca.transform()
    z2 = pca.transform(y)
    assert z1.shape == y.shape, \
        'Transformed y has wrong shape.\n{} != {}'.format(z1.shape, y.shape)
    assert np.allclose(z1, z2), \
        'Transformations are inconsistent.\n{} != {}'.format(z1, z2)

    y2 = pca.inverse(z2)
    assert np.allclose(y2, y), \
        ('Inverse transformation does not recover original data.\n'
         '{} != {}'.format(y2, y))
    a = np.random.rand(2, nfeatures)
    a2 = pca.inverse(pca.transform(a))
    assert np.allclose(a2, a), \
        ('Inverse transformation does not recover original data.\n'
         '{} != {}'.format(a2, a))

    mean = z1.mean(axis=0)
    assert np.allclose(mean, 0), \
        'Transformed y must have mean zero.\n{} != 0'.format(mean)
    var = z1.var(axis=0)
    assert np.allclose(var, 1), \
        'Transformed y must have unit variance.\n{} != 0'.format(var)

    pc = pca.pc
    inner = np.array([np.inner(pc[i], pc[j])
                      for (i, j) in ((0, 1), (0, 2), (1, 2))])
    assert np.allclose(inner, 0), \
        'PC must be orthogonal.\n{} != 0'.format(inner)
    norm = np.square(pc).sum(axis=1)
    assert np.allclose(norm, 1), \
        'PC must be unit vectors.\n{} != 0'.format(norm)

    mu, cov = pca.inverse(np.zeros((1, nfeatures)),
                          var=np.ones((1, nfeatures)), y_cov=False)
    assert np.allclose(mu, y.mean(axis=0)), \
        'Inverse transform of zero must equal sample mean.'
    assert np.allclose(cov, y.var(axis=0)), \
        'Inverse transform of unit variance must equal sample variance.'

    # minimum explained variance
    npc = nfeatures - 1
    var = .99*pca.weights.cumsum()[npc - 1]
    pca = PCA(y, npc=var)
    print(pca.weights.cumsum())
    assert pca.npc == 2, \
        '{:g} of variance requires {} PC, not {}.'.format(var, npc, pca.npc)

    z1 = pca.transform()
    z2 = pca.transform(y)
    shape = nsamples, npc
    assert z1.shape == shape, \
        'Transformed y has wrong shape.\n{} != {}'.format(z1.shape, shape)
    assert np.allclose(z1, z2), \
        'Transformations are inconsistent.\n{} != {}'.format(z1, z2)

    z = np.array([[.1, .2], [-1., 1.5]])
    zvar = np.array([[.05, .04], [.8, .9]])
    ycov = pca.inverse(z, var=zvar, y_cov=True)[1]
    yvar = pca.inverse(z, var=zvar, y_cov=False)[1]

    shape = (2, nfeatures, nfeatures)
    assert ycov.shape == shape, \
        'Inverse transformed covariance has incorrect shape.\n' \
        '{} != {}'.format(ycov.shape, shape)
    shape = (2, nfeatures)
    assert yvar.shape == shape, \
        'Inverse transformed variance has incorrect shape.\n' \
        '{} != {}'.format(yvar.shape, shape)
    ycovdiag = np.diagonal(ycov, axis1=1, axis2=2)
    assert np.all(ycovdiag == yvar), \
        'Diagonal of covariance matrix does not agree with variance.\n' \
        '{} != {}'.format(ycovdiag, yvar)
