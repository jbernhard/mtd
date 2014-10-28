# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .. import PCA


def test_pca():
    """principal component analysis"""

    # trivial test data which can be fully described by one PC
    nsamples, nfeatures = 5, 2
    y = np.linspace(0, 1, nsamples*nfeatures).reshape(nsamples, nfeatures)

    # test with 2 PC
    pca = PCA(y)
    assert np.allclose(pca.weights, (1, 0)), \
        'Incorrect PC weights.\n{} != (1, 0)'.format(pca.weights)
    assert np.allclose(np.abs(pca.pc), np.sqrt(2)/2), \
        'Incorrect PCs.\n{} != +/-sqrt(2)/2'.format(pca.pc)
    ratio = np.divide(*pca.pc)
    assert np.allclose(ratio, (1, -1)) or np.allclose(ratio, (-1, 1)), \
        'Incorrect PC ratio.\n{} != (+/-1, -/+1)'.format(ratio)

    z1 = pca.transform()
    z2 = pca.transform(y)
    assert z1.shape == y.shape, \
        'Transformed y has wrong shape.\n{} != {}'.format(z1.shape, y.shape)
    assert np.allclose(z1, z2), \
        'Transformations are inconsistent.\n{} != {}'.format(z1, z2)
    y2 = pca.inverse(z1)
    assert np.allclose(y2, y), \
        ('Inverse transformation does not recover original data.\n'
         '{} != {}'.format(y2, y))

    # now check it still works with 1 PC
    npc = 1
    pca = PCA(y, npc=npc)
    z = pca.transform()
    assert z.shape == (nsamples, npc), \
        'Transformed data should have {} PC, not {}.'.format(npc, z.shape[1])
    y2 = pca.inverse(pca.transform(y))
    assert np.allclose(y2, y), \
        ('Inverse transformation does not recover original data.\n'
         '{} != {}'.format(y2, y))

    # random data
    nsamples, nfeatures = 5, 3
    y = np.random.rand(nsamples, nfeatures)
    pca = PCA(y)

    explained_var = pca.weights.sum()
    assert np.allclose(explained_var, 1), \
        'All PC must explain full variance.\n{} != 1'.format(explained_var)

    z1 = pca.transform()
    z2 = pca.transform(y)
    assert z1.shape == y.shape, \
        'Transformed y has wrong shape.\n{} != {}'.format(z1.shape, y.shape)
    assert np.allclose(z1, z2), \
        'Transformations are inconsistent.\n{} != {}'.format(z1, z2)
    a = np.random.rand(2, nfeatures)
    a2 = pca.inverse(pca.transform(a))
    assert np.allclose(a2, a), \
        ('Inverse transformation does not recover original data.\n'
         '{} != {}'.format(a2, a))

    mean = z1.mean(axis=0)
    assert np.allclose(mean, 0), \
        'Transformed y must have mean zero.\n{} != 0'.format(mean)

    pc = pca.pc
    inner = np.array([np.inner(pc[i], pc[j])
                      for (i, j) in ((0, 1), (0, 2), (1, 2))])
    assert np.allclose(inner, 0), \
        'PC must be orthogonal.\n{} != 0'.format(inner)

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

    # normalization
    pca = PCA(y, normalize=True)
    z1 = pca.transform()
    z2 = pca.transform(y)
    mean = z1.mean(axis=0)
    assert np.allclose(mean, 0), \
        'Transformed y must have mean zero.\n{} != 0'.format(mean)
    var = z1.var()
    assert np.allclose(var, 1), \
        'Transformed y must have unit variance.\n{} != 1'.format(var)
    assert np.allclose(z1, z2), \
        'Transformations are inconsistent.\n{} != {}'.format(z1, z2)
    y2 = pca.inverse(z1)
    assert np.allclose(y2, y), \
        ('Inverse transformation does not recover original data.\n'
         '{} != {}'.format(y2, y))
    a = np.random.rand(2, nfeatures)
    a2 = pca.inverse(pca.transform(a))
    assert np.allclose(a2, a), \
        ('Inverse transformation does not recover original data.\n'
         '{} != {}'.format(a2, a))
