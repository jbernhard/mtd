# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import linalg

__all__ = 'PCA',


# adapted from sklearn.decomposition.PCA
class PCA(object):
    """
    Principal component analysis.

    y: (nsamples, nfeatures)
        Array of observations.
    npc: optional, integer or float in (0, 1)
        If an integer, explicitly sets the number of PC.  If a float in (0, 1),
        it sets the minimum explained variance fraction; the number of PC is
        automatically determined.  If not provided, all PC are used.
    normalize:  boolean, default false
        Whether to normalize features to unit variance (i.e. divide the columns
        of y by their standard deviation).

    Note that the full SVD of y is always saved:
      - PCA.weights and PCA.pc always return information for all PC.
      - PCA.npc only determines the number of components used by transform()
        and inverse(). It may be reset at any time.

    """
    def __init__(self, y, npc=None, normalize=False):
        y = np.copy(np.atleast_2d(y))
        self._mean = y.mean(axis=0)
        y -= self._mean

        if normalize:
            self._std = y.std(axis=0)
            y /= self._std
        else:
            self._std = None

        self._svd = linalg.svd(y, full_matrices=False)

        nfeatures = y.shape[1]
        if npc is None or npc > nfeatures:
            self.npc = nfeatures
        elif 0 < npc < 1:
            self.npc = np.count_nonzero(self.weights.cumsum() < npc) + 1
        else:
            self.npc = npc

    @property
    def weights(self):
        """
        Fraction of variance explained by each PC.

        """
        s_sq = np.square(self._svd[1])
        return s_sq / s_sq.sum()

    @property
    def pc(self):
        """
        PC vectors.

        """
        return self._svd[2]

    def transform(self, y=None):
        """
        Transform a set of observations into PC space.

        y: (nobservations, nfeatures), optional
            If not provided, the original y used to construct the class is
            transformed.

        """
        # SVD:  y = U.S.Vt
        # transformed y:  z = y.V = (U.S.Vt).V = U.S
        # Therefore the left-singular vectors U can be reused to calculate the
        # PC of the original y.  Right-singular vectors V are used for any
        # other y.
        if y is None:
            U, s, Vt = self._svd
            return (U[:, :self.npc]) * (s[:self.npc])

        y = np.copy(np.atleast_2d(y))

        y -= self._mean
        if self._std is not None:
            y /= self._std

        return np.dot(y, self.pc[:self.npc].T)

    def inverse(self, z):
        """
        Transform principal components back to feature space.

        z: (nobservations, npc)

        """
        y = np.dot(z, self.pc[:self.npc])

        if self._std is not None:
            y *= self._std
        y += self._mean

        return y
