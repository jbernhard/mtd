# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import linalg

from .util import atleast_2d_column

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

    Principal components are "whitened" so that each has unit variance.  This
    simplifies prior specifications for GP training.

    Note that the full SVD of y is always saved:
      - PCA.weights and PCA.pc always return information for all PC.
      - PCA.npc only determines the number of components used by transform()
        and inverse(). It may be reset at any time.

    """
    def __init__(self, y, npc=None):
        y = atleast_2d_column(y, copy=True)
        self._mean = y.mean(axis=0)
        y -= self._mean

        self._svd = linalg.svd(y, full_matrices=False)

        nsamples, nfeatures = y.shape
        self._sqrt_nsamples = np.sqrt(nsamples)

        # determine number of PC
        if npc is None or npc > nfeatures:
            self.npc = nfeatures
        elif 0 < npc < 1:
            self.npc = np.count_nonzero(self.weights.cumsum() < npc) + 1
        else:
            self.npc = npc

    @property
    def std(self):
        """
        Standard deviation (sqrt of variance) explained by each PC.

        """
        s = self._svd[1]
        return s / self._sqrt_nsamples

    @property
    def var(self):
        """
        Variance explained by each PC.

        """
        return np.square(self.std)

    @property
    def weights(self):
        """
        Fraction of variance explained by each PC.

        """
        var = self.var
        return var / var.sum()

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
        # transformed and whitened y:  z = y.V.S^-1*sqrt(nsamples)
        #                                = (U.S.Vt).V.S^-1*sqrt(nsamples)
        #                                = U*sqrt(nsamples)
        # Therefore the left-singular vectors U can be reused to calculate the
        # PC of the original y.  Right-singular vectors V are used for any
        # other y.
        if y is None:
            U = self._svd[0]
            return self._sqrt_nsamples * U[:, :self.npc]

        y = atleast_2d_column(y, copy=True)
        y -= self._mean

        z = np.dot(y, self.pc[:self.npc].T)
        z /= self.std[:self.npc]

        return z

    def inverse(self, z):
        """
        Transform principal components back to feature space.

        z: (nobservations, npc)

        """
        z = atleast_2d_column(z, copy=True)
        z *= self.std[:self.npc]

        y = np.dot(z, self.pc[:self.npc])
        y += self._mean

        return y
