# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import linalg

from .util import atleast_2d_column

__all__ = 'PCA',


# adapted from sklearn.decomposition.PCA
class PCA(object):
    """
    Principal component analysis.  Transform a set of observations in feature
    space to orthogonal principal components and back.

    y : (nsamples, nfeatures)
        Array of features.
    npc : optional, integer or float in (0, 1)
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

    def transform(self, y=None, copy=True):
        """
        Transform a set of observations into PC space.

        y : (nobservations, nfeatures), optional
            Array of features to transform.  If not provided, the original y
            used to construct the class is transformed.
        copy : boolean, default True
            Whether to copy y or modify in place.

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

        y = atleast_2d_column(y, copy=copy)
        y -= self._mean

        z = np.dot(y, self.pc[:self.npc].T)
        z /= self.std[:self.npc]

        return z

    def inverse(self, z, var=None, y_cov=True):
        """
        Transform principal components back to feature space.

        z : (nobservations, npc)
            Array of principal components to inverse transform.
        var : (nobservations, npc), optional
            Variance of z.  PCs are assumed to be uncorrelated, so each row of
            var represents the diagonal of the covariance matrix for the
            corresponding row of z.  If given, both the inverse-transform y and
            its covariance matrices are returned.
        y_cov : boolean, default True
            Whether to calculate the full covariance matrices of y or
            only the diagonals.  If true, an array
            (nobservations, nfeatures, nfeatures) is returned, otherwise
            (nobservations, nfeatures).  No effect if var is not given.
            Note that the covariance between rows of y is never calculated,
            this only refers to covariance between components of rows.

        """
        z = atleast_2d_column(z, copy=False)

        if var is not None:
            var = atleast_2d_column(var, copy=False)
            if z.shape != var.shape:
                raise ValueError('z and var must have the same shape.')

        # transformation matrix
        A = self.pc[:self.npc] * self.std[:self.npc, np.newaxis]

        y = np.dot(z, A)
        y += self._mean

        if var is not None:
            # cov = A^t.diag(row).A for each row in var
            # np.einsum() does this efficiently and easily
            if y_cov:
                # full covariance matrix
                # cov_aij = sum_k(A_ki var_ak A_kj)
                subscripts = 'ki,ak,kj'
            else:
                # diagonals only
                # cov_aii = sum_k(A_ki var_ak A_ki)
                subscripts = 'ki,ak,ki->ai'
            cov = np.einsum(subscripts, A, var, A)
            return y, cov
        else:
            return y
