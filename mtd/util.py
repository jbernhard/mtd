# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

__all__ = 'atleast_2d_column',


def atleast_2d_column(arr, dtype=float, copy=False):
    """
    Cast input as an array with at least two dimensions.  If the input is
    one-dimensional, an extra axis is appended so the shape is column-like
    (n, 1) instead of row-like (1, n) which would be produced by
    np.atleast_2d.

    Keyword args (passed directly to np.array):

    dtype : default float
    copy : default False

    """
    arr = np.array(arr, dtype=dtype, copy=copy)

    if arr.ndim <= 1:
        arr = arr.reshape(-1, 1)

    return arr
