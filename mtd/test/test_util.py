# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from ..util import atleast_2d_column


def test_util():
    """utility functions"""

    n = 10
    x = np.linspace(0, 1, 10)

    xcol = atleast_2d_column(x)
    assert xcol.shape == (n, 1), \
        'New array is not a column.'
    assert not xcol.flags.owndata and xcol.base is x, \
        'New array is a copy, not a view.'

    xcolcopy = atleast_2d_column(x, copy=True)
    assert xcolcopy.shape == (n, 1), \
        'New array is not a column.'
    assert xcolcopy.flags.owndata or xcolcopy.base is not x, \
        'New array is a view, not a copy.'

    xrow = atleast_2d_column(np.atleast_2d(x))
    assert xrow.shape == (1, n), \
        '2D input array should retain its shape.'
    assert not xrow.flags.owndata and xrow.base is x, \
        'New array is a copy, not a view.'

    xint = np.array([1, 2, 3], dtype=int)
    xintcol = atleast_2d_column(xint)
    assert xintcol.shape == (3, 1), \
        'New array is not a column.'
    assert xintcol.dtype == float, \
        'Integer array was not upcast to float.'
    assert xintcol.flags.owndata or xintcol.base is not xint, \
        'New array is a view, not a copy.'
