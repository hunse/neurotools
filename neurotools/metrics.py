"""
metrics.py
  Functions for computing various norms, distances,
  information metrics, etc.
"""

import numpy as np

def norm(x, axis=-1, p=2, keepdims=False):
    """Compute p-norm (defaults to Euclidean norm, p = 2)

    Parameters
    ----------
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute the norm. If not given,
        the last axis is used.
    p : float, optional
        The 'p' of the p-norm. Defaults to 2, the Euclidean norm.
    keepdims : bool, optional
        If True, the reduced axes are left in the result as dimensions
        with size one, for easy broadcasting. Defaults to False.
    """
    if p == 2:  # Euclidean norm
        return np.sqrt((x**2).sum(axis, keepdims=keepdims))
    elif p == 1:  # Manhattan norm
        return np.abs(x).sum(axis, keepdims=keepdims)
    elif p == np.inf:  # maximum norm
        return np.abs(x).max(axis, keepdims=keepdims)
    elif p == -np.inf:  # minimum norm
        return np.abs(x).min(axis, keepdims=keepdims)
    elif p == 0:  # counting norm
        return (x != 0).sum(axis, keepdims=keepdims)
    else:
        return (np.abs(x)**p).sum(axis, keepdims=keepdims)**(1./p)
