"""
signals.py
  Functions for generating different types of signals.
"""

import numpy as np

def bandlimited(dt, t_final, max_freq, mean=0.0, std=1.0, n=None, rng=None):
    """Generate a random signal with equal power below a maximum frequency.

    Parameters
    ----------
    dt : float
        Time difference between consecutive signal points [in seconds]
    t_final : float
        Length of the signal [in seconds]
    max_freq : float
        Maximum frequency of the signal [in Hertz]
    mean : float
        Signal mean (default = 0.0)
    std : float
        Signal standard deviation (default = 1.0)
    n : integer
        Number of signals to generate

    Returns
    -------
    s : array_like
        Generated signal(s), where each row is a signal, and each column a time
    """

    if rng is None:
        rng = np.random

    df = 1. / t_final    # fundamental frequency

    ns = n if n is not None else 1     # number of independent signals
    nt = np.round(t_final / dt)        # number of time points / frequencies
    nf = np.round(max_freq / df)       # number of non-zero frequencies
    assert nf < nt

    theta = rng.uniform(low=0, high=2*np.pi, size=(ns, nf))
    B = np.cos(theta) + 1.0j * np.sin(theta)

    A = np.zeros((ns, nt), dtype=np.complex)
    A[:,1:nf+1] = B
    A[:,-nf:] = np.conj(B)[:,::-1]

    S = np.fft.ifft(A, axis=1).real

    S = (std / S.std(axis=1))[:,None] * (S - S.mean(axis=1)[:,None] + mean)
    return S if n is not None else S.flatten()
