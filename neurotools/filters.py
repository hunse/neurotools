
import numpy as np
# import numpy.random as npr

# import scipy as sp
# import scipy.signal


# def alpha(signals, **kwargs):
#     return firstorder(signals, kind='alpha', **kwargs)

def alpha(signals, alpha, axis=0):

    outputs = np.zeros_like(signals)

    nt = signals.shape[axis]
    t = np.arange(1, nt + 1)

    signalsr = np.rollaxis(signals, axis)
    outputsr = np.rollaxis(outputs, axis)

    xi = np.zeros_like(signalsr[0])
    yi = np.zeros_like(signalsr[0])
    # xi = np.array(signalsr[0])

    alpha = 1. - np.exp(-alpha)

    for i, [ti, ui] in enumerate(zip(t, signalsr)):
        xi += alpha * (ui - xi)
        yi += alpha * (xi - yi)
        outputsr[i] = yi

    return outputsr


def lowpass(signals, alpha, axis=0):

    outputs = np.zeros_like(signals)

    nt = signals.shape[axis]
    t = np.arange(1, nt + 1)

    signalsr = np.rollaxis(signals, axis)
    outputsr = np.rollaxis(outputs, axis)

    xi = np.zeros_like(signalsr[0])
    # xi = np.array(signalsr[0])

    alpha = 1 - np.exp(-alpha)

    for i, [ti, ui] in enumerate(zip(t, signalsr)):
        xi += alpha * (ui - xi)
        outputsr[i] = xi

    return outputsr


# def firstorder(signals, kind='lowpass', axis=0, alpha=None):
#     """
#     Apply a first-order IIR filter to a set of signals.

#     Parameters
#     ----------
#     signals : array_like
#         Signals to filter.
#     axis : int
#         Axis to filter across. Defaults to the first axis.
#     """

#     outputs = np.zeros_like(signals)

#     nt = signals.shape[axis]
#     t = np.arange(1, nt + 1)

#     signalsr = np.rollaxis(signals, axis)
#     outputsr = np.rollaxis(outputs, axis)

#     xi = np.zeros_like(signalsr[0])




#     for i, [ti, ui] in enumerate(zip(t, signalsr)):
#         # xi += (1. / ti - alpha) * xi + ui
#         xi += (1. / ti - alpha) * xi + alpha * ui
#         # xi += (1. / ti - alpha) * (xi - ui)
#         outputsr[i] = xi

#     return outputsr


def lti(signals, transfer_fn, axis=0):

    outputs = np.zeros_like(signals)

    nt = signals.shape[axis]
    t = np.arange(1, nt + 1)

    signalsr = np.rollaxis(signals, axis)
    outputsr = np.rollaxis(outputs, axis)

    a, b = transfer_fn
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    na, nb = len(a), len(b)
    # assert a.ndim == 1, b.ndim == 1

    if b[0] != 1.:
        a = a / b[0]
        b = b / b[0]

    if 1:
        # no states
        a = a[::-1]  # flip a
        b = b[:0:-1]  # flip b and drop first element (equal to 1)

        signalsr = np.concatenate([np.zeros_like(signalsr[:na]), signalsr], axis=0)
        outputsr = np.concatenate([np.zeros_like(outputsr[:nb - 1]), outputsr], axis=0)

        for i in xrange(nt):
            x = signalsr[i:i + na]
            y = outputsr[i:i + nb - 1]
            outputsr[i + nb - 1] = np.dot(a, x) - np.dot(b, y)

        return outputsr[nb - 1:]
    else:
        # states
        a = a[::-1]  # flip a
        b = b[:0:-1]  # flip b and drop first element (equal to 1)

        x = np.zeros_like(signalsr[:len(a)])
        y = np.zeros_like(outputsr[:len(b)])

        xzero = 0
        yzero = 0
        a = np.roll(a, 1)

        for i, si in enumerate(signalsr):
            x[xzero] = si
            y[yzero] = np.dot(a, x) - np.dot(b, y)
            outputsr[i] = y[yzero]
            xzero = (xzero + 1) % len(x)
            yzero = (yzero + 1) % len(y)
            a = np.roll(a, 1)
            b = np.roll(b, 1)

        return outputsr

# from util.conversion import tofrom_matrix
# # def lowpass_filter_conv(s, dt, tau_c, s0=None, axis=-1):
# #     t_kern = np.arange(0,5*tau_c,dt)
# #     kern = np.exp(-t_kern/tau_c)
# #     kern /= kern.sum()

# @tofrom_matrix('s',0)
# def lowpass_filter(s, dt, tau_c, s0=None, axis=-1):
#     """Low-pass filter a signal using a causal filter

#     Parameters
#     ----------
#     s : array_like
#         Input signal
#     dt : float
#         Time difference between consecutive signal points [in seconds]
#     tau_c : float
#         Filter time constant [in seconds]
#     s0 : array_like
#         Initial conditions for the filtered signal (default: mean of early signal)
#     axis : int
#         Index of the axis to filter along (default: last axis)

#     Returns
#     -------
#     s : array_like
#         Filtered copy of the original signal
#     """
#     s = s.copy()
#     nt = s.shape[axis]
#     pind = [slice(None) for i in xrange(s.ndim)]
#     nind = [slice(None) for i in xrange(s.ndim)]
#     pind[axis] = 0

#     ### set initial conditions
#     if s0 is not None:
#         s[pind] = s0
#     else:
#         ind = np.round(5*tau_c / dt)
#         nind[axis] = slice(None, min(ind, nt))
#         s[pind] = s[nind].mean(axis=axis)

#     ### filtering is much faster when done with a loop than with convolution
#     # alpha = dt / tau_c           # forward Euler
#     alpha = dt / (tau_c + dt)    # reverse Euler
#     # for i in xrange(1,nt):
#     #     s[:,i] = s[:,i-1] + alpha*(s[:,i] - s[:,i-1])

#     for i in xrange(1,nt):
#         pind[axis] = i-1
#         nind[axis] = i
#         s[nind] = s[pind] + alpha*(s[nind] - s[pind])

#     return s

# # @tofrom_matrix('s',0)
# # def lowpass_filter2(s, dt, tau_c, s0=None):
# #     nt = s.shape[1]
# #     if s0 is None:
# #         ind = np.round(5*tau_c / dt)
# #         s0 = s[:,:min(ind, nt)].mean(axis=1)
# #         # s[:,0] =

# #     ### filtering is much faster when done with a loop than with convolution
# #     print s0
# #     t = dt*np.arange(0,nt)
# #     system = (1, [tau_c, 1])
# #     _, s, _ = sp.signal.lsim(system, s.T, t)
# #     # _, s, x = sp.signal.lsim(system, s.T, t, X0=s0[0])

# #     # print x[:100]
# #     # print s[:100]

# #     return s.T

