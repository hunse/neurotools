
import numpy as np

import neurotools
import neurotools.filters

import matplotlib.pyplot as plt

def get_spectrum(t, x, y):
    assert x.shape == y.shape and t.size == x.shape[0]

    nt = len(t)
    df = 1. / (t[-1] - t[0])
    f = df * np.arange(nt)[:nt/2]
    Fx = np.fft.fft(x, axis=0)[:nt/2]
    Fy = np.fft.fft(y, axis=0)[:nt/2]
    F = Fy / Fx
    mag = np.abs(F)
    ang = np.unwrap(np.angle(F), axis=0)
    if F.ndim > 1:
        mag = mag.mean(tuple(np.arange(1, F.ndim)))
        ang = ang.mean(tuple(np.arange(1, F.ndim)))

    return f, mag, ang


def test_lowpass():
    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 1e-2 / dt
    alpha = 1. / tau

    s = np.random.normal(size=nt)

    result1 = neurotools.filters.lowpass(s, alpha=alpha)

    tk = np.arange(0, 30*tau)
    k = alpha * np.exp(-alpha*tk)
    result2 = np.convolve(s, k, mode='full')[:nt]

    plt.figure(1)
    plt.clf()

    plt.subplot(211)
    plt.plot(t, result1)
    plt.plot(t, result2)

    plt.subplot(212)
    plot_spectrum(t, result1)
    plot_spectrum(t, result2)
    plt.show()


def test_alpha():

    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 1e-2 / dt
    alpha = 1. / tau

    s = np.random.normal(size=nt)

    result1 = neurotools.filters.alpha(s, alpha=alpha)

    tk = np.arange(0, 10*tau)
    k = alpha**2 * tk * np.exp(-alpha*tk)
    result2 = np.convolve(s, k, mode='full')[:nt]

    plt.figure(1)
    plt.clf()

    plt.subplot(211)
    plt.plot(t, result1)
    plt.plot(t[:-1], result2[1:])

    plt.subplot(212)
    plot_spectrum(t, result1)
    plot_spectrum(t, result2)
    plt.show()


def test_multi(kind='alpha'):

    if kind == 'alpha':
        func = neurotools.filters.alpha
    else:
        func = neurotools.filters.lowpass

    dt = 1e-4
    tend = 10.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 1e-2 / dt
    alpha = 1. / tau

    x = np.random.normal(size=(nt,100))
    y = func(x, alpha=alpha)


    plt.figure(1)
    plt.clf()

    plt.subplot(211)
    plt.plot(t, y[:,0])

    plt.subplot(212)
    plot_spectrum(t, y)
    plt.show()


def test_lti():
    import scipy.signal as sps

    dt = 1e-3
    tend = 10.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 1e-2
    tf = sps.cont2discrete(([1], [tau, 1]), dt, method='euler')[:2]
    # tf = sps.cont2discrete(([1], [tau**2, 2*tau, 1]), dt, method='zoh')[:2]
    # tf = sps.butter(6, dt / (np.pi * tau))
    # tf = sps.cheby1(5, 40, dt / (np.pi * tau))
    print tf

    # tf2 = sps.cont2discrete(([1], [tau, 1]), dt, method='zoh')[:2]
    tf2 = sps.cont2discrete(([1], [tau, 1]), dt, method='gbt', alpha=0.5)[:2]

    ### plot filtered signal and spectrum
    x = np.random.normal(size=(nt,1000))
    y = neurotools.filters.lti(x, tf)
    y2 = neurotools.filters.lti(x, tf2)

    plt.figure(1)
    plt.clf()

    plt.subplot(211)
    plt.plot(t, y[:,0])
    plt.plot(t, y2[:,0])

    f, mag, ang = get_spectrum(t, x, y)
    f2, mag2, ang2 = get_spectrum(t, x, y2)
    plt.subplot(223)
    plt.loglog(f, mag)
    plt.loglog(f2, mag2)

    plt.subplot(224)
    delay = ang / (2 * np.pi * f)
    delay2 = ang2 / (2 * np.pi * f2)
    plt.semilogx(f[1:], delay[1:])
    plt.semilogx(f2[1:], delay2[1:])

    ### plot impulse response
    tend = 10 * tau
    t = dt * np.arange(tend / dt)
    nt = len(t)

    x = np.zeros(nt)
    x[0] = 1. / dt
    y = neurotools.filters.lti(x, tf)

    plt.figure(2)
    plt.clf()
    plt.plot(t, y)
    print("Impulse response sum:", y.sum() * dt)

    ### plot sine input
    x = np.sin(3 * (2 * np.pi) * t)
    y = neurotools.filters.lti(x, tf)
    plt.figure(3)
    plt.clf()
    plt.plot(t, x)
    plt.plot(t, y)

    plt.show()


if __name__ == '__main__':
    # plt.ion()

    test_lti()

    # test_alpha()
    # test_lowpass()

    # test_multi('alpha')
    # test_multi('lowpass')
