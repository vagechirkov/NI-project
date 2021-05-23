"""
Some code taken from https://github.com/caglarcakan/stimulus_neural_populations
Copyright (c) 2019, Caglar Cakan BSD 2-Clause License
"""

from scipy import ndimage, signal
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def kuramoto(traces, smoothing=0.0, dt=0.1, debug=False,
             peakrange=[0.1, 0.2]):
    phases = []
    nTraces = len(traces)
    peakslist = []
    for n in range(nTraces):
        tList = np.dot(range(len(traces[n])), dt/1000)
        a = traces[n]

        # find peaks
        if smoothing > 0:
            # smooth data
            a = ndimage.filters.gaussian_filter(traces[n], smoothing)
        maximalist = signal.find_peaks_cwt(a, np.arange(peakrange[0],
                                                        peakrange[1]))
        # maximalist = signal.find_peaks(a, distance=20, prominence=2)[0]
        maximalist = np.append(maximalist, len(traces[n])-1).astype(int)
        if debug:
            peakslist.append(maximalist)

        if len(maximalist) > 1:
            phases.append([])
            lastMax = 0
            for m in maximalist:
                for t in range(lastMax, m):
                    phi = 2 * np.pi * float(t - lastMax) / float(m - lastMax)
                    phases[n].append(phi)
                lastMax = m
            phases[n].append(2 * np.pi)
        else:
            return 0
    # determine kuramoto order paramter
    kuramoto = []
    for t in range(len(tList)):
        R = 1j*0
        for n in range(nTraces):
            R += np.exp(1j * phases[n][t])
        R /= nTraces
        kuramoto.append(np.absolute(R))
    if debug:
        return kuramoto, phases, peakslist, traces
    else:
        return kuramoto


def plot_kuramoto_example(traces):
    kur, phases, peakslist, traces = fast_kuramoto(traces, debug=True)
    plt.figure(figsize=(20, 15))
    ax0 = plt.subplot(311)
    n = traces.shape[0]
    for i in range(n):
        ax0.plot(np.arange(len(traces[i])), traces[i], c='C{}'.format(i))
        for m in peakslist[i]:
            ax0.scatter(m, traces[i][m], c='r', zorder=20, s=10.5)
    plt.ylabel('Rate of E [Hz]')
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1 = plt.subplot(312)
    for i in range(n):
        ax1.plot(phases[i], c='C{}'.format(i),)
    plt.ylabel('Phase in radians')
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.subplot(313)
    plt.plot(kur, c='g')
    plt.ylabel('Kuramoto order parameter')
    plt.xlabel('Time, samples')

    plt.subplots_adjust(hspace=.0)
    plt.show()


def fast_kuramoto(traces, smoothing=5.0, dt=0.1, debug=False):
    nTraces, nTimes = traces.shape
    peakslist = []
    phases = np.empty_like(traces)
    for n in range(nTraces):
        a = traces[n]

        # find peaks
        if smoothing > 0:
            # smooth data
            a = ndimage.filters.gaussian_filter(traces[n], smoothing)
        maximalist = signal.find_peaks(a, distance=10, prominence=5)[0]
        maximalist = np.append(maximalist, len(traces[n])-1).astype(int)

        if debug:
            peakslist.append(maximalist)

        if len(maximalist) > 1:
            phases[n, :] = _estimate_phase(maximalist, nTimes)
        else:
            return 0

    # determine kuramoto order paramter
    kuramoto = _estimate_r(nTraces, nTimes, phases)

    if debug:
        return kuramoto, phases, peakslist, traces
    return kuramoto


@jit(nopython=True)
def _estimate_phase(maximalist, n_times):
    lastMax = 0
    phases = np.empty((n_times), dtype=np.float64)
    n = 0
    for m in maximalist:
        for t in range(lastMax, m):
            phi = 2 * np.pi * float(t - lastMax) / float(m - lastMax)
            phases[n] = phi
            n += 1
        lastMax = m
    phases[-1] = 2 * np.pi
    return phases


@jit(nopython=True)
def _estimate_r(ntraces, times, phases):
    kuramoto = np.empty((times), dtype=np.float64)
    for t in range(times):
        R = 1j*0
        for n in range(ntraces):
            R += np.exp(1j * phases[n, t])
        R /= ntraces
        kuramoto[t] = np.absolute(R)
    return kuramoto
