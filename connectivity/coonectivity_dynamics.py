"""
Some code taken from https://github.com/caglarcakan/stimulus_neural_populations
Copyright (c) 2019, Caglar Cakan BSD 2-Clause License
"""

from scipy import ndimage, signal
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


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


def fast_kuramoto(traces, smoothing=2.0, dt=0.1, debug=False):
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


def evaluate_model(model, cmat, path, fname, fc_real=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import mne
    from fastdtw import fastdtw
    from neurolib.utils import functions as func
    import brainplot as bp
    import xarray as xr
    from neurolib.utils.signal import Signal
    from connectivity import make_graph, graph_measures

    plt.figure(figsize=(15, 5))
    plt.imshow(
                model.output, aspect="auto", extent=[0, model.t[-1] / 1000,
                                                     model.params.N, 0],
                clim=(0, 20), cmap="viridis",
            )
    cbar = plt.colorbar(extend='max', fraction=0.046, pad=0.04)
    cbar.set_label("Rate $r_{exc}$ [Hz]")
    plt.ylabel("Node")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(path / f"{fname}_ts.png", dpi=100)
    plt.close()

    data = model.rates_exc

    # ----------------- PLV ----------------- #
    con, _, _, _, _ = mne.connectivity.spectral_connectivity(
        np.split(data, 12, axis=1), method='plv',
        sfreq=10000, fmin=(0, 4, 8, 13, 30),
        fmax=(4, 8, 12, 30, 70), faverage=True
        )

    fig, ax = plt.subplots(1, 5, figsize=(20, 10), sharey=True)

    all_freq_bands = ("0-4Hz", "4-8Hz", "8-12Hz", "13-30Hz", "30-70Hz")
    for i, (_ax, freq_label) in enumerate(zip(ax, all_freq_bands)):
        im = _ax.imshow(con[:, :, i], clim=(0, 1), cmap="viridis")
        _ax.set_title(f'Frequency band: {freq_label}')
        if i == 0:
            _ax.set_ylabel("Node")
        _ax.set_xlabel("Node")
        plt.tight_layout()

    divider = make_axes_locatable(_ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Phase-Locking Value")
    plt.tight_layout()
    plt.savefig(path / f"{fname}_plv_freq_bands.png", dpi=100)
    plt.close()

    # ----------------- DTW ----------------- #
    dtw = np.zeros((80, 80))

    for i, j in zip(np.tril_indices(80)[0], np.tril_indices(80)[1]):
        distance, _ = fastdtw(data[i, ::100][:10_000], data[j, ::100][:10_000])
        dtw[i, j] = distance
    dtw_norm = 1 - dtw/dtw.max()
    dtw_norm[np.triu_indices(80)] = dtw_norm.T[np.triu_indices(80)]

    plt.figure(figsize=(6, 6))
    plt.imshow(dtw_norm, clim=(0, 1), cmap="viridis")
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label("Absolute distance")
    plt.ylabel("Node")
    plt.xlabel("Node")
    plt.title("Dynamic Time Warping")
    plt.tight_layout()
    plt.savefig(path / f"{fname}_dtw_norm.png", dpi=100)
    plt.close()

    # ----------------- FC ----------------- #

    plt.figure(figsize=(6, 6))
    plt.imshow(func.fc(model.BOLD.BOLD[:, 10:]), clim=(0, 1), cmap="viridis")
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    # cbar.set_label("Absolute distance")
    plt.ylabel("Node")
    plt.xlabel("Node")
    plt.title("BOLD FC")
    plt.tight_layout()
    plt.savefig(path / f"{fname}_bold_fc.png", dpi=100)
    plt.close()

    if not isinstance(fc_real, bool):
        plt.figure(figsize=(6, 6))
        plt.imshow(fc_real, clim=(0, 1), cmap="viridis")
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        # cbar.set_label("Absolute distance")
        plt.ylabel("Node")
        plt.xlabel("Node")
        plt.title("BOLD FC real data")
        plt.tight_layout()
        plt.savefig(path / f"{fname}_bold_fc_real.png", dpi=100)
        plt.close()

    # ----------------- Involvment ----------------- #
    states = bp.detectSWs(model, filter_long=True)

    # involvement =  1 - np.sum(states, axis=0) / states.shape[0]
    involvement = bp.get_involvement(states)

    inv_xr = xr.DataArray(involvement, coords=[model.t], dims=["time"])
    sig = Signal(inv_xr, time_in_ms=True)

    def get_phase(signal, filter_args, pad=None):
        """
        Extract phase of the signal. Steps: detrend -> pad -> filter -> Hilbert
        transform -> get phase -> un-pad.
        :param signal: signal to get phase from
        :type signal: `neuro_signal.Signal`
        :param filter_args: arguments for `Signal`'s filter method (see its
            docstring)
        :type filter_args: dict
        :param pad: how many seconds to pad, if None, won't pad
        :type pad: float|None
        :return: wrapped Hilbert phase of the signal
        :rtype: `neuro_signal.Signal`
        """
        assert isinstance(signal, Signal)
        phase = signal.detrend(inplace=False)
        if pad:
            phase.pad(
                how_much=pad, in_seconds=True,
                padding_type="reflect", side="both"
            )
        phase.filter(**filter_args)
        phase.hilbert_transform(return_as="phase_wrapped")
        if pad:
            phase.sel([phase.start_time + pad, phase.end_time - pad])
        return phase

    phase = get_phase(sig, filter_args={"low_freq": 0.5, "high_freq": 2})
    (node_mean_phases_down,
     node_mean_phases_up) = bp.get_transition_phases(states, phase.data)
    node_mean_phases_down = np.array(node_mean_phases_down)
    node_mean_phases_up = np.array(node_mean_phases_up)

    if np.any(node_mean_phases_up < 0) or np.any(node_mean_phases_down > 0):
        print("Modulo was necessary")
    node_mean_phases_up = np.mod(node_mean_phases_up, np.pi)
    node_mean_phases_down = np.mod(node_mean_phases_down, -np.pi)

    # mean_states = np.mean(states, axis=1)  # * 1000
    len_states = np.sum(states, axis=1) * model.params.dt / 1000

    normalized_down_lengths = model.params.duration / 1000 - len_states
    # to percent
    normalized_down_lengths = (normalized_down_lengths /
                               (model.params.duration / 1000) * 100)
    normalized_down_lengths = 1 - normalized_down_lengths/100

    # ----------------- Correlations ----------------- #
    columns = ['mean_degree', 'degree', 'closeness', 'betweenness',
               'mean_shortest_path', 'neighbor_degree', 'neighbor_degree_new',
               'clustering_coefficient', 'omega', 'sigma',
               'mean_clustering_coefficient', 'backbone', 'Cmat', 'Dmat']
    subset_cols = ['degree', 'closeness', 'betweenness', 'neighbor_degree_new',
                   'clustering_coefficient']
    df = pd.DataFrame(columns=columns)

    G = make_graph(cmat)
    G, gm = graph_measures(G)  # , dmat
    df.loc[0] = gm

    results_sc = pd.DataFrame(df.loc[0, subset_cols].to_dict())
    results_sc.columns = ['degree_sc', 'closeness_sc', 'betweenness_sc',
                          'neighbor_sc', 'clustering_sc']
    df = pd.DataFrame(columns=columns)

    fc = func.fc(model.BOLD.BOLD[:, 10:])
    fc[fc < 0] = 0
    G = make_graph(fc)
    G, gm = graph_measures(G)  # , dmat
    df.loc[0] = gm
    results_fc = pd.DataFrame(df.loc[0, subset_cols].to_dict())
    results_fc.columns = ['degree_fc', 'closeness_fc', 'betweenness_fc',
                          'neighbor_fc', 'clustering_fc']

    results = pd.concat([results_sc, results_fc], axis=1)
    for i, freq in enumerate(all_freq_bands):
        G = make_graph(con[:, :, i])
        G, gm = graph_measures(G)  # , dmat
        df = pd.DataFrame(columns=columns)
        df.loc[0] = gm
        results_plv = pd.DataFrame(df.loc[0, ['degree']].to_dict())
        results_plv.columns = [f'degree_plv_{freq}']
        results = pd.concat([results, results_plv], axis=1)

    G = make_graph(dtw_norm)
    G, gm = graph_measures(G)  # , dmat
    df = pd.DataFrame(columns=columns)
    df.loc[0] = gm
    results_dtw = pd.DataFrame(df.loc[0, ['degree']].to_dict())
    results_dtw.columns = ['degree_dtw']
    results = pd.concat([results, results_dtw], axis=1)

    if not isinstance(fc_real, bool):
        G = make_graph(fc_real)
        G, gm = graph_measures(G)  # , dmat
        df = pd.DataFrame(columns=columns)
        df.loc[0] = gm
        results_fc_real = pd.DataFrame(df.loc[0, subset_cols].to_dict())
        results_fc_real.columns = ['degree_fc_real', 'closeness_fc_real',
                                   'betweenness_fc_real', 'neighbor_fc_real',
                                   'clustering_fc_real']
        results = pd.concat([results, results_fc_real], axis=1)

    results.loc[:, 'time_up'] = normalized_down_lengths
    results.loc[:, 'phases_up'] = node_mean_phases_up
    results.loc[:, 'phases_down'] = node_mean_phases_down

    corr = results.corr()
    corr[pd.isna(corr)] = 0
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(20, 18))
        ax = sns.heatmap(corr, cmap=cmap,  # mask=mask,
                         vmax=1., vmin=-1.,
                         square=True, annot=True)
        # plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(path / f"{fname}_correlations.png")
    plt.close()

    sns.clustermap(corr, cmap=cmap, vmax=1., vmin=-1.)
    plt.tight_layout()
    plt.savefig(path / f"{fname}_correlations_clustes.png")
    plt.close()
    return corr
