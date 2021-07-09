import os.path as op

import numpy as np

import mne
from mne.datasets import sample
from mne.simulation import simulate_raw, add_noise


def simulate_raw_eeg(aal2_atlas, cortex, model_data):
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, 'subjects')
    subject = 'sample'
    meg_path = op.join(data_path, 'MEG', subject)

    # To simulate sources, we also need a source space. It can be obtained from the
    # forward solution of the sample subject.
    fwd_fname = op.join(meg_path, 'sample_audvis-meg-eeg-oct-6-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)

    vertices = [i['vertno'] for i in fwd['src']]
    verts_aal2 = [[], []]

    aal2_cortex = aal2_atlas.loc[cortex, :].copy()

    for hemi, hemi_n in zip([0, 1], ['L', "R"]):
        fwd_mni = mne.vertex_to_mni(
            vertices[hemi], hemis=hemi, subject='sample',
            subjects_dir=subjects_dir)
        for i in aal2_cortex[aal2_cortex['hemi'] == hemi_n].index:
            node = aal2_cortex.loc[i, ['x.mni', 'y.mni',
                                       'z.mni']].to_numpy(dtype=np.float64)
            dist = np.linalg.norm(fwd_mni - node, ord=2, axis=1)
            aal2_cortex.loc[i, ['vertex']] = vertices[hemi][np.argmin(dist)]

        verts_aal2[hemi] = aal2_cortex.loc[aal2_cortex['hemi'] == hemi_n,
                                           'vertex'].to_numpy(dtype=np.int32)

    node_data = [[], []]
    node_verts = [[], []]

    for i, v in enumerate(verts_aal2):
        node_data[i] = model_data[np.argsort(v), :]
        node_verts[i] = np.sort(v)

    data = np.vstack(node_data)
    data = 1e-7 * data / data.max()  # scaled by 1e+06 to plot in µV

    # Create SourceEstimate object
    stc = mne.SourceEstimate(
        data, node_verts,
        tmin=0, tstep=0.0001, subject='sample')
    raw_fname = op.join(meg_path, 'sample_audvis_raw.fif')
    info = mne.io.read_info(raw_fname)
    info.update(sfreq=10000., bads=[])

    # Simulate raw
    picks = mne.pick_types(info, eeg=True, meg=False, stim=True, exclude=())
    mne.pick_info(info, picks, copy=False)

    snr = 2
    cov = mne.cov.make_ad_hoc_cov(info)
    cov['data'] *= (20. / snr) ** 2

    raw = simulate_raw(info, stc, forward=fwd)
    # add_noise(raw, cov, iir_filter=[4, -4, 0.8], random_state=42)
    return raw, stc


def find_peaks(model, output, min_distance=1000):
    import brainplot as bp
    import scipy

    def filter_peaks(peaks, inv, t_max, t_min=0):
        return [p for p in peaks if (inv[p] > t_min and inv[p] <= t_max)]

    def peak_isi(peaks):
        return np.diff(peaks)/1000*model.params.dt

    model.outputs[model.default_output] = output
    states = bp.detectSWs(model)
    # durations = bp.get_state_lengths(states)
    involvement = bp.get_involvement(states)

    len_states = np.sum(states, axis=1) * model.params.dt / 1000

    normalized_down_lengths = model.params.duration / 1000 - len_states
    # to percent
    normalized_down_lengths = (normalized_down_lengths /
                               (model.params.duration / 1000) * 100)

    filtered_involvement = scipy.ndimage.gaussian_filter1d(involvement, 2000)
    peaks = scipy.signal.find_peaks(
        filtered_involvement, height=0.1, distance=min_distance)[0]

    peaks25 = filter_peaks(peaks, involvement, 0.50, 0.0)
    peaks50 = filter_peaks(peaks, involvement, 0.75, 0.50)
    peaks75 = filter_peaks(peaks, involvement, 1, 0.75)

    return np.array(peaks25), np.array(peaks50 + peaks75)
