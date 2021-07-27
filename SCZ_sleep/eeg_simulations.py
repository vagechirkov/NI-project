import os.path as op

import numpy as np

import mne
from mne.datasets import sample
from mne.simulation import simulate_raw, add_noise
from neurolib.utils import atlases


def _simulate_raw_eeg(aal2_atlas, cortex, model_data):
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

    # Prepare ts
    data = np.vstack(node_data)
    data -= np.outer(data.mean(axis=1), np.ones(data.shape[1]))
    data = 1e-7 * data   #  / data.max()  # scaled by 1e+07 to plot in µV

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

    raw = simulate_raw(info, stc, forward=fwd, n_jobs=8)
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


def simulate_raw_eeg(fsave="eeg_simulation.fif", minutes=1):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import scipy

    import mne

    # NB: add more subjects in the data folder in neurolib package
    from neurolib.utils.loadData import Dataset
    from neurolib.models.aln import ALNModel

    from neurolib.utils import functions as func

    import brainplot as bp

    from neurolib.utils import atlases
    from nilearn import plotting  
    atlas = atlases.AutomatedAnatomicalParcellation2()
    # AAL2 atlas is taken from here: https://github.com/cwatson/brainGraph
    aal2_atlas = pd.read_csv("SCZ_sleep/aal2_coords.csv")
    coords = aal2_atlas.loc[atlas.cortex, ["x.mni", "y.mni", "z.mni"]].to_numpy()

    ds = Dataset("gw")
    # ds.Cmat = ds.Cmats[3]
    # ds.Dmat = ds.Dmats[3]
    model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)

    # add custom parameter for downsampling results
    # 10 ms sampling steps for saving data, should be multiple of dt
    model.params['save_dt'] = 10.0
    model.params["tauA"] = 600.0
    model.params["sigma_ou"] = 0.0
    model.params["b"] = 20.0

    model.params["Ke_gl"] = 300.0
    model.params["mue_ext_mean"] = 0.2
    model.params["mui_ext_mean"] = 0.1

    # Sleep model from newest evolution October 2020
    model.params["mue_ext_mean"] = 3.3202829454334535
    model.params["mui_ext_mean"] = 3.682451894176651
    model.params["b"] = 3.2021806735984186
    model.params["tauA"] = 4765.3385276559875
    model.params["sigma_ou"] = 0.36802952978628106
    model.params["Ke_gl"] = 265.48075753153

    model.params['dt'] = 0.1
    model.params['duration'] = minutes * 60 * 1000  #ms
    model.params["signalV"] = 80.0

    model.output_vars += ['seem', 'seev', 'mufe']
    # model.run(bold=False, append=True)

    # local_peaks, global_peaks = find_peaks(model, model.output)
    local_peaks = np.load("SCZ_sleep/simultaion_dataset/local_peaks.npy")
    global_peaks = np.load("SCZ_sleep/simultaion_dataset/global_peaks.npy")
    aln_data = np.load("SCZ_sleep/simultaion_dataset/aln_model.npy")

    sims = []
    for part in np.split(aln_data, minutes, axis=1):
        raw_sim, _ = _simulate_raw_eeg(aal2_atlas, atlas.cortex, part)
        sims.append(raw_sim)

    raw = mne.concatenate_raws(sims)

    local_peaks_annot = mne.Annotations(onset=(local_peaks)/10000,  # in seconds
                                        duration=[0.001]*len(local_peaks),  # in seconds, too
                                        description=['LP']*len(local_peaks))
    global_peaks_annot = mne.Annotations(onset=(global_peaks)/10000,  # in seconds
                                         duration=[0.001]*len(global_peaks),  # in seconds, too
                                         description=['GP']*len(global_peaks))
    peaks_annot = local_peaks_annot + global_peaks_annot
    raw.set_annotations(peaks_annot)
    raw = raw.resample(sfreq=200.)
    raw.pick_types(eeg=True)
    raw.save(fsave, overwrite=True)


if __name__ == '__main__':
    simulate_raw_eeg("/SCZ_sleep/simultaion_dataset/eeg_simulation.fif", 10)
