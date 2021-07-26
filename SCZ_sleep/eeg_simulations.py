import os.path as op

import numpy as np

import mne
from mne.datasets import sample
from mne.simulation import simulate_raw, add_noise
from neurolib.utils import atlases


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


def convert_aal2_to_aparca2009():
    return {
        0: 'G_postcentral-lh',  # 'Precentral_L'
        1: 'G_postcentral-rh',  # 'Precentral_R'
        2: 'G_front_sup-lh',  # 'Frontal_Sup_2_L'
        3: 'G_front_sup-rh',  # 'Frontal_Sup_2_R'
        4: 'G_front_middle-lh',  # 'Frontal_Mid_2_L'
        5: 'G_front_middle-rh',  # 'Frontal_Mid_2_R'
        6: 'G_front_inf-Opercular-lh',  # 'Frontal_Inf_Oper_L'
        7: 'G_front_inf-Opercular-rh',  # 'Frontal_Inf_Oper_R'
        8: 'G_front_inf-Triangul-lh',  # 'Frontal_Inf_Tri_L'
        9: 'G_front_inf-Triangul-rh',  # 'Frontal_Inf_Tri_R'
        10: 'G_front_inf-Orbital-lh',  # 'Frontal_Inf_Orb_2_L'
        11: 'G_front_inf-Orbital-rh',  # 'Frontal_Inf_Orb_2_R'
        12: '',  # 'Rolandic_Oper_L'
        13: '',  # 'Rolandic_Oper_R'
        14: '',  # 'Supp_Motor_Area_L'
        15: '',  # 'Supp_Motor_Area_R'
        16: 'S_orbital_med-olfact-lh',  # 'Olfactory_L'
        17: 'S_orbital_med-olfact-rh',  # 'Olfactory_R'
        18: 'S_front_sup-lh',  # 'Frontal_Sup_Medial_L'
        19: 'S_front_sup-rh',  # 'Frontal_Sup_Medial_R'
        20: '',  # 'Frontal_Med_Orb_L'
        21: '',  # 'Frontal_Med_Orb_R'
        22: '',  # 'Rectus_L'
        23: '',  # 'Rectus_R'
        24: 'S_orbital_med-olfact-lh',  # 'OFCmed_L'
        25: 'S_orbital_med-olfact-rh',  # 'OFCmed_R'
        26: '',  # 'OFCant_L'
        27: '',  # 'OFCant_R'
        28: '',  # 'OFCpost_L'
        29: '',  # 'OFCpost_R'
        30: 'S_orbital_lateral-lh',  # 'OFClat_L'
        31: 'S_orbital_lateral-rh',  # 'OFClat_R'
        32: 'G_insular_short-lh',  # 'Insula_L'
        33: 'G_insular_short-rh',  # 'Insula_R'
        34: 'G_cingul-Post-ventral-lh',  # 'Cingulate_Ant_L'
        35: 'G_cingul-Post-ventral-rh',  # 'Cingulate_Ant_R'
        36: '',  # 'Cingulate_Mid_L'
        37: '',  # 'Cingulate_Mid_R'
        38: 'G_cingul-Post-dorsal-lh',  # 'Cingulate_Post_L'
        39: 'G_cingul-Post-dorsal-rh',  # 'Cingulate_Post_R'
        40: 'S_calcarine-lh',  # 'Calcarine_L'
        41: 'S_calcarine-rh',  # 'Calcarine_R'
        42: 'G_cuneus-lh',  # 'Cuneus_L'
        43: 'G_cuneus-rh',  # 'Cuneus_R'
        44: 'G_oc-temp_med-Lingual-lh',  # 'Lingual_L'
        45: 'G_oc-temp_med-Lingual-rh',  # 'Lingual_R'
        46: 'G_occipital_sup-lh',  # 'Occipital_Sup_L'
        47: 'G_occipital_sup-rh',  # 'Occipital_Sup_R'
        48: 'G_occipital_middle-lh',  # 'Occipital_Mid_L'
        49: 'G_occipital_middle-rh',  # 'Occipital_Mid_R'
        50: '',  # 'Occipital_Inf_L'
        51: '',  # 'Occipital_Inf_R'
        52: 'G_oc-temp_lat-fusifor-lh',  # 'Fusiform_L'
        53: 'G_oc-temp_lat-fusifor-rh',  # 'Fusiform_R'
        54: 'S_postcentral-lh',  # 'Postcentral_L'
        55: 'S_postcentral-rh',  # 'Postcentral_R'
        56: 'S_precentral-sup-part-lh',  # 'Parietal_Sup_L'
        57: 'S_precentral-sup-part-rh',  # 'Parietal_Sup_R'
        58: 'S_precentral-inf-part-lh',  # 'Parietal_Inf_L'
        59: 'S_precentral-inf-part-rh',  # 'Parietal_Inf_R'
        60: 'G_pariet_inf-Supramar-lh',  # 'SupraMarginal_L'
        61: 'G_pariet_inf-Supramar-rh',  # 'SupraMarginal_R'
        62: 'G_pariet_inf-Angular-lh',  # 'Angular_L'
        63: 'G_pariet_inf-Angular-rh',  # 'Angular_R'
        64: 'G_precuneus-lh',  # 'Precuneus_L'
        65: 'G_precuneus-rh',  # 'Precuneus_R'
        66: 'G_and_S_paracentral-lh',  # 'Paracentral_Lobule_L'
        67: 'G_and_S_paracentral-rh',  # 'Paracentral_Lobule_R'
        68: 'G_temp_sup-G_T_transv-lh',  # 'Heschl_L'
        69: 'G_temp_sup-G_T_transv-rh',  # 'Heschl_R'
        70: 'S_temporal_sup-lh',  # 'Temporal_Sup_L'
        71: 'S_temporal_sup-rh',  # 'Temporal_Sup_R'
        72: 'G_temp_sup-Plan_polar-lh',  # 'Temporal_Pole_Sup_L'
        73: 'G_temp_sup-Plan_polar-rh',  # 'Temporal_Pole_Sup_R'
        74: 'G_temporal_middle-lh',  # 'Temporal_Mid_L'
        75: 'G_temporal_middle-rh',  # 'Temporal_Mid_R'
        76: 'Pole_temporal-lh',  # 'Temporal_Pole_Mid_L'
        77: 'Pole_temporal-rh',  # 'Temporal_Pole_Mid_R'
        78: 'S_temporal_inf-lh',  # 'Temporal_Inf_L'
        79: 'S_temporal_inf-rh',  # 'Temporal_Inf_R'
    }


if __name__ == '__main__':
    from neurolib.utils.loadData import Dataset
    from neurolib.models.aln import ALNModel
    ds = Dataset("gw")
    # ds.Cmat = ds.Cmats[3]
    # ds.Dmat = ds.Dmats[3]
    model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
    model.output_vars += ['seem', 'seev']
    model.run(append=True)
    
    