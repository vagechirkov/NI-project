import numpy as np
import scipy


from neurolib.utils import functions as func

from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.stimulus import construct_stimulus
import brainplot as bp

# https://doi.org/10.1016/j.neuroimage.2015.07.075 Table 2
# number corresponds to AAL2 labels indices
CORTICAL_REGIONS = {
    'central_region': [1, 2, 61, 62, 13, 14],
    'frontal_lobe': [3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 15, 16, 73, 74,
                     11, 12, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                     29, 30, 31, 32],
    'temporal_lobe': {
        'Lateral surface': [83, 84, 85, 86, 89, 90, 93, 94]
        },
    'parietal_lobe': {
        'Lateral surface': [63, 64, 65, 66, 67, 68, 69, 70],
        'Medial surface': [71, 72],
        },
    'occipital_lobe': {
        'Lateral surface': [53, 54, 55, 56, 57, 58],
        'Medial and inferior surfaces': [47, 48, 49, 50, 51, 52, 59, 60],
        },
    'limbic_lobe': [87, 88, 91, 92, 35, 36, 37, 38, 39, 40, 33, 34]
    }


def param_search(model, parameters, fname='scz_sleep.hdf', run=True):

    def evaluateSimulation(traj):
        model = search.getModelFromTraj(traj)
        # initiate the model with random initial contitions
        model.randomICs()
        defaultDuration = model.params['duration']

        # -------- stage wise simulation --------

        # Stage 3: full and final simulation
        # ---------------------------------------
        model.params['duration'] = defaultDuration

        rect_stimulus = construct_stimulus(stim="rect",
                                           duration=model.params.duration,
                                           dt=model.params.dt)
        model.params['ext_exc_current'] = rect_stimulus * 5.0

        model.run()

        # up down difference
        state_length = 2000
        # last_state = (model.t > defaultDuration - state_length)
        # time period in ms where we expect the down-state
        down_window = ((defaultDuration/2 - state_length < model.t)
                       & (model.t < defaultDuration/2))
        # and up state
        up_window = ((defaultDuration - state_length < model.t)
                     & (model.t < defaultDuration))
        up_state_rate = np.mean(model.output[:, up_window], axis=1)
        down_state_rate = np.mean(model.output[:, down_window], axis=1)
        up_down_difference = np.max(up_state_rate - down_state_rate)

        # check rates!
        max_amp_output = np.max(
            np.max(model.output[:, up_window], axis=1)
            - np.min(model.output[:, up_window], axis=1)
        )
        max_output = np.max(model.output[:, up_window])

        model_frs, model_pwrs = func.getMeanPowerSpectrum(
            model.output, dt=model.params.dt, maxfr=40,
            spectrum_windowsize=10)
        model_frs, model_pwrs = func.getMeanPowerSpectrum(
            model.output[:, up_window], dt=model.params.dt,
            maxfr=40, spectrum_windowsize=5)
        domfr = model_frs[np.argmax(model_pwrs)]

        # -------- SWS analysis all nodes --------
        (normalized_down_lengths, n_local_waves, n_global_waves,
         loca_waves_isi, global_waves_isi) = sws_analysis(model.output, model)

        # -------- SWS analysis frontal nodes --------
        frontal_lobe_nodes = [i - 1 for i in CORTICAL_REGIONS["frontal_lobe"]]
        (frontal_normalized_down_lengths, frontal_n_local_waves,
         frontal_n_global_waves,
         frontal_loca_waves_isi,
         frontal_global_waves_isi) = sws_analysis(
             model.output[frontal_lobe_nodes, :], model)

        result = {
            "max_output": max_output,
            "max_amp_output": max_amp_output,
            "domfr": domfr,
            "up_down_difference": up_down_difference,
            "normalized_down_lengths": normalized_down_lengths,
            "normalized_down_lengths_mean": np.mean(normalized_down_lengths),
            "normalized_up_lengths_mean":
                100 - np.mean(normalized_down_lengths),
            "n_local_waves": n_local_waves,
            "perc_local_waves": (n_local_waves * 100
                                 / (n_local_waves + n_global_waves + 1)),
            "n_global_waves": n_global_waves,
            "all_SWS": n_local_waves+n_global_waves,
            "SWS_per_min": (n_local_waves+n_global_waves) * 3,
            "local_waves_isi": np.mean(loca_waves_isi),
            "global_waves_isi": np.mean(global_waves_isi),
            "frontal_normalized_down_lengths":
                frontal_normalized_down_lengths,
            "frontal_normalized_down_lengths_mean":
                np.mean(frontal_normalized_down_lengths),
            "frontalnormalized_up_lengths_mean":
                100 - np.mean(frontal_normalized_down_lengths),
            "frontal_n_local_waves": frontal_n_local_waves,
            "frontal_perc_local_waves":
                (frontal_n_local_waves * 100 /
                 (frontal_n_local_waves + frontal_n_global_waves + 1)),
            "frontal_all_SWS": frontal_n_local_waves + frontal_n_global_waves,
            "frontal_SWS_per_min": (frontal_n_local_waves +
                                    frontal_n_global_waves) * 3,
            "frontal_n_global_waves": frontal_n_global_waves,
            "frontal_local_waves_isi": np.mean(frontal_loca_waves_isi),
            "frontal_global_waves_isi": np.mean(frontal_global_waves_isi)
        }
        search.saveToPypet(result, traj)
        return

    search = BoxSearch(evalFunction=evaluateSimulation, model=model,
                       parameterSpace=parameters,
                       filename=fname)
    if run:
        search.run()
    return search


def sws_analysis(output, model, min_distance=1000):
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
    n_local_waves = len(peaks25)
    peaks50 = filter_peaks(peaks, involvement, 0.75, 0.50)
    peaks75 = filter_peaks(peaks, involvement, 1, 0.75)
    n_global_waves = len(peaks50 + peaks75)

    # ups, downs = bp.get_state_durations_flat(model, states)

    loca_waves_isi = peak_isi(peaks25)
    global_waves_isi = peak_isi(np.sort(peaks50 + peaks75).tolist())

    return (normalized_down_lengths,  # list one value/node
            n_local_waves, n_global_waves, loca_waves_isi,
            global_waves_isi)
