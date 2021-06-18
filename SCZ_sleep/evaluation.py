import numpy as np
import scipy


from neurolib.utils import functions as func

from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.stimulus import construct_stimulus
import brainplot as bp

def param_search(model, parameters, fname='scz_sleep.hdf'):

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

        # -------- SWS analysis --------
        (normalized_down_lengths, n_local_waves, n_global_waves,
         loca_waves_isi, global_waves_isi) = sws_analysis(model.output, model)

        result = {
            "max_output": max_output,
            "max_amp_output": max_amp_output,
            "domfr": domfr,
            "up_down_difference": up_down_difference,
            "normalized_down_lengths": normalized_down_lengths,
            "normalized_down_lengths_mean": np.mean(normalized_down_lengths),
            "n_local_waves": n_local_waves,
            "n_global_waves": n_global_waves,
            "loca_waves_isi": loca_waves_isi,
            "global_waves_isi": global_waves_isi
        }
        search.saveToPypet(result, traj)
        return

    search = BoxSearch(evalFunction=evaluateSimulation, model=model,
                       parameterSpace=parameters,
                       filename=fname)
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
