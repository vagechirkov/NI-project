import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np

import neurolib.optimize.exploration.explorationUtils as eu
import neurolib.utils.functions as func
from neurolib.models.aln import ALNModel
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.loadData import Dataset
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.utils.stimulus import construct_stimulus

logger = logging.getLogger()
warnings.filterwarnings("ignore")
logger.setLevel(logging.INFO)

plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.style.reload_library()
plt.style.use("seaborn-white")
plt.rcParams['image.cmap'] = 'plasma'


def evaluateSimulation(traj):
    # get the model from the trajectory using `search.getModelFromTraj(traj)`
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

    model_frs, model_pwrs = func.getMeanPowerSpectrum(model.output,
                                                      dt=model.params.dt,
                                                      maxfr=40,
                                                      spectrum_windowsize=10)
    model_frs, model_pwrs = func.getMeanPowerSpectrum(
        model.output[:, up_window], dt=model.params.dt,
        maxfr=40, spectrum_windowsize=5)
    domfr = model_frs[np.argmax(model_pwrs)]

    result = {
        "max_output": max_output,
        "max_amp_output": max_amp_output,
        "domfr": domfr,
        "up_down_difference": up_down_difference
    }
    search.saveToPypet(result, traj)
    return


ds = Dataset("gw")
model = ALNModel(Cmat=ds.Cmats[0], Dmat=ds.Dmats[0])
model.params['dt'] = 0.1
model.params['duration'] = 20 * 1000  # ms

# add custom parameter for downsampling results
# 10 ms sampling steps for saving data, should be multiple of dt
model.params['save_dt'] = 10.0
model.params["tauA"] = 600.0
model.params["sigma_ou"] = 0.0
model.params["b"] = 20.0

model.params["Ke_gl"] = 300.0
model.params["signalV"] = 80.0


parameters = ParameterSpace({"mue_ext_mean": np.linspace(0.0, 4, 51),
                             "mui_ext_mean": np.linspace(0.0, 4, 51),
                             "b": [20.0]
                             })
search = BoxSearch(evalFunction=evaluateSimulation, model=model,
                   parameterSpace=parameters,
                   filename='exploration-8.0-brain.hdf')


search.run()
fname = "/Users/valery/NEUROLIB/neurolib/data/hdf/exploration-8.0-brain.hdf"
search.loadResults(filename=fname, all=False)


plot_key_label = "Max. $r_E$ [Hz]"
eu.plotExplorationResults(search.dfResults,
                          par1=['mue_ext_mean', 'Input to E [nA]'],
                          par2=['mui_ext_mean', 'Input to I [nA]'],
                          by=['b'],
                          plot_key='max_output',
                          plot_clim=[0.0, 80.0],
                          nan_to_zero=False,
                          plot_key_label=plot_key_label,
                          one_figure=True,
                          multiply_axis=0.2,
                          contour=["max_amp_output", "up_down_difference"],
                          contour_color=[['white'], ['springgreen']],
                          contour_levels=[[10], [10]],
                          contour_alpha=[1.0, 1.0],
                          contour_kwargs={0: {"linewidths": (5,)},
                                          1: {"linestyles": "--",
                                              "linewidths": (5,)}},
                          # alpha_mask="relative_amplitude_BOLD",
                          mask_threshold=0.1,
                          mask_alpha=0.2,
                          savename="gw_brain_nap_001.pdf")

