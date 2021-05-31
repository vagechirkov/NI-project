"""
Some code taken from https://github.com/caglarcakan/sleeping_brain

Copyright (c) 2020 Caglar Cakan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
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
from connectivity.coonectivity_dynamics import fast_kuramoto


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

    model.run(bold=True)

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

    # Kuramoto
    # kur = func.kuramoto(model.rates_exc)
    kur = fast_kuramoto(model.rates_exc)
    if isinstance(kur, int):
        kur_mean = 0
        kur_std = 0
    else:
        kur_mean = kur.mean()
        kur_std = kur.std()

    # FC
    fc = func.fc(model.BOLD.BOLD[:, 5:])

    result = {
        "max_output": max_output,
        "max_amp_output": max_amp_output,
        "domfr": domfr,
        "up_down_difference": up_down_difference,
        "mean_kuramoto": kur_mean,
        "std_kuramoto": kur_std,
        "BOLD_FC": fc
    }
    search.saveToPypet(result, traj)
    return


ds = Dataset("gw")
model = ALNModel(Cmat=ds.Cmats[0], Dmat=ds.Dmats[0])
model.params['dt'] = 0.1
model.params['duration'] = 2 * 60 * 1000  # ms

# add custom parameter for downsampling results
# 10 ms sampling steps for saving data, should be multiple of dt
model.params['save_dt'] = 10.0
model.params["tauA"] = 600.0
model.params["sigma_ou"] = 0.0
model.params["b"] = 20.0

model.params["Ke_gl"] = 300.0
model.params["signalV"] = 80.0

# Sleep model from newest evolution October 2020
model.params["b"] = 3.2021806735984186
model.params["tauA"] = 4765.3385276559875
model.params["sigma_ou"] = 0.36802952978628106
model.params["Ke_gl"] = 265.48075753153
model.params["b"] += 0.5


parameters = ParameterSpace({"mue_ext_mean": np.linspace(0.0, 4, 2),
                             "mui_ext_mean": np.linspace(0.0, 4, 2)
                             })
search = BoxSearch(evalFunction=evaluateSimulation, model=model,
                   parameterSpace=parameters,
                   filename='exploration-8.0-brain.hdf')


search.run()
fname = "/Users/valery/NEUROLIB/NI-project/data/hdf/exploration-8.0-brain.hdf"
search.loadResults(filename=fname, all=False)


plot_key_label = "Max. $r_E$ [Hz]"
eu.plotExplorationResults(search.dfResults,
                          par1=['mue_ext_mean', 'Input to E [nA]'],
                          par2=['mui_ext_mean', 'Input to I [nA]'],
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
