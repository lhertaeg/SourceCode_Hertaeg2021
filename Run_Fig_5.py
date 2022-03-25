"""
The role of PE neurons in biased perception.
"""

# %% Import

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

from Functions_Save import LoadActivityBiasTowardsMean, LoadActivityPredProcNet, LoadAnalysisActivity, LoadErrorEnvironments, LoadLearningRate
from Functions_Plot import Plot_BiasTowradsMean, Plot_TransitionNewDistribution, Plot_SwitchAttractor, Plot_CoupledEnvironments, Plot_ModulationLearnRates
from Run_PPE import run_contraction_bias, run_vanishing_bias, run_switching_attarctors, run_generalise, run_modulate_leaning


# %% Figure 5

### Universal parameters
fs = 6
inch = 2.54

### Define path and folder
folder = 'Behavior'
path = 'Results/Data/' + folder
figPath = 'Results/Figures/' + folder

if not os.path.exists(path):
    os.mkdir(path)
    
if not os.path.exists(figPath):
    os.mkdir(figPath)

### Define figure structure
figsize=(8/inch,13/inch)
fig = plt.figure(figsize=figsize)#, tight_layout=True)

G = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1,1,1], wspace=4, hspace=1) 

ax_B = fig.add_subplot(G[0,:2])
ax_B.set_title('Contraction bias\n', fontsize=fs, pad=5)
ax_C = fig.add_subplot(G[0,2:])
ax_C.set_title('Bias vanishes for \npredictable stimulus', fontsize=fs, pad=5)
ax_D = fig.add_subplot(G[1,:3])
ax_D.set_title('PE as internal cue to detect distribution shift', fontsize=fs, pad=5)
ax_E = fig.add_subplot(G[2,:2])
ax_E.set_title('Generalization through \nprior knowledge', fontsize=fs, pad=5)
ax_F = fig.add_subplot(G[2,2:])
ax_F.set_title('Faster learning through \nlearning rate modulation', fontsize=fs, pad=5)


### Load data if possible or run simulation/analysis, plotting

fln_save = 'BiasTowardsMeans'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_contraction_bias('Plasticity', folder, fln_save)
    
stimuli_weak, stimuli_strong, Out_weak, Out_strong = LoadActivityBiasTowardsMean(folder, fln_save)
Plot_BiasTowradsMean(stimuli_weak, stimuli_strong, Out_weak, Out_strong, folder, None, axs=ax_B, ms=4, fs=fs)

fln_save = 'TransitionNewDistribution'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_vanishing_bias('Plasticity', folder, fln_save)

_, _, _, _, _, Out_per_stimulus, StimPar, SimPar = LoadActivityPredProcNet(folder, fln_save)
Plot_TransitionNewDistribution(StimPar, Out_per_stimulus, 30, folder, None, axs=ax_C, fs=fs)

fln_save = 'SwitchAttractor'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_switching_attarctors('Plasticity', folder, fln_save)

t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar, = LoadActivityPredProcNet(folder, fln_save)
_, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity('Plasticity', 'Example_Target_Input_After')
Plot_SwitchAttractor(StimPar, SimPar, t, B, RE, bool_nPE, bool_pPE, stimuli_weak, stimuli_strong, folder, None, ms = 1.5, axs=ax_D, fs=fs)

fln_save = 'CoupledEnvironments'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_generalise('Plasticity', folder, fln_save)

Ratio_AB, Error_coupled, Error_uncoupled = LoadErrorEnvironments(folder, fln_save)
Plot_CoupledEnvironments(Ratio_AB, Error_coupled, Error_uncoupled, folder, None, axs=ax_E, fs=fs)

fln_save = 'ModulationLearningRate'
if not os.path.isfile(path + '/Data_' + fln_save + '_4.pickle'):
    run_modulate_leaning('Plasticity', folder, fln_save)

R_mod, R_classic, dR_mod, dR_classic = LoadLearningRate(folder, fln_save + '_4')
Plot_ModulationLearnRates(stimuli_weak, R_mod, R_classic, dR_mod, dR_classic, 500, folder, None, ms=2, axs=ax_F, fs=fs)


### Save figure
plt.savefig(figPath + '/Fig_5.pdf', bbox_inches='tight', transparent=True, dpi=500)
