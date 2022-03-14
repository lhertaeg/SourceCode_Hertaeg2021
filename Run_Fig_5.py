#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:40:11 2021

@author: loreen.hertaeg
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

#from Functions_Save import Create_PathPlot
from Functions_Save import LoadActivityBiasTowardsMean, LoadActivityPredProcNet, LoadAnalysisActivity, LoadErrorEnvironments, LoadLearningRate
from Functions_Plot import Plot_BiasTowradsMean, Plot_TransitionNewDistribution, Plot_SwitchAttractor, Plot_CoupledEnvironments, Plot_ModulationLearnRates

inch = 2.54

# %% Figure 5

### Define Data structure for storage
folder = 'Behavior'
fln_plot = 'Figure_5'
fs = 6

### Define figure structure
figsize=(8/inch,13/inch)
fig = plt.figure(figsize=figsize)#, tight_layout=True)

G = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1,1,1], wspace=4, hspace=1) #, height_ratios=[2, 1, 1]
#G1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=G[0,:])
#G_sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G1[0,2:], wspace=0.5)

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


### Load and Plot data
stimuli_weak, stimuli_strong, Out_weak, Out_strong = LoadActivityBiasTowardsMean(folder, 'BiasTowardsMeans')
Plot_BiasTowradsMean(stimuli_weak, stimuli_strong, Out_weak, Out_strong, folder, None, axs=ax_B, ms=4, fs=fs)

N_transistion=30
_, _, _, _, _, Out_per_stimulus, StimPar, SimPar = LoadActivityPredProcNet(folder, 'TransitionNewDistribution')
Plot_TransitionNewDistribution(StimPar, Out_per_stimulus, N_transistion, folder, None, axs=ax_C, fs=fs)

t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar, = LoadActivityPredProcNet(folder, 'SwitchAttractor')
_, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity('Plasticity', 'Example_Target_Input_After')
Plot_SwitchAttractor(StimPar, SimPar, t, B, RE, bool_nPE, bool_pPE, stimuli_weak, stimuli_strong, 
                     folder, None, ms = 1.5, axs=ax_D, fs=fs)

Ratio_AB, Error_coupled, Error_uncoupled = LoadErrorEnvironments(folder, 'CoupledEnvironments')
Plot_CoupledEnvironments(Ratio_AB, Error_coupled, Error_uncoupled, folder, None, axs=ax_E, fs=fs)

R_mod, R_classic, dR_mod, dR_classic = LoadLearningRate(folder, 'ModulationLearningRate_4')
Plot_ModulationLearnRates(stimuli_weak, R_mod, R_classic, dR_mod, dR_classic, 500, folder, None, ms=2, axs=ax_F, fs=fs)


### Save figure
FigPath = 'Results/Figures/' + folder
plt.savefig(FigPath + '/Fig_5.pdf', bbox_inches='tight', transparent=True, dpi=500)

plt.close(fig)
