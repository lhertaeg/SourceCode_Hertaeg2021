#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:34:54 2021

@author: loreen.hertaeg
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from Functions_Save import Create_PathFilePickle, Create_PathPlot
from Functions_Plot import Plot_Manipulation_BL_FB, Plot_Manipulation_Space, Plot_Manipulation_Space_Summary
from Functions_Plot import Plot_Manipulation_Space_Scheme, Plot_Colorbar

inch = 2.54

# %% Figure 2

### Define Data structure for storage
folder = 'Manipulation'
fln_plot = 'Figure_2'
fs = 5

### Define figure structure
figsize=(13/inch,8/inch)
fig = plt.figure(figsize=figsize)#, tight_layout=True)

G = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1.5, 1, 1], hspace=0.3)
B = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[0,1:])
G_sub = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G[1:,:], width_ratios=[1.5, 1, 1])
C = gridspec.GridSpecFromSubplotSpec(20, 6, subplot_spec=G_sub[:,0])
D = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G_sub[:,1:]) #wspace=0.5

ax_B1 = fig.add_subplot(B[0,0])
ax_B2 = fig.add_subplot(B[0,1], sharey=ax_B1)
plt.setp(ax_B2.get_yticklabels(), visible=False)
ax_B1.set_title('Target: Excitatory neurons', fontsize=fs, pad=10)
ax_B2.set_title('Target: Inhibitory neurons', fontsize=fs, pad=10)

ax_colorbar = fig.add_subplot(C[1,0:-3])
ax_C = fig.add_subplot(C[7:, :-2])

ax_D11 = fig.add_subplot(D[0,0])
plt.setp(ax_D11.get_xticklabels(), visible=False)
ax_D12 = fig.add_subplot(D[0,1])
plt.setp(ax_D12.get_xticklabels(), visible=False)
plt.setp(ax_D12.get_yticklabels(), visible=False)
ax_D21 = fig.add_subplot(D[1,0])
ax_D22 = fig.add_subplot(D[1,1])
plt.setp(ax_D22.get_yticklabels(), visible=False)

### Load data
_, filename_load_10 = Create_PathFilePickle(folder, 'Data_Manipulation_MFN_10')

with open(filename_load_10,'rb') as f:
    stim_extra_all, Inputs_nPE_10, Inputs_pPE_10 = pickle.load(f)  

### Plot 
Plot_Colorbar(ax_colorbar, 'horizontal', 'Perturbation (1/s)', vmin=-1.5, vmax=1.5)

Plot_Manipulation_BL_FB(None, Inputs_nPE_10, Inputs_pPE_10, stim_extra_all, range_target=np.arange(4), 
                        range_pert=[0,3,5,7,10], ax=ax_B1, ms=3, fs=fs, lw=0.5, x_flag=False)
Plot_Manipulation_BL_FB(None, Inputs_nPE_10, Inputs_pPE_10, stim_extra_all, range_target=np.arange(4,8), 
                        range_pert=[0,3,5,7,10], ax=ax_B2, ms=3, fs=fs, lw=0.5, ylab_flag=0, x_flag=False)

Plot_Manipulation_Space_Summary(None, Inputs_nPE_10, Inputs_pPE_10, stim_extra_all, show=1, ax=ax_D11, ms=2, fs=fs, 
                                lw=0.5, xlab=False, range_target=np.arange(4))
Plot_Manipulation_Space_Summary(None, Inputs_nPE_10, Inputs_pPE_10, stim_extra_all, show=1, ax=ax_D12, ms=2, fs=fs, 
                                lw=0.5, xlab=False, range_target=np.arange(4,8), ylab=False)

Plot_Manipulation_Space_Summary(None, Inputs_nPE_10, Inputs_pPE_10, stim_extra_all, show=-1, ax=ax_D21, ms=2, fs=fs, 
                                lw=0.5, range_target=np.arange(4), xlab=False)
Plot_Manipulation_Space_Summary(None, Inputs_nPE_10, Inputs_pPE_10, stim_extra_all, show=-1, ax=ax_D22, ms=2, fs=fs, 
                                lw=0.5, range_target=np.arange(4,8), ylab=False, xlab=False)

Plot_Manipulation_Space_Scheme(None, ax=ax_C, ms=3, fs=fs, lw=0.5)


### Save figure
FigPath = 'Results/Figures/' + folder
plt.savefig(FigPath + '/Fig_2.pdf', bbox_inches='tight', transparent=True, dpi=500)

plt.close(fig)
