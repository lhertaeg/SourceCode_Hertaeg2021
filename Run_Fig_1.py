#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:54:31 2021

@author: loreen.hertaeg
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from Functions_Save import Create_PathFilePickle, Create_PathPlot
from Functions_Plot import Plot_Pathways, Plot_Test_MFN, Plot_PE_MFN, Plot_Inputs_Phases

inch = 2.54

# %% Figure 1

### Define Data structure for storage
fln_plot = 'Figure_1'
fs = 6

### Define figure structure
figsize=(13/inch,4/inch)
fig = plt.figure(figsize=figsize)#, tight_layout=True)

G = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 0.65, 1, 1], wspace=0.7)#, hspace=0.7, height_ratios=[2, 2, 1.7])
B = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=G[:,0], hspace=0.5)
C = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=G[:,1], hspace=0.5)
D = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=G[:,2], hspace=0.0)
E = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=G[:,3])
ax_fig = fig.add_subplot(G[:,:])
ax_B = fig.add_subplot(B[:,:])
ax_C = fig.add_subplot(C[:,:])
ax_D = fig.add_subplot(D[:,:])
ax_E = fig.add_subplot(E[:,:])

ax_B1 = fig.add_subplot(B[0,0])
ax_B2 = fig.add_subplot(B[1,0])
ax_C1 = fig.add_subplot(C[0,0])
ax_C2 = fig.add_subplot(C[1,0])
ax_D1 = fig.add_subplot(D[0,0])
ax_E1 = fig.add_subplot(E[1:4,0])
ax_E2 = fig.add_subplot(E[6:9,0], sharex=ax_E1)
plt.setp(ax_E1.get_xticklabels(), visible=False)

ax_fig.axis('off')
ax_B.axis('off')
ax_C.axis('off')
ax_D.axis('off')
ax_E.axis('off')

ax_B.set_title('E/I balance \nof inputs ', fontsize=fs, pad=10)
ax_C.set_title('E/I balance \nof pathways', fontsize=fs, pad=10)
ax_D1.set_title('Activity example\n', fontsize=fs, pad=10)
ax_E.set_title('Mismatch response scales \nwith diff. between S and P', fontsize=fs, pad=10)


### Load data
folder = 'Balance'
fln_data_10 = 'Data_Optimal_Parameters_MFN_10'
fln_Test_10 = 'Data_Optimal_Parameters_MFN_10_Example'

_, filename_weights_10 = Create_PathFilePickle(folder, fln_data_10)
_, filename_PE_10 = Create_PathFilePickle(folder, 'Data_PE_MFN_10')

with open(filename_PE_10,'rb') as f:
    stim_low_all, RE_MM_10, RE_PB_10 = pickle.load(f)
    
    
### Plot 
Plot_Inputs_Phases(folder, fln_data_10 + '_Inputs', None, fs=fs, anno=True, ax1=ax_B1, ax2=ax_B2)
#Plot_Inputs_Phases(1, 0, filename_weights_10, None, fs=fs, anno=True, ax1=ax_B1, ax2=ax_B2)
Plot_Pathways(1, 0, filename_weights_10, None, fs=fs, ax1=ax_C1, ax2=ax_C2) #, anno=True
Plot_Test_MFN([0, 1, 4, 5, 6, 7], folder, fln_Test_10, None, lab_flag=1, fs=fs, lw=0.8, G=D)
Plot_PE_MFN(stim_low_all, RE_MM_10, RE_PB_10, None, ylab_flag=0, ms=3, lw=1, fs=fs, ax1=ax_E1, ax2=ax_E2)

### Save figure
FigPath = 'Results/Figures/' + folder
plt.savefig(FigPath + '/Fig_1.pdf', bbox_inches='tight', transparent=True, dpi=500)