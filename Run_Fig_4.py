"""
Initial connectivity and distribution of inputs onto interneurons determine mismatch responses of PE neurons. 
"""

# %% Import 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

from Functions_Save import LoadNetworkPara, LoadAnalysisActivity, LoadNumbersDepInp
from Functions_Plot import Plot_PC_Activity_Test, Plot_Numbers_DependenceInput
from Run_FullNet import run_input_distribution, run_initial_weights

# %% Figure 4


### Universal parameters
fs = 6
inch = 2.54

### Define path and folder
folder = 'Experience'
path = 'Results/Data/' + folder
figPath = 'Results/Figures/' + folder

if not os.path.exists(path):
    os.mkdir(path)
    
if not os.path.exists(figPath):
    os.mkdir(figPath)

### Define figure structure
figsize=(8/inch,7/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.2,1], hspace=1.0)#, hspace=0.7)
B = gridspec.GridSpecFromSubplotSpec(20, 3, subplot_spec=G[0,:], wspace=0.5)
C = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[1,:], wspace=0.4)

B1 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=B[2:,0])
B2 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=B[2:,1])
B3 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=B[2:,2])

ax_B1 = fig.add_subplot(B[:2,0])
ax_B2 = fig.add_subplot(B[:2,1])
ax_B3 = fig.add_subplot(B[:2,2])
ax_B2.set_title('Initial connectivity determines mismatch responses', fontsize=fs, pad=1)
ax_C1 = fig.add_subplot(C[0,0])
ax_C2 = fig.add_subplot(C[0,1], sharey=ax_C1)
ax_C2.set_title('Distribution of inputs changes ratio of nPE & pPE neurons', fontsize=fs, pad=10) #Distribution of actual and predicted sensory \ninputs changes ratio of nPE and pPE neurons
plt.setp(ax_C2.get_yticklabels(), visible=False)
ax_C3 = fig.add_subplot(C[0,2], sharey=ax_C1)
plt.setp(ax_C3.get_yticklabels(), visible=False)

ax_B1.axis('off')
ax_B2.axis('off')
ax_B3.axis('off')

### Load data if possible or run simulation/analysis, plotting

## Examples for dependence on initial connectivity
_, _, _, StimPar_Test, SimPar_Test, _, _ = LoadNetworkPara('Plasticity', 'Example_Before')

fln_save = 'DependenceInitialConnectivity_nPE'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_initial_weights('Plasticity', folder, 0)
    
t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, xlab=False, fs=fs, G=B1)

fln_save = 'DependenceInitialConnectivity_pPE'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_initial_weights('Plasticity', folder, 1)
    
t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, ylab=False, fs=fs, G=B2)

fln_save = 'DependenceInitialConnectivity_PE'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_initial_weights('Plasticity', folder, 2)
    
t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, ylab=False, xlab=False, fs=fs, G=B3)

## Dependenc on inputs
fln_save = 'DependenceInput_PV'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_input_distribution('Plasticity', folder, 0)

V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, fln_save)
Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C1, xlab_flag=0, leg_flag=True, ms=2, fs=fs, title='PV')

fln_save = 'DependenceInput_SOM'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_input_distribution('Plasticity', folder, 1)

V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, fln_save)
Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C2, ylab_flag=0, ms=2, fs=fs, title='SOM')

fln_save = 'DependenceInput_VIP'
if not os.path.isfile(path + '/Data_' + fln_save + '.pickle'):
    run_input_distribution('Plasticity', folder, 2)

V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, fln_save)
Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C3, ylab_flag=0, xlab_flag=0, ms=2, fs=fs, title='VIP')


## Save figure
plt.savefig(figPath + '/Fig_4.png', bbox_inches='tight', transparent=True, dpi=600)

