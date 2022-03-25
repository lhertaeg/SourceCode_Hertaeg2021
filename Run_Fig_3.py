"""
nPE and pPE neurons develop through inhibitory plasticity with a low homeostatic target rate. 
"""


# %% Import 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

from Functions_Save import LoadAnalysisActivity, LoadGeneralisability, LoadNetworkPara, LoadInputsPCs, LoadRobustness
from Functions_Plot import Plot_Generalisability, Plot_PC_Activity_Test, Plot_Inputs_to_PCs, Plot_Robustness
from Run_FullNet import run_static_network_before, run_plastic_network, run_generalise, run_robustness

# %% Figure 3

### Universal parameters
fs = 6
inch = 2.54


### Define path and folder
folder = 'Plasticity'
path = 'Results/Data/' + folder
figPath = 'Results/Figures/' + folder

if not os.path.exists(path):
    os.mkdir(path)
    
if not os.path.exists(figPath):
    os.mkdir(figPath)


### Define figure structure
figsize=(17/inch,14/inch)
fig = plt.figure(figsize=figsize)#, tight_layout=True)

G = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 3], wspace=0.3, hspace=0.5) # , hspace=0.7
G_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G[1,:], height_ratios=[1, 60], wspace=0.3)
Left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G_sub[1,0], wspace=0.6, hspace=0.7)
Right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G_sub[1,1], wspace=0.6, hspace=0.7)

ax_sub = fig.add_subplot(G_sub[0,:])
ax_sub.axis('off')
ax_sub.set_title('After learning', fontsize=fs+1, pad=15)

B = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G[0,1], wspace=0.7, height_ratios=[1, 20])
ax_B = fig.add_subplot(B[0,:])
ax_B.axis('off')
ax_B.set_title('Before learning', fontsize=fs+1, pad=15)
B1 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=B[1,0])
B2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=B[1,1], height_ratios=[2, 2, 1])

C = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=Left[0,:], wspace=0.7, height_ratios=[1, 20])
ax_C = fig.add_subplot(C[0,:])
ax_C.axis('off')
ax_C.set_title('Target: Rate', fontsize=fs+1, pad=15)
C1 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=C[1,0])
C2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=C[1,1], height_ratios=[2, 2, 1])

ax_D = fig.add_subplot(Left[1,0])
ax_D.set_title('Generalizability', fontsize=fs, pad=5)

E = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=Left[1,1])
ax_E = fig.add_subplot(E[0,1])
ax_E.set_title('Robustness', fontsize=fs, pad=5)

F = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=Right[0,:], wspace=0.7, height_ratios=[1, 20])
ax_F = fig.add_subplot(F[0,:])
ax_F.axis('off')
ax_F.set_title('Target: Input', fontsize=fs+1, pad=15)
F1 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=F[1,0])
F2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=F[1,1], height_ratios=[2, 2, 1])

ax_G = fig.add_subplot(Right[1,0], sharey=ax_D)
ax_G.set_title('Generalizability', fontsize=fs, pad=5)

H = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=Right[1,1])
ax_H = fig.add_subplot(H[0,1])
ax_H.set_title('Robustness', fontsize=fs, pad=5)


### Load data if possible or run simulation/analysis, plotting

## Before learning
fln_before = 'Example_Before'
fln_data = 'Data_Inputs2PCs_Network_Before'

if not os.path.isfile(path + '/Data_NetworkParameters_' + fln_before + '.pickle'):
    run_static_network_before(folder, fln_before)
    

NeuPar, NetPar, InPar, StimPar_Test, SimPar_Test, RatePar, _ = LoadNetworkPara(folder, fln_before)

t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_before)
Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, None, G=B1, fs=fs) # Panel = 'B'

t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M = LoadInputsPCs(folder, fln_data)
Plot_Inputs_to_PCs(NeuPar, t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M, None, G=B2, fs=fs)


## After learning (target rate)
fln = 'Target_Rate_After'
fln_exp = 'Example_' + fln
fln_inp_FBBL = 'Data_Inputs2PCs_Network_' + fln
fln_gen = 'Data_Generalisability_Network_' + fln
fln_rob_supp = 'Data_Robustness_Network_' + fln + '_suppress'
fln_rob_stim = 'Data_Robustness_Network_' + fln + '_stimulate'

if not os.path.isfile(path + '/Data_NetworkParameters_' + fln_exp + '.pickle'):
    run_plastic_network(folder, fln, 0)
    
if not os.path.isfile(path + '/' + fln_gen + '.pickle'):
    run_generalise(folder, 0)

if not os.path.isfile(path + '/' + fln_rob_supp + '.pickle'):
    run_robustness(folder, 0)

NeuPar, NetPar, _, _, _, _, LearnPar = LoadNetworkPara(folder, fln_exp) 
stimuli, FB, stim_max_training = LoadGeneralisability(folder, fln_gen)
FB_1, _, E_Ctrl_1 = LoadRobustness(folder, fln_rob_supp)
FB_2, _, E_Ctrl_2 = LoadRobustness(folder, fln_rob_stim)

t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_exp) 
Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, None, fs=fs, G=C1)

t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M = LoadInputsPCs(folder, fln_inp_FBBL)         
Plot_Inputs_to_PCs(NeuPar, t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M, None, fs=fs, G=C2)

Plot_Generalisability(stimuli, FB, stim_max_training,  None, idx=(bool_nPE + bool_pPE), ms=2, fs=fs, axs=ax_D)

Plot_Robustness(FB_1, E_Ctrl_1[:,1], FB_2, E_Ctrl_2[:,1], [-2,2], None, idx=(bool_nPE + bool_pPE), ms=3, 
                range_target=[2,3,4], ylim=[-1,2.2], fs=fs, G=E)


# # After learning (target input) 
fln = 'Target_Input_After'
fln_exp = 'Example_' + fln
fln_inp_FBBL = 'Data_Inputs2PCs_Network_' + fln
fln_gen = 'Data_Generalisability_Network_' + fln
fln_rob_supp = 'Data_Robustness_Network_' + fln + '_suppress'
fln_rob_stim = 'Data_Robustness_Network_' + fln + '_stimulate'

if not os.path.isfile(path + '/Data_NetworkParameters_' + fln_exp + '.pickle'):
    run_plastic_network(folder, fln, 1)
    
if not os.path.isfile(path + '/' + fln_gen + '.pickle'):
    run_generalise(folder, 1)

if not os.path.isfile(path + '/' + fln_rob_supp + '.pickle'):
    run_robustness(folder, 1)

NeuPar, NetPar, _, _, _, _, LearnPar = LoadNetworkPara(folder, fln_exp) 
stimuli, FB, stim_max_training = LoadGeneralisability(folder, fln_gen)
FB_1, _, E_Ctrl_1 = LoadRobustness(folder, fln_rob_supp)
FB_2, _, E_Ctrl_2 = LoadRobustness(folder, fln_rob_stim)

t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_exp)
Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, None, fs=fs, G=F1)

t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M = LoadInputsPCs(folder, fln_inp_FBBL)
Plot_Inputs_to_PCs(NeuPar, t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M, None, fs=fs, G=F2)

Plot_Generalisability(stimuli, FB, stim_max_training,  None, idx=(bool_nPE + bool_pPE), ms=2, fs=fs, axs=ax_G)

Plot_Robustness(FB_1, E_Ctrl_1[:,1], FB_2, E_Ctrl_2[:,1], [-2,2], None, idx=(bool_nPE + bool_pPE), ms=3,
                range_target=[2,3,4], ylim=[-1,2.2], leg_flg=True, fs=fs, G=H)


### Save figure
plt.savefig(figPath + '/Fig_3.png', bbox_inches='tight', transparent=True, dpi=600)

