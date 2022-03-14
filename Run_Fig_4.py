#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:19:53 2021

@author: loreen.hertaeg
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from Functions_Save import LoadAnalysisExperience, LoadNetworkPara, LoadAnalysisActivity, LoadNumbersDepInp
from Functions_Analysis import GenerateExamples_PhasePlane
from Functions_Plot import Plot_Experience_Reliability, Plot_Experience_NumPE, Plot_PC_Activity_Test
from Functions_Plot import Plot_Numbers_DependenceInput, Plot_InputIllustration, Plot_Numbers_DependenceWeights
from Functions_Plot import Plot_WeightPhasePlane

inch = 2.54


# %% Figure 4

flag = 1

if flag==1:

    ### Define Data structure for storage
    folder = 'Experience'
    fs = 6
    
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
    
    ### Load data and Plot
    
    ## Examples for dependence on initial connectivity
    _, _, _, StimPar_Test, SimPar_Test, _, _ = LoadNetworkPara('Plasticity', 'Example_Before')
    
    fln_save = 'DependenceInitialConnectivity_nPE'
    t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
    Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, xlab=False, fs=fs, G=B1)
    
    fln_save = 'DependenceInitialConnectivity_pPE'
    t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
    Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, ylab=False, fs=fs, G=B2)
    
    fln_save = 'DependenceInitialConnectivity_PE'
    t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
    Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, ylab=False, xlab=False, fs=fs, G=B3)
    
    ## Dependenc on inputs
    V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, 'DependenceInput_PV')
    Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C1, xlab_flag=0, 
                                  leg_flag=True, ms=2, fs=fs, title='PV')
    
    V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, 'DependenceInput_SOM')
    Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C2, ylab_flag=0, ms=2, fs=fs, title='SOM')
    
    V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, 'DependenceInput_VIP')
    Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C3, ylab_flag=0, xlab_flag=0, ms=2, fs=fs, title='VIP')
    
    
    ## Save figure
    FigPath = 'Results/Figures/' + folder
    plt.savefig(FigPath + '/Fig_4_new.png', bbox_inches='tight', transparent=True, dpi=600)
    
    plt.close(fig)


# %% ############
# %% Figure 4

flag = 0

if flag==1:

    ### Define Data structure for storage
    folder = 'Experience'
    fln_plot = 'Figure_4'
    fs = 6
    
    figsize=(15/inch,6/inch)
    fig = plt.figure(figsize=figsize)
    
    G = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[5,1,1,1], hspace=1.3)#, hspace=0.7)
    A = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=G[:,0])
    B = gridspec.GridSpecFromSubplotSpec(20, 3, subplot_spec=G[0,1:], wspace=0.5)
    C = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[1,1:], wspace=0.4)
    
    B1 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=B[2:,0])
    B2 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=B[2:,1])
    B3 = gridspec.GridSpecFromSubplotSpec(9, 22, subplot_spec=B[2:,2])
    
    ax_A1 = fig.add_subplot(A[0,0], projection='3d')
    ax_A2 = fig.add_subplot(A[1,0], projection='3d')
    ax_B1 = fig.add_subplot(B[:2,0])
    ax_B2 = fig.add_subplot(B[:2,1])
    ax_B3 = fig.add_subplot(B[:2,2])
    ax_B2.set_title('Initial connectivity determines mismatch responses', fontsize=fs, pad=5)
    ax_C1 = fig.add_subplot(C[0,0])
    ax_C2 = fig.add_subplot(C[0,1], sharey=ax_C1)
    ax_C2.set_title('Distribution of inputs changes ratio of nPE & pPE neurons', fontsize=fs, pad=10) #Distribution of actual and predicted sensory \ninputs changes ratio of nPE and pPE neurons
    plt.setp(ax_C2.get_yticklabels(), visible=False)
    ax_C3 = fig.add_subplot(C[0,2], sharey=ax_C1)
    plt.setp(ax_C3.get_yticklabels(), visible=False)

    ax_B1.axis('off')
    ax_B2.axis('off')
    ax_B3.axis('off')
    
    ### Load data and Plot
    
    ## Weight space
    num, wSV, wVS = 40, 0.6, 0.7
    
    fln_all = ['DependenceInitialConnectivity_nPE_Before', 
               'DependenceInitialConnectivity_pPE_Before', 
               'DependenceInitialConnectivity_PE_Before']
    
    #Examples = GenerateExamples_PhasePlane(folder, fln_all, M=5)
    # Examples represent mean of values (see Run_Experience)
    Examples = np.zeros((1,3,len(fln_all)))
    Examples[0,:,0] = [0.6+0.5,0.75,2]
    Examples[0,:,1] = [0.4+0.5,2,0.75]
    Examples[0,:,2] = [0.4+0.5,1.75,1.25]
    
    VS, VV = 1, 0
    _, _, _ = Plot_WeightPhasePlane(1, 1, 1, VS, VV, wSV, wVS, num, folder, Examples=Examples, ms=2, fs=fs, axs=ax_A1)
    
    VS, VV = 0, 1
    _, _, _ = Plot_WeightPhasePlane(1, 1, 1, VS, VV, wSV, wVS, num, folder, ms=2, fs=fs, axs=ax_A2)
    
    ## Examples for dependence on initial connectivity
    _, _, _, StimPar_Test, SimPar_Test, _, _ = LoadNetworkPara('Plasticity', 'Example_Before')
    
    fln_save = 'DependenceInitialConnectivity_nPE'
    t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
    Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, xlab=False, fs=fs, G=B1)
    ax_B1.plot(0.1,0.5, color='k', marker='s', ms=3)
    ax_B1.set_xlim([0,1])
    
    fln_save = 'DependenceInitialConnectivity_pPE'
    t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
    Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, ylab=False, fs=fs, G=B2)
    ax_B2.plot(0.1,0.5, color='k', marker='o', ms=3)
    ax_B2.set_xlim([0,1])
    
    fln_save = 'DependenceInitialConnectivity_PE'
    t, RE, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_save)
    Plot_PC_Activity_Test(t, RE, bool_nPE, bool_pPE, StimPar_Test, SimPar_Test, folder, fln_save, ylab=False, xlab=False, fs=fs, G=B3)
    ax_B3.plot(0.1,0.5, color='k', marker='v', ms=3)
    ax_B3.set_xlim([0,1])
    
    ## Dependenc on inputs
    V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, 'DependenceInput_PV')
    Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C1, xlab_flag=0, 
                                 leg_flag=True, ms=2, fs=fs, title='PV')
    
    V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, 'DependenceInput_SOM')
    Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C2, ylab_flag=0, ms=2, fs=fs, title='SOM')
    
    V_fraction, Num_nPE, Num_pPE = LoadNumbersDepInp(folder, 'DependenceInput_VIP')
    Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, None, axs=ax_C3, ylab_flag=0, xlab_flag=0, ms=2, fs=fs, title='VIP')
    
    
    ## Save figure
    FigPath = 'Results/Figures/' + folder
    plt.savefig(FigPath + '/Fig_4.png', bbox_inches='tight', transparent=True, dpi=600)
    
    plt.close(fig)