#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:06:21 2021

@author: loreen.hertaeg
"""

#import numpy as np
import os
import pickle

# %% Functions


def Create_PathFilePickle(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/' + fln + '.pickle'
    
    if not os.path.exists(path):
        os.mkdir(path)
        
    return path, filename


def Create_PathPlot(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/' + fln
    
    if not os.path.exists(path):
        os.mkdir(path)
        
    return path, filename


def SaveNetworkPara(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, LearnPar, folder, fln):
    
    path = 'Results/Data/' + folder
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    filename = path + '/Data_NetworkParameters_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
        pickle.dump([NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, LearnPar],f) 
        
        
def LoadNetworkPara(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/Data_NetworkParameters_' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, LearnPar = pickle.load(f)
        
    return NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, LearnPar


def SaveAnalysisActivity(t, RE, RP, RS, RV, bool_nPE, bool_pPE, folder, fln):

    path = 'Results/Data/' + folder
    filename = path + '/Activity_relative_to_BL_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
        pickle.dump([t, RE, RP, RS, RV, bool_nPE, bool_pPE],f)
            
            
def LoadAnalysisActivity(folder, fln, dir_path = None) :         
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
    filename = path + '/Activity_relative_to_BL_' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        t, RE, RP, RS, RV, bool_nPE, bool_pPE = pickle.load(f)
            
    return t, RE, RP, RS, RV, bool_nPE, bool_pPE


def SaveInputsPCs(t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'wb') as f:
        pickle.dump([t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M],f)
            
            
def LoadInputsPCs(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M = pickle.load(f)
    
    return t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M


def SaveGeneralisability(stimuli, FB, stim_max_training, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([stimuli, FB, stim_max_training],f)
            
            
def LoadGeneralisability(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        stimuli, FB, stim_max_training = pickle.load(f)
        
    return stimuli, FB, stim_max_training


def SaveRobustness(FB, BL, E_Ctrl, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([FB, BL, E_Ctrl],f)
            
            
def LoadRobustness(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        FB, BL, E_Ctrl = pickle.load(f)
        
    return FB, BL, E_Ctrl


def SaveAnalysisExperience(p_out, Mean_Phases_nPE, Mean_Phases_pPE, SEM_Phases_nPE, 
                               SEM_Phases_pPE, Num_nPE, Num_pPE, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([p_out, Mean_Phases_nPE, Mean_Phases_pPE, SEM_Phases_nPE, SEM_Phases_pPE, Num_nPE, Num_pPE],f)
       
            
def LoadAnalysisExperience(folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        p_out, Mean_Phases_nPE, Mean_Phases_pPE, SEM_Phases_nPE, SEM_Phases_pPE, Num_nPE, Num_pPE = pickle.load(f)
        
    return p_out, Mean_Phases_nPE, Mean_Phases_pPE, SEM_Phases_nPE, SEM_Phases_pPE, Num_nPE, Num_pPE


def SaveInputs_MM_PB(Inp_MM, Inp_PB, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_Inputs_MM_PB_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([Inp_MM, Inp_PB],f)
            
            
def LoadInputs_MM_PB(folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        Inp_MM, Inp_PB = pickle.load(f)
        
    return Inp_MM, Inp_PB


def SaveActivityPredProcNet(t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar],f)


def LoadActivityPredProcNet(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar = pickle.load(f)
        
    return t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar


def SaveActivityBiasTowardsMean(stimuli_weak, stimuli_strong, Out_weak, Out_strong, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([stimuli_weak, stimuli_strong, Out_weak, Out_strong],f)


def LoadActivityBiasTowardsMean(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        stimuli_weak, stimuli_strong, Out_weak, Out_strong = pickle.load(f)
        
    return stimuli_weak, stimuli_strong, Out_weak, Out_strong


def SaveErrorEnvironments(Ratio_AB, Error_coupled, Error_uncoupled, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([Ratio_AB, Error_coupled, Error_uncoupled],f)


def LoadErrorEnvironments(folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        Ratio_AB, Error_coupled, Error_uncoupled = pickle.load(f)
        
    return Ratio_AB, Error_coupled, Error_uncoupled


def SaveLearningRate(R_mod, R_classic, dR_mod, dR_classic, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([R_mod, R_classic, dR_mod, dR_classic],f)
            
            
            
def SaveNumPE(para, Num_nPE, Num_pPE, folder, fln):
        
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([para, Num_nPE, Num_pPE],f)
            
            
def LoadNumPE(folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        para, Num_nPE, Num_pPE = pickle.load(f)
        
    return para, Num_nPE, Num_pPE
            

def LoadLearningRate(folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        R_mod, R_classic, dR_mod, dR_classic = pickle.load(f)
        
    return R_mod, R_classic, dR_mod, dR_classic


def SaveNumbersDepInp(V_fraction, Num_nPE, Num_pPE, folder, fln):
    
    path = 'Results/Data/' + folder
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'wb') as f:
            pickle.dump([V_fraction, Num_nPE, Num_pPE],f)
            

def LoadNumbersDepInp(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
        
    filename = path + '/Data_' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        V_fraction, Num_nPE, Num_pPE = pickle.load(f)
        
    return V_fraction, Num_nPE, Num_pPE


def SaveInput2PC_MFN(Weights, stim_duration, stim_high, E_nE_phases, I_nE_phases, E_pE_phases, 
                     I_pE_phases, E_nD_phases, I_nD_phases, E_pD_phases, I_pD_phases, folder, fln):
    
    path = 'Results/Data/' + folder
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'wb') as f:
        pickle.dump([Weights, stim_duration, stim_high, E_nE_phases, I_nE_phases, E_pE_phases, 
                     I_pE_phases, E_nD_phases, I_nD_phases, E_pD_phases, I_pD_phases],f) 
        
        
def LoadInput2PC_MFN(folder, fln, dir_path = None):
    
    if dir_path is None:
        path = 'Results/Data/' + folder
    else:
        path = dir_path + 'Results/Data/' + folder
    
    filename = path + '/' + fln + '.pickle'
    
    with open(filename,'rb') as f:
        Weights, stim_duration, stim_high, E_nE_phases, I_nE_phases, E_pE_phases, I_pE_phases, E_nD_phases, I_nD_phases, E_pD_phases, I_pD_phases = pickle.load(f)
        
    return Weights, stim_duration, stim_high, E_nE_phases, I_nE_phases, E_pE_phases, I_pE_phases, E_nD_phases, I_nD_phases, E_pD_phases, I_pD_phases


