#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 14 2021

@author: loreen.hertaeg
"""

# %% import packages

import numpy as np
from scipy import optimize
# from scipy import interpolate
import pickle

from Functions_Save import LoadNetworkPara

dtype = np.float32

# %% Functions

def GenerateExamples_PhasePlane(folder, fln_all, M=20):
    
    Examples = np.zeros((M,3,len(fln_all)))
    
    for i in range(len(fln_all)):
        
        _, NetPar, _, _, _, _, _ = LoadNetworkPara(folder, fln_all[i])
        
        wEP = abs(np.sum(NetPar.wEP,1))
        wPP = abs(np.sum(NetPar.wPP,1))
        wPS = abs(np.sum(NetPar.wPS,1))
        wPV = abs(np.sum(NetPar.wPV,1))
        
        idx_wEP = np.random.randint(0,len(wEP), size=M)
        idx_wPP = np.random.randint(0,len(wPP), size=M)
        idx_wPS = np.random.randint(0,len(wPS), size=M)
        idx_wPV = np.random.randint(0,len(wPV), size=M)
        
        Examples[:,0,i] = (1+wPP[idx_wPP])/wEP[idx_wEP]
        Examples[:,1,i] = wPS[idx_wPS]
        Examples[:,2,i] = wPV[idx_wPV]
    
    return Examples
    

def Output_ModLearnRate(NeuPar, StimPar, bool_nPE, bool_pPE, paras, folder, fln_save):
    
    # Extract parameters
    NCells = NeuPar.NCells
    NE = np.int32(NCells[0]/2)
    
    stimulus = StimPar.stimuli[0]
    
    w1, w2, eta, N_ini = paras
    
    ### Load data
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticPredNet_PE_' + fln_save + '.dat',delimiter=' ')
    t, RE = arr[:,0], arr[:, 1:NE+1]
    
    ### Compute output with varying learning rates
    nsteps = len(t)
    R_mod = np.zeros(nsteps)    
    R_classic = np.zeros(nsteps)
    
    for n in range(nsteps):
        
        R_step = RE[n,:]
        nPE = np.mean(R_step[bool_nPE==1])
        pPE = np.mean(R_step[bool_pPE==1])
            
        R_mod[n] = w1 * stimulus
        w1 -= eta * (1+nPE+pPE) * (R_mod[n] - stimulus)
        
        R_classic[n] = w2 * stimulus
        w2 -= eta * (R_classic[n] - stimulus)
        
    dR_mod = np.mean(np.diff(R_mod[:N_ini]))
    dR_classic = np.mean(np.diff(R_classic[:N_ini]))
        
    return R_mod, R_classic, dR_mod, dR_classic


def Output_PredNet(NeuPar, NetPar, stimuli, folder, fln_save):
    
    ### Pre-processing
    w_out_E = NetPar.w_out_E
    w_out_B = NetPar.w_out_B # not used in this version
    
    ### P-PE activity
    NCells = NeuPar.NCells
    NE = NCells[0]
    
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticPredNet_P_' + fln_save + '.dat',delimiter=' ')
    L, B = arr[:,1:3], arr[:,3:]
    
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticPredNet_PE_' + fln_save + '.dat',delimiter=' ')
    t, RE = arr[:,0], arr[:, 1:NE+1]
    
    ### Output neuron activity
    RE = RE.transpose()
    B = B.transpose()
    
    #Out = w_out_E @ RE + w_out_B @ B # dimension: 2 x n_timesteps
    Out = w_out_E @ RE + stimuli # dimension: 2 x n_timesteps
    Out[Out<0] = dtype(0)

    return t, RE, L, B, Out


def Output_per_stimulus_PredNet(t, Out, StimPar, SimPar):
    
    ### Pre-processing
    stim_duration = SimPar.stim_duration
    stimuli = StimPar.stimuli
    
    ### Activity of output neurons per stimulus
    Out_per_stimulus = np.zeros((2, len(stimuli)))
    
    for i in range(len(stimuli)):
        Out_per_stimulus[:,i] = np.mean(Out[:, (t>(0.5+i)*stim_duration) & (t<(1+i)*stim_duration)], 1)
    
    return Out_per_stimulus


def Inputs_MM_PB(folder, fln_load, fln_save):
    
    ### Load data
    _, _, InPar, StimPar_Test, SimPar_Test, _, _ = LoadNetworkPara(folder, fln_load)
    NeuPar, NetPar, _, _, _, _, _ = LoadNetworkPara(folder, fln_save) 

    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_' + fln_save + '.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:]
    
    ### Break R into populations
    NCells = NeuPar.NCells
    
    rE = R[:,:NCells[0]]
    rD = R[:,-NCells[0]:]
    rP = R[:,NCells[0]:sum(NCells[:2])]
    rS = R[:,sum(NCells[:2]):sum(NCells[:3])]
    
    ### Compute input in MM and PB phase
    stim_duration = SimPar_Test.stim_duration
    num_time_steps_per_stimuli = int(stim_duration/SimPar_Test.dt)
    V = np.repeat(StimPar_Test.stim_visual, num_time_steps_per_stimuli) 
    M = np.repeat(StimPar_Test.stim_motor, num_time_steps_per_stimuli)  
    
    Inp_soma_Inh = np.zeros((len(t), NCells[0]))
    Inp_soma_Exc = np.zeros((len(t), NCells[0]))
    Inp_dend_Inh = np.zeros((len(t), NCells[0]))
    Inp_dend_Exc = np.zeros((len(t), NCells[0]))
    
    for i in range(len(t)):
        Inp_soma_Inh[i,:] = NetPar.wEP @ rP[i,:] 
        Inp_soma_Exc[i,:] = NetPar.wED @ rD[i,:] + V[i] + InPar.inp_ext_soma[:NCells[0]]
        Inp_dend_Inh[i,:] = NetPar.wDS @ rS[i,:] 
        Inp_dend_Exc[i,:] = NetPar.wDE @ rE[i,:] + M[i] + InPar.inp_ext_dend
    
    Inp_soma = Inp_soma_Inh + Inp_soma_Exc
    
    Inp_MM = np.zeros(NCells[0])
    Inp_PB = np.zeros(NCells[0])
    
    Inp_MM = np.mean(Inp_soma[(t>3.5*stim_duration) & (t<4*stim_duration),:],0)
    Inp_PB = np.mean(Inp_soma[(t>5.5*stim_duration) & (t<6*stim_duration),:],0)
        
    return Inp_MM, Inp_PB


def Activity_Phases(t, RE, RP, RS, RV, SimPar):
    
    stim_duration = SimPar.stim_duration
    
    E_phases  = np.zeros((np.size(RE,1),4))
    E_phases[:,0] = np.mean(RE[(t>0.5*stim_duration) & (t<stim_duration),:],0)
    E_phases[:,1] = np.mean(RE[(t>1.5*stim_duration) & (t<2*stim_duration),:],0)
    E_phases[:,2] = np.mean(RE[(t>3.5*stim_duration) & (t<4*stim_duration),:],0)
    E_phases[:,3] = np.mean(RE[(t>5.5*stim_duration) & (t<6*stim_duration),:],0)
    
    P_phases  = np.zeros((np.size(RP,1),4))
    P_phases[:,0] = np.mean(RP[(t>0.5*stim_duration) & (t<stim_duration),:],0)
    P_phases[:,1] = np.mean(RP[(t>1.5*stim_duration) & (t<2*stim_duration),:],0)
    P_phases[:,2] = np.mean(RP[(t>3.5*stim_duration) & (t<4*stim_duration),:],0)
    P_phases[:,3] = np.mean(RP[(t>5.5*stim_duration) & (t<6*stim_duration),:],0)
    
    S_phases  = np.zeros((np.size(RS,1),4))
    S_phases[:,0] = np.mean(RS[(t>0.5*stim_duration) & (t<stim_duration),:],0)
    S_phases[:,1] = np.mean(RS[(t>1.5*stim_duration) & (t<2*stim_duration),:],0)
    S_phases[:,2] = np.mean(RS[(t>3.5*stim_duration) & (t<4*stim_duration),:],0)
    S_phases[:,3] = np.mean(RS[(t>5.5*stim_duration) & (t<6*stim_duration),:],0)
    
    V_phases  = np.zeros((np.size(RV,1),4))
    V_phases[:,0] = np.mean(RV[(t>0.5*stim_duration) & (t<stim_duration),:],0)
    V_phases[:,1] = np.mean(RV[(t>1.5*stim_duration) & (t<2*stim_duration),:],0)
    V_phases[:,2] = np.mean(RV[(t>3.5*stim_duration) & (t<4*stim_duration),:],0)
    V_phases[:,3] = np.mean(RV[(t>5.5*stim_duration) & (t<6*stim_duration),:],0)
    
    return E_phases, P_phases, S_phases, V_phases


def Inputs_to_PCs(NeuPar, NetPar, InPar, StimPar, SimPar, folder, fln):
    
    ### Load data
    arr = np.loadtxt('Results/Data/' + folder + '/Data_StaticNetwork_Inputs2PCs_' + fln + '.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:]
    
    ### Excitatory and inhibitory inputs to PCs
    NCells = NeuPar.NCells
    
    rE = R[:,:NCells[0]]
    rD = R[:,-NCells[0]:]
    rP = R[:,NCells[0]:sum(NCells[:2])]
    rS = R[:,sum(NCells[:2]):sum(NCells[:3])]
    
    num_time_steps_per_stimuli = int(SimPar.stim_duration/SimPar.dt)
    V = np.repeat(StimPar.stim_visual, num_time_steps_per_stimuli) 
    M = np.repeat(StimPar.stim_motor, num_time_steps_per_stimuli)  
    
    Inp_soma_Inh = np.zeros((len(t), NCells[0]))
    Inp_soma_Exc = np.zeros((len(t), NCells[0]))
    Inp_dend_Inh = np.zeros((len(t), NCells[0]))
    Inp_dend_Exc = np.zeros((len(t), NCells[0]))
    
    for i in range(len(t)):
        Inp_soma_Inh[i,:] = NetPar.wEP @ rP[i,:] 
        Inp_soma_Exc[i,:] = NetPar.wED @ rD[i,:] + V[i] + InPar.inp_ext_soma[:NCells[0]]
        Inp_dend_Inh[i,:] = NetPar.wDS @ rS[i,:] 
        Inp_dend_Exc[i,:] = NetPar.wDE @ rE[i,:] + M[i] + InPar.inp_ext_dend
        
    return t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M
    

def Classify_PCs(t, PC, SimPar):

    ### Averaged steady-state activity in phases 
    NE = np.size(PC,1)
    stim_duration = SimPar.stim_duration
    
    PC_phases  = np.zeros((NE,4))
    PC_phases[:,0] = np.mean(PC[(t>0.5*stim_duration) & (t<stim_duration),:],0)
    PC_phases[:,1] = np.mean(PC[(t>1.5*stim_duration) & (t<2*stim_duration),:],0)
    PC_phases[:,2] = np.mean(PC[(t>3.5*stim_duration) & (t<4*stim_duration),:],0)
    PC_phases[:,3] = np.mean(PC[(t>5.5*stim_duration) & (t<6*stim_duration),:],0)
    
    ### Parameters defining classification criterion
    Max_per_neuron = np.max(abs(PC_phases),1)
    tol_per_neuron = 0.1 * Max_per_neuron # tolerance
    min_peak_factor = 3
    min_peak = 0.5

    ### Find nPE neurons
    bool_nPE = 1*((abs(PC_phases[:,1]) < tol_per_neuron) & 
                 (abs(PC_phases[:,3]) < tol_per_neuron) &
                 (PC_phases[:,2] > np.clip(min_peak_factor * tol_per_neuron, min_peak, None)))
    
    ### Find pPE neurons
    bool_pPE = 1*((abs(PC_phases[:,1]) < tol_per_neuron) & 
                 (abs(PC_phases[:,2]) < tol_per_neuron) &
                 (PC_phases[:,3] > np.clip(min_peak_factor*tol_per_neuron, min_peak, None)))
            
    return bool_nPE, bool_pPE



def Activity_relative_to_BL(NeuPar, StimPar, SimPar, folder, fln, window_out = dtype(500)):
    
    ### Load data
    NCells = NeuPar.NCells
    NC = np.cumsum(NCells)

    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_' + fln + '.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:]
    
    ### Define phases
    stim_duration = SimPar.stim_duration
    Test_phases = stim_duration * np.array([1,2,3,4,5,6,7])
    
    ### Subtract baseline firing rate
    BL_end = Test_phases[0]
    R -= np.mean(R[(t>window_out) & (t<BL_end),:],0)
    
    ### Break R into populations
    RE = R[:,:NC[0]]
    RP = R[:,NC[0]:NC[1]]
    RS = R[:,NC[1]:NC[2]]
    RV = R[:,NC[2]:NC[2]+NCells[2]]
    
    return t, RE, RP, RS, RV           
    

def Activity_Test_MFN(folder, fln, stim_duration):
    
    ### Load data    
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + str(fln) + '.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:] 
    
    ### Compute R in all phases
    R_phases = np.zeros((8,4),dtype=dtype) # row: neuron types/compartments | column: BL, FB, MM, PB
    
    R_phases[:,0] = np.mean(R[(t>0.5*stim_duration) & (t<stim_duration),:],0)
    R_phases[:,1] = np.mean(R[(t>1.5*stim_duration) & (t<2*stim_duration),:],0)
    R_phases[:,2] = np.mean(R[(t>3.5*stim_duration) & (t<4*stim_duration),:],0)
    R_phases[:,3] = np.mean(R[(t>5.5*stim_duration) & (t<6*stim_duration),:],0)
    
    return R_phases
    

def PathwaysCont_old(VS, VV, filename):
    
    # load data
    with open(filename,'rb') as f:
        xopt, W, optimize_flag, VS, VV = pickle.load(f)
        
    W[optimize_flag!=0] = xopt
    N = np.size(W,0)
    neurons_visual = np.array([1, 1, 0, 0, 1, 0, VS, VV])
    neurons_motor = np.array([0, 0, 1, 1, 0, 1, 1-VS, 1-VV])
    
    # compute pathways or net contributions from different cell types
    stim_PB = neurons_visual
    stim_MM = neurons_motor
    VD_inv = np.linalg.inv(W-np.eye(N))
    Pthwys_nE_E = np.zeros((N,3))
    Pthwys_pE_E = np.zeros((N,3))
    Pthwys_nE_I = np.zeros((N,3))
    Pthwys_pE_I = np.zeros((N,3))
    
    Pthwys_nD_E = np.zeros((N,3))
    Pthwys_pD_E = np.zeros((N,3))
    Pthwys_nD_I = np.zeros((N,3))
    Pthwys_pD_I = np.zeros((N,3))
    
    # Feedback phase
    Pthwys_nE = -VD_inv[0,:]
    Pthwys_pE = -VD_inv[1,:]
    Pthwys_nE_E[:,0] = Pthwys_nE*(Pthwys_nE>0)
    Pthwys_nE_I[:,0] = abs(Pthwys_nE*(Pthwys_nE<0))
    Pthwys_pE_E[:,0] = Pthwys_pE*(Pthwys_pE>0)
    Pthwys_pE_I[:,0] = abs(Pthwys_pE*(Pthwys_pE<0))
    
    Pthwys_nD = -VD_inv[2,:]
    Pthwys_pD = -VD_inv[3,:]
    Pthwys_nD_E[:,0] = Pthwys_nD*(Pthwys_nD>0)
    Pthwys_nD_I[:,0] = abs(Pthwys_nD*(Pthwys_nD<0))
    Pthwys_pD_E[:,0] = Pthwys_pD*(Pthwys_pD>0)
    Pthwys_pD_I[:,0] = abs(Pthwys_pD*(Pthwys_pD<0))
    
    # Mismatch phase
    if VS==1: # dendrites are excited
        Pthwys_nE = -VD_inv[0,:] * stim_MM
        Pthwys_pE = -VD_inv[1,:] * stim_MM
        Pthwys_nE_E[:,1] = Pthwys_nE*(Pthwys_nE>0)
        Pthwys_nE_I[:,1] = abs(Pthwys_nE*(Pthwys_nE<0))
        Pthwys_pE_E[:,1] = Pthwys_pE*(Pthwys_pE>0)
        Pthwys_pE_I[:,1] = abs(Pthwys_pE*(Pthwys_pE<0))
    else: # dendrites are inhibited
        W_new = np.repeat(W,[1,1,0,0,1,1,1,1],axis=0)
        W_new = np.repeat(W_new,[1,1,0,0,1,1,1,1],axis=1)
        V0_inv = np.linalg.inv(W_new-np.eye(N-2))
        V0_inv = np.insert(V0_inv, [2,2], values=0, axis=1)
        V0_inv = np.insert(V0_inv, [2,2], values=0, axis=0)
        
        Pthwys_nE = -V0_inv[0,:] * stim_MM
        Pthwys_pE = -V0_inv[1,:] * stim_MM
        Pthwys_nE_E[:,1] = Pthwys_nE*(Pthwys_nE>0)
        Pthwys_nE_I[:,1] = abs(Pthwys_nE*(Pthwys_nE<0))
        Pthwys_pE_E[:,1] = Pthwys_pE*(Pthwys_pE>0)
        Pthwys_pE_I[:,1] = abs(Pthwys_pE*(Pthwys_pE<0))
        
    Pthwys_nD = -VD_inv[2,:] * stim_MM
    Pthwys_pD = -VD_inv[3,:] * stim_MM
    Pthwys_nD_E[:,1] = Pthwys_nD*(Pthwys_nD>0)
    Pthwys_nD_I[:,1] = abs(Pthwys_nD*(Pthwys_nD<0))
    Pthwys_pD_E[:,1] = Pthwys_pD*(Pthwys_pD>0)
    Pthwys_pD_I[:,1] = abs(Pthwys_pD*(Pthwys_pD<0))
        
    # Playback phase
    if VS==1: # dendrites are inhibited
        W_new = np.repeat(W,[1,1,0,0,1,1,1,1],axis=0)
        W_new = np.repeat(W_new,[1,1,0,0,1,1,1,1],axis=1)
        V0_inv = np.linalg.inv(W_new-np.eye(N-2))
        V0_inv = np.insert(V0_inv, [2,2], values=0, axis=1)
        V0_inv = np.insert(V0_inv, [2,2], values=0, axis=0)
        
        Pthwys_nE = -V0_inv[0,:] * stim_PB
        Pthwys_pE = -V0_inv[1,:] * stim_PB
        Pthwys_nE_E[:,2] = Pthwys_nE*(Pthwys_nE>0)
        Pthwys_nE_I[:,2] = abs(Pthwys_nE*(Pthwys_nE<0))
        Pthwys_pE_E[:,2] = Pthwys_pE*(Pthwys_pE>0)
        Pthwys_pE_I[:,2] = abs(Pthwys_pE*(Pthwys_pE<0))
    else:
        Pthwys_nE = -VD_inv[0,:] * stim_PB
        Pthwys_pE = -VD_inv[1,:] * stim_PB
        Pthwys_nE_E[:,2] = Pthwys_nE*(Pthwys_nE>0)
        Pthwys_nE_I[:,2] = abs(Pthwys_nE*(Pthwys_nE<0))
        Pthwys_pE_E[:,2] = Pthwys_pE*(Pthwys_pE>0)
        Pthwys_pE_I[:,2] = abs(Pthwys_pE*(Pthwys_pE<0))
        
    Pthwys_nD = -VD_inv[0,:] * stim_PB
    Pthwys_pD = -VD_inv[1,:] * stim_PB
    Pthwys_nD_E[:,2] = Pthwys_nD*(Pthwys_nD>0)
    Pthwys_nD_I[:,2] = abs(Pthwys_nD*(Pthwys_nD<0))
    Pthwys_pD_E[:,2] = Pthwys_pD*(Pthwys_pD>0)
    Pthwys_pD_I[:,2] = abs(Pthwys_pD*(Pthwys_pD<0))
    
    return Pthwys_nE_E, Pthwys_nE_I, Pthwys_pE_E, Pthwys_pE_I, Pthwys_nD_E, Pthwys_nD_I, Pthwys_pD_E, Pthwys_pD_I
    

def Inputs_to_PCs_MFN(folder, fln, Weights, stim_duration, stim_high): #, fixed_input):

    R_phases = Activity_Test_MFN(folder, fln, stim_duration)
    
    r_nE_phases = R_phases[0,:]
    r_pE_phases = R_phases[1,:]
    r_nD_phases = R_phases[2,:]
    r_pD_phases = R_phases[3,:]
    r_Pv_phases = R_phases[4,:]
    r_Pm_phases = R_phases[5,:]
    r_S_phases = R_phases[6,:]
    
    w_nEnD = Weights[0,2]
    w_nEPv = Weights[0,4]
    w_nEPm = Weights[0,5]
    w_pEpD = Weights[1,3]
    w_pEPv = Weights[1,4]
    w_pEPm = Weights[1,5]
    w_nDnE = Weights[2,0]
    w_nDpE = Weights[2,1]
    w_nDS = Weights[2,6]
    w_pDnE = Weights[3,0]
    w_pDpE = Weights[3,1]
    w_pDS = Weights[3,6]

    E_nE_phases =  w_nEnD * r_nD_phases[1:] + np.array([1,0,1])*stim_high - w_nEnD * r_nD_phases[0] 
    I_nE_phases = w_nEPv * r_Pv_phases[1:] + w_nEPm * r_Pm_phases[1:] - (w_nEPv * r_Pv_phases[0] + w_nEPm * r_Pm_phases[0])
     
    E_pE_phases =  w_pEpD * r_pD_phases[1:] + np.array([1,0,1])*stim_high - w_pEpD * r_pD_phases[0] 
    I_pE_phases = w_pEPv * r_Pv_phases[1:] + w_pEPm * r_Pm_phases[1:] - (w_pEPv * r_Pv_phases[0] + w_pEPm * r_Pm_phases[0])
    
    E_nD_phases = w_nDnE * r_nE_phases[1:] + w_nDpE * r_pE_phases[1:] + np.array([1,1,0])*stim_high - (w_nDnE * r_nE_phases[0] + w_nDpE * r_pE_phases[0])
    I_nD_phases = w_nDS * r_S_phases[1:] - w_nDS * r_S_phases[0]
    
    E_pD_phases = w_pDnE * r_nE_phases[1:] + w_pDpE * r_pE_phases[1:] + np.array([1,1,0])*stim_high - (w_pDnE * r_nE_phases[0] + w_pDpE * r_pE_phases[0])
    I_pD_phases = w_pDS * r_S_phases[1:] - w_pDS * r_S_phases[0] 
    
    E_nE_phases[I_nE_phases>0] += I_nE_phases[I_nE_phases>0]
    I_nE_phases[I_nE_phases>0] = 0
    I_nE_phases[E_nE_phases<0] += E_nE_phases[E_nE_phases<0]
    E_nE_phases[E_nE_phases<0] = 0
    E_pE_phases[I_pE_phases>0] += I_pE_phases[I_pE_phases>0]
    I_pE_phases[I_pE_phases>0] = 0
    I_pE_phases[E_pE_phases<0] += E_pE_phases[E_pE_phases<0]
    E_pE_phases[E_pE_phases<0] = 0
    
    E_nD_phases[I_nD_phases>0] += I_nD_phases[I_nD_phases>0]
    I_nD_phases[I_nD_phases>0] = 0
    I_nD_phases[E_nD_phases<0] += E_nD_phases[E_nD_phases<0]
    E_nD_phases[E_nD_phases<0] = 0
    E_pD_phases[I_pD_phases>0] += I_pD_phases[I_pD_phases>0]
    I_pD_phases[I_pD_phases>0] = 0
    I_pD_phases[E_pD_phases<0] += E_pD_phases[E_pD_phases<0]
    E_pD_phases[E_pD_phases<0] = 0
    
    return E_nE_phases, I_nE_phases, E_pE_phases, I_pE_phases, E_nD_phases, I_nD_phases, E_pD_phases, I_pD_phases


def PathwaysCont(VS, VV, filename):
    
    # load data
    with open(filename,'rb') as f:
        xopt, W, optimize_flag, VS, VV = pickle.load(f)
        
    W[optimize_flag!=0] = xopt
    N = np.size(W,0)
      
    # initialise 
    VD_inv = np.linalg.inv(W-np.eye(N))
    Pthwys_nE_E = np.zeros(N)
    Pthwys_pE_E = np.zeros(N)
    Pthwys_nE_I = np.zeros(N)
    Pthwys_pE_I = np.zeros(N)
    
    Pthwys_nD_E = np.zeros(N)
    Pthwys_pD_E = np.zeros(N)
    Pthwys_nD_I = np.zeros(N)
    Pthwys_pD_I = np.zeros(N)
    
    # compute pathways or net contributions from different cell types
    Pthwys_nE = -VD_inv[0,:]
    Pthwys_pE = -VD_inv[1,:]
    Pthwys_nE_E = Pthwys_nE*(Pthwys_nE>0)
    Pthwys_nE_I = abs(Pthwys_nE*(Pthwys_nE<0))
    Pthwys_pE_E = Pthwys_pE*(Pthwys_pE>0)
    Pthwys_pE_I = abs(Pthwys_pE*(Pthwys_pE<0))
    
    Pthwys_nD = -VD_inv[2,:]
    Pthwys_pD = -VD_inv[3,:]
    Pthwys_nD_E = Pthwys_nD*(Pthwys_nD>0)
    Pthwys_nD_I = abs(Pthwys_nD*(Pthwys_nD<0))
    Pthwys_pD_E = Pthwys_pD*(Pthwys_pD>0)
    Pthwys_pD_I = abs(Pthwys_pD*(Pthwys_pD<0))
    
    return Pthwys_nE_E, Pthwys_nE_I, Pthwys_pE_E, Pthwys_pE_I, Pthwys_nD_E, Pthwys_nD_I, Pthwys_pD_E, Pthwys_pD_I
    
    
def Solve_Balance_PE(Weights, optimize_flag, VS, VV, path=None, filename=None, save_flag=0):
    
    ### Nelder-simplex algorithm to solve PE equations
    if VS==1:
        xopt, fopt, nter, funcalls, warnflag = optimize.fmin(Objective_functon_VS_1, Weights[optimize_flag!=0], 
                                                             args=(Weights, optimize_flag, VS, VV), full_output=1)
    else:
        xopt, fopt, nter, funcalls, warnflag = optimize.fmin(Objective_functon_VS_0, Weights[optimize_flag!=0], 
                                                             args=(Weights, optimize_flag, VS, VV), full_output=1)
        
    ### Save optimal weights if save_flag=1
    if save_flag==1:
        with open(filename,'wb') as f:
            pickle.dump([xopt, Weights, optimize_flag, VS, VV],f)
        
    return xopt, fopt


def Objective_functon_VS_1(x, W, optimize_flag, VS, VV):
    
    ones_PB = np.array([1,0,VS,VV]) # IN input on/off
    ones_MM = np.array([0,1,1-VS,1-VV]) # IN input on/off
    N = np.size(W,0)
    
    M = np.copy(W)
    M[optimize_flag!=0] = x
    M -= np.eye(N)
    
    # compute errors
    MII = M[4:,4:]
    MII_inv = np.linalg.inv(MII)
    MpDI, MnDI = M[3,4:], M[2,4:]
    MnEI, MpEI  = M[0,4:], M[1,4:]
    MIpE, MInE = M[4:,1], M[4:,0]
    wpEpD, wpDnE, wnEnD, wnDnE = M[1,3], M[3,0], M[0,2], M[2,0]
    
    MpEpE = (1 + MpEI @ MII_inv @ MIpE)
    MpE_ = (1 - MpEI @ MII_inv @ ones_PB)
    MnEnE = 1 - wnEnD*(wnDnE - MnDI @ MII_inv @ MInE) + MnEI @ MII_inv @ MInE
    
    M_pD_In = MpDI @ MII_inv @ ones_MM
    M_pE_In = MpEI @ MII_inv @ ones_MM
    M_pD_nE = MpDI @ MII_inv @ MInE
    M_pE_nE = MpEI @ MII_inv @ MInE
    M_nD_In = MnDI @ MII_inv @ ones_MM
    M_nE_In = MnEI @ MII_inv @ ones_MM
    
    Error_nE_FBD = (1 - MnDI @  MII_inv @ np.ones(4))**2
    Error_pE_FBD = (1 - MpDI @  MII_inv @ np.ones(4))**2
    Error_nE_FB = (1 - MnEI @ MII_inv @ np.ones(4))**2
    Error_pE_FB = (1 - MpEI @ MII_inv @ np.ones(4))**2
    Error_nE_PB = (1 - MnEI @ MII_inv @ (ones_PB + MIpE * MpE_/MpEpE))**2
    Error_pE_MM = ((wpEpD * (1 - M_pD_In) - M_pE_In) + 
                   (wpEpD * (wpDnE - M_pD_nE) - M_pE_nE)*
                   (wnEnD * (1 - M_nD_In) - M_nE_In)/MnEnE)**2
    
    if any(x>0):
        Error_all = 1e5
    else:
        Error_all = Error_nE_FB + Error_pE_FB + Error_nE_PB + Error_pE_MM
        Error_all += Error_nE_FBD + Error_pE_FBD
    
    return Error_all


def Objective_functon_VS_0(x, W, optimize_flag, VS, VV):
    
    ones_PB = np.array([1,0,VS,VV]) # IN input on/off
    ones_MM = np.array([0,1,1-VS,1-VV]) # IN input on/off
    N = np.size(W,0)
    
    M = np.copy(W)
    M[optimize_flag!=0] = x
    M -= np.eye(N)
    
    # compute errors
    MII = M[4:,4:]
    MII_inv = np.linalg.inv(MII)
    MpDI, MnDI = M[3,4:], M[2,4:]
    MnEI, MpEI  = M[0,4:], M[1,4:]
    MIpE, MInE = M[4:,1], M[4:,0]
    wpEpD, wnDpE, wnEnD, wpDpE = M[1,3], M[2,1], M[0,2], M[3,1] 
    
    MnEnE = (1 + MnEI @ MII_inv @ MInE)
    M_pD_In = MpDI @ MII_inv @ ones_PB
    M_pE_In = MpEI @ MII_inv @ ones_PB
    M_pD_pE = MpDI @ MII_inv @ MIpE
    M_pE_pE = MpEI @ MII_inv @ MIpE
    
    M_nE_In = MnEI @ MII_inv @ ones_MM
    M_nE_pE = MnEI @ MII_inv @ MIpE
    A = 1 - wpEpD * M_pD_In - M_pE_In
    B = 1 - wpEpD * (wpDpE - M_pD_pE) + M_pE_pE
    
    Error_nE_FBD = (1 - MnDI @  MII_inv @ np.ones(4))**2
    Error_pE_FBD = (1 - MpDI @  MII_inv @ np.ones(4))**2
    Error_nE_FB = (1 - MnEI @ MII_inv @ np.ones(4))**2
    Error_pE_FB = (1 - MpEI @ MII_inv @ np.ones(4))**2
    Error_pE_MM = (MpEI @ MII_inv @ (ones_MM - MInE*M_nE_In/MnEnE))**2
    Error_nE_PB = (1 - M_nE_pE * A/B - MnEI @ MII_inv @ ones_PB
                   + wnEnD*(wnDpE*A/B - MnDI @ MII_inv @ (MIpE*A/B + ones_PB)))**2
    
    if any(x>0):
        Error_all = 1e5
    else:
        Error_all = Error_nE_FB + Error_pE_FB + Error_nE_PB + Error_pE_MM
        Error_all += Error_nE_FBD + Error_pE_FBD
    
    return Error_all 


def DeterminePhasePlane_10(VE, VP, MP, wSV, wVS, num):
    
    if wSV>=1:
        print('VIP -> SOM must be smaller than 1')
    
    ww = (1 - wVS*wSV)
    wDS = ww/(1-wSV)
    
    if ww<0:
        print('Product of mutual inhbition between SOM and VIP must be smaller than 1')
    
    # nPE neurons
    if VE>0:
        alpha_bound_nPE = (VP + wVS*MP)/VE
        alpha_nPE = np.linspace(0.0, alpha_bound_nPE, num, dtype=dtype)
        wPS_nPE = VP + wVS*MP - alpha_nPE*VE
        wPV_nPE = wSV*wPS_nPE + ww*MP
    else:
        alpha_nPE = np.linspace(0.0, 2, num, dtype=dtype) # no upper bound ... just for visualization
        wPS_nPE = (VP + wVS*MP) * np.ones_like(alpha_nPE) 
        wPV_nPE = wSV*wPS_nPE + ww*MP 
        
    # pPE neurons
    alpha_bound_pPE_1 = (VP + wVS*MP)/(VE - (1-wVS)/(1-wSV)) 
    alpha_bound_pPE_2 = (MP + wSV*VP)/(1+wSV*VE)
    alpha_bound_pPE = min(alpha_bound_pPE_1 if alpha_bound_pPE_1 > 0 else 1000,
                          alpha_bound_pPE_2 if alpha_bound_pPE_2 > 0 else 1000)
    alpha_pPE = np.linspace(0.0, alpha_bound_pPE, num, dtype = dtype)
    wPS_pPE = VP + wVS * MP - alpha_pPE * (VE - (1-wVS)/(1-wSV))
    wPV_pPE = wSV*wPS_pPE + ww * MP - ww * alpha_pPE/(1-wSV)
    
    # dMM neurons
    wPS_bound_1 = np.maximum(VP + wVS*MP - (VE - (1-wVS)/(1-wSV)) * alpha_nPE, 0)
    wPS_bound_2 = np.maximum(VP + wVS*MP - VE * alpha_nPE, 0)
    wPS_bound_3 = np.maximum(wDS * (MP + VP - alpha_nPE * VE), 0)
    wPS_max = np.minimum(wPS_bound_1, wPS_bound_2, wPS_bound_3)
    alpha_dMM = np.repeat(alpha_nPE, num)
    wPS_dMM = np.ravel(np.linspace(0, wPS_max, num, dtype=dtype), order='F')
    wPV_dMM = (ww * (MP + VP - alpha_dMM*VE) - wPS_dMM*(1-wSV)) / (1-wVS)
    
    # hMM neurons
    wPS_hMM_lower_1 = np.maximum(VP + wVS*MP - (VE - (1-wVS)/(1-wSV)) * alpha_pPE, 0)
    wPS_hMM_lower_2 = np.maximum(VP + wVS*MP - VE * alpha_pPE, 0)
    wPS_hMM_lower = np.maximum(wPS_hMM_lower_1, wPS_hMM_lower_2)
    wPS_hMM_upper = wDS * (MP + VP - alpha_pPE * VE)   
    alpha_hMM = np.repeat(alpha_pPE, num)
    wPS_hMM = np.ravel(np.linspace(wPS_hMM_lower,wPS_hMM_upper,num, dtype=dtype), order='F')
    wPV_hMM = (ww * (MP + VP - alpha_hMM*VE) - wPS_hMM*(1-wSV)) / (1-wVS)
    
    alpha_all = {"alpha_nPE": alpha_nPE, "alpha_pPE": alpha_pPE, "alpha_dMM": alpha_dMM, "alpha_hMM": alpha_hMM}
    wPS_all = {"wPS_nPE": wPS_nPE, "wPS_pPE": wPS_pPE, "wPS_dMM": wPS_dMM, "wPS_hMM": wPS_hMM}
    wPV_all = {"wPV_nPE": wPV_nPE, "wPV_pPE": wPV_pPE, "wPV_dMM": wPV_dMM, "wPV_hMM": wPV_hMM}
    
    return alpha_all, wPS_all, wPV_all


def DeterminePhasePlane_11(VE, VP, MP, wSV, wVS, num):
    
    ww = (1 - wVS*wSV)
    wDS = ww/(1-wSV)
    
    if MP==0:
        # nPE
        if VE>0:
            alpha_nPE_once = np.linspace(0.0, (MP + VP)/VE, num, dtype=dtype)
            
        else:
            alpha_nPE_once = np.linspace(0.0, 2 , num, dtype=dtype) # no upper bound
        alpha_nPE = np.repeat(alpha_nPE_once, num)
        wPS_nPE = np.ravel(np.linspace(0, wDS * (MP + VP - alpha_nPE_once * VE), num, dtype=dtype), order='F')
        wPV_nPE = (ww*(VP + MP - alpha_nPE*VE) - wPS_nPE*(1-wSV))/(1-wVS)
    
        # pPE
        alpha_pPE = [np.nan]
        wPS_pPE = [np.nan]
        wPV_pPE = [np.nan]
        
        # hMM 
        alpha_hMM = [np.nan]
        wPS_hMM = [np.nan]
        wPV_hMM = [np.nan]
    
    else:
        # nPE
        alpha_nPE = [np.nan]
        wPS_nPE = [np.nan]
        wPV_nPE = [np.nan]
        
        # pPE
        alpha_pPE_value = MP
        alpha_pPE = np.repeat(alpha_pPE_value, num)
        wPS_pPE = np.ravel(np.linspace(0.0, wDS * (MP + VP - alpha_pPE_value*VE), num, dtype=dtype), order='F')  
        wPV_pPE = (ww * (MP+VP-alpha_pPE*VE) - wPS_pPE*(1-wSV)) / (1-wVS)
        
        # hMM
        alpha_hMM_once = np.linspace(0, alpha_pPE_value, num, endpoint=False, dtype=dtype)
        alpha_hMM = np.repeat(alpha_hMM_once, num)
        wPS_hMM = np.ravel(np.linspace(0.0, wDS * (MP + VP - alpha_hMM_once*VE), num, dtype=dtype), order='F') 
        wPV_hMM = (ww * (MP+VP-alpha_hMM*VE) - wPS_hMM*(1-wSV)) / (1-wVS)
        
    # dMM
    alpha_dMM = alpha_nPE
    wPS_dMM = wPS_nPE
    wPV_dMM = wPV_nPE

    alpha_all = {"alpha_nPE": alpha_nPE, "alpha_pPE": alpha_pPE, "alpha_dMM": alpha_dMM, "alpha_hMM": alpha_hMM}
    wPS_all = {"wPS_nPE": wPS_nPE, "wPS_pPE": wPS_pPE, "wPS_dMM": wPS_dMM, "wPS_hMM": wPS_hMM}
    wPV_all = {"wPV_nPE": wPV_nPE, "wPV_pPE": wPV_pPE, "wPV_dMM": wPV_dMM, "wPV_hMM": wPV_hMM}
        
    return alpha_all, wPS_all, wPV_all
    

def DeterminePhasePlane_00(VE, VP, MP, wSV, wVS, num):
    
    ww = (1-wVS*wSV)
    wDS = ww/(1-wSV)
    
    print('Please note that in the current network neurons remain at baseline in all phases. The necessary\
          non-linearity in the dendrites is not reached when SOM and VIP neurons receive the same input as the dendrites.')
       
    if VE>0:
        alpha_nPE_once = VP/VE
    else:
        if VP!=0:
            print('No nPE, pPE, dMM, hMM or baseline neurons possible as long as wEP>0 or wSVwVS<1 ... !')
            alpha_nPE_once = np.nan
        else:
            alpha_nPE_once = np.linspace(0.0, 2 , num, dtype=dtype)
   
    # only baseline neurons possible => nPE=pPE
    alpha_nPE = np.repeat(alpha_nPE_once, num)
    wPS_nPE = np.ravel(np.linspace(0.0, wDS * (MP + VP - alpha_nPE_once*VE), num, dtype=dtype), order='F')  
    wPV_nPE = (ww * (MP+VP-alpha_nPE*VE) - wPS_nPE*(1-wSV)) / (1-wVS)
    
    alpha_pPE = alpha_nPE
    wPS_pPE = wPS_nPE
    wPV_pPE = wPV_nPE
    
    alpha_hMM = [np.nan]
    wPS_hMM = [np.nan]
    wPV_hMM = [np.nan]
    
    alpha_dMM = [np.nan]
    wPS_dMM = [np.nan]
    wPV_dMM = [np.nan]
        
    alpha_all = {"alpha_nPE": alpha_nPE, "alpha_pPE": alpha_pPE, "alpha_dMM": alpha_dMM, "alpha_hMM": alpha_hMM}
    wPS_all = {"wPS_nPE": wPS_nPE, "wPS_pPE": wPS_pPE, "wPS_dMM": wPS_dMM, "wPS_hMM": wPS_hMM}
    wPV_all = {"wPV_nPE": wPV_nPE, "wPV_pPE": wPV_pPE, "wPV_dMM": wPV_dMM, "wPV_hMM": wPV_hMM}
    
    return alpha_all, wPS_all, wPV_all
    

def DeterminePhasePlane_01(VE, VP, MP, wSV, wVS, num):
    
    ww = (1-wVS*wSV)
    wDS = ww/(1-wSV)
    
    print('External input to dendrites might be necessary!')
    
    # nPE neurons
    alpha_nPE_bound_1 = (VP + wSV*MP)/(VE + wSV)
    alpha_nPE_bound_2 = (MP + wVS*VP)/(wVS*VE - wSV * (1-wVS)/(1-wSV))
    alpha_nPE_max = min(alpha_nPE_bound_1 if alpha_nPE_bound_1 > 0 else 1000,
                        alpha_nPE_bound_2 if alpha_nPE_bound_2 > 0 else 1000)
    
    alpha_nPE = np.linspace(0, alpha_nPE_max, num)
    wPS_nPE = MP + wVS*VP - (wVS*VE - wSV * (1-wVS)/(1-wSV))*alpha_nPE
    wPV_nPE = wSV*wPS_nPE - ww * ((VE + wSV/(1-wSV)) * alpha_nPE - VP)
    
    # pPE neurons
    if VE>0:
        alpha_pPE_bound_1 = (VP + wSV*MP)/VE
        alpha_pPE_bound_2 = (MP + wVS*VP)/(wVS*VE)
        alpha_pPE_max = min(alpha_pPE_bound_1 if alpha_pPE_bound_1 > 0 else 1000,
                            alpha_pPE_bound_2 if alpha_pPE_bound_2 > 0 else 1000)
        alpha_pPE = np.linspace(0, alpha_pPE_max, num, dtype=dtype)
    else:
        alpha_pPE = np.linspace(0, 2.8, num, dtype=dtype) # no actual maximum (just for visualization)
    wPS_pPE = MP + wVS*VP - wVS*VE*alpha_pPE
    wPV_pPE = (wPS_pPE - ww*MP)/wVS
    
    # dMM neurons
    alpha_dMM_once = alpha_nPE
    alpha_dMM = np.repeat(alpha_dMM_once, num) 
    wPS_dMM_upper = wDS * (MP+VP-alpha_dMM_once*VE)
    wPS_dMM_bound_1 = MP + wVS*VP - alpha_dMM_once*wVS*VE
    wPS_dMM_bound_2 = MP + wVS*VP - alpha_dMM_once*wVS*VE + (1-wVS)/(1-wSV) * wSV * alpha_dMM_once
    wPS_dMM_lower = np.maximum(wPS_dMM_bound_1, wPS_dMM_bound_2)
    wPS_dMM = np.ravel(np.linspace(wPS_dMM_lower, wPS_dMM_upper, num, dtype=dtype), order='F')  
    wPV_dMM = (ww * (MP + VP - alpha_dMM*VE) - wPS_dMM * (1-wSV))/(1-wVS)
    
    # hMM neurons
    alpha_hMM_once = alpha_pPE
    alpha_hMM = np.repeat(alpha_hMM_once, num) 
    wPS_hMM_bound_0 = wDS * (MP+VP-alpha_hMM_once*VE)
    wPS_hMM_bound_1 = MP + wVS*VP - alpha_hMM_once*wVS*VE
    wPS_hMM_bound_2 = MP + wVS*VP - alpha_hMM_once*wVS*VE + (1-wVS)/(1-wSV) * wSV * alpha_hMM_once
    wPS_hMM_upper = np.minimum(wPS_hMM_bound_0, wPS_hMM_bound_1, wPS_hMM_bound_2)
    wPS_hMM = np.ravel(np.linspace(0, wPS_hMM_upper, num, dtype=dtype), order='F')  
    wPV_hMM = (ww * (MP + VP - alpha_hMM*VE) - wPS_hMM * (1-wSV))/(1-wVS)
    
    alpha_all = {"alpha_nPE": alpha_nPE, "alpha_pPE": alpha_pPE, "alpha_dMM": alpha_dMM, "alpha_hMM": alpha_hMM}
    wPS_all = {"wPS_nPE": wPS_nPE, "wPS_pPE": wPS_pPE, "wPS_dMM": wPS_dMM, "wPS_hMM": wPS_hMM}
    wPV_all = {"wPV_nPE": wPV_nPE, "wPV_pPE": wPV_pPE, "wPV_dMM": wPV_dMM, "wPV_hMM": wPV_hMM}
    
    return alpha_all, wPS_all, wPV_all