#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 08:34:59 2021

@author: loreen.hertaeg
"""

# %% import packages

import numpy as np
import os
from numba import njit
from typing import NamedTuple

dtype = np.float32

# %% Classes

class Neurons(NamedTuple):
        
        NCells: list = np.array([280,40,40,40], dtype=np.int32)
        tau_inv_E: dtype = dtype(1.0/60.0)
        tau_inv_I: dtype = dtype(1.0/2.0)
        
        
class Network:
    def __init__(self, Neurons, NetPar_PE, mBE, mBB, mLE, m_out, bool_nPE, bool_pPE, neurons_motor):
        
        # Pre-processing
        NE, NP, NS, NV = np.int32(Neurons.NCells/2)
        N_nPE = sum(bool_nPE)
        N_pPE = sum(bool_pPE)
        
        P_Pred_flag = neurons_motor[NE:(NE+NP)]
        S_Pred_flag = neurons_motor[(NE+NP):(NE+NP+NS)]
        V_Pred_flag = neurons_motor[(NE+NP+NS):]
        
        # PE network connectivity (PE-PE)
        wEP = NetPar_PE.wEP
        wED = NetPar_PE.wED
        wDE = NetPar_PE.wDE
        wDS = NetPar_PE.wDS 
        wPE = NetPar_PE.wPE
        wPP = NetPar_PE.wPP
        wPS = NetPar_PE.wPS
        wPV = NetPar_PE.wPV
        wSE = NetPar_PE.wSE
        wSP = NetPar_PE.wSP
        wSS = NetPar_PE.wSS
        wSV = NetPar_PE.wSV
        wVE = NetPar_PE.wVE
        wVP = NetPar_PE.wVP
        wVS = NetPar_PE.wVS
        wVV = NetPar_PE.wVV
        
        self.wEP = np.block([[wEP, np.zeros((NE,NP),dtype=dtype)],[np.zeros((NE,NP),dtype=dtype), wEP]])
        self.wED = np.block([[wED, np.zeros((NE,NE),dtype=dtype)],[np.zeros((NE,NE),dtype=dtype), wED]])
        self.wDE = np.block([[wDE, np.zeros((NE,NE),dtype=dtype)],[np.zeros((NE,NE),dtype=dtype), wDE]])
        self.wDS = np.block([[wDS, np.zeros((NE,NS),dtype=dtype)],[np.zeros((NE,NS),dtype=dtype), wDS]])
        self.wPE = np.block([[wPE, np.zeros((NP,NE),dtype=dtype)],[np.zeros((NP,NE),dtype=dtype), wPE]])
        self.wPP = np.block([[wPP, np.zeros((NP,NP),dtype=dtype)],[np.zeros((NP,NP),dtype=dtype), wPP]])
        self.wPS = np.block([[wPS, np.zeros((NP,NS),dtype=dtype)],[np.zeros((NP,NS),dtype=dtype), wPS]])
        self.wPV = np.block([[wPV, np.zeros((NP,NV),dtype=dtype)],[np.zeros((NP,NV),dtype=dtype), wPV]])
        self.wSE = np.block([[wSE, np.zeros((NS,NE),dtype=dtype)],[np.zeros((NS,NE),dtype=dtype), wSE]])
        self.wSP = np.block([[wSP, np.zeros((NS,NP),dtype=dtype)],[np.zeros((NS,NP),dtype=dtype), wSP]])
        self.wSS = np.block([[wSS, np.zeros((NS,NS),dtype=dtype)],[np.zeros((NS,NS),dtype=dtype), wSS]])
        self.wSV = np.block([[wSV, np.zeros((NS,NV),dtype=dtype)],[np.zeros((NS,NV),dtype=dtype), wSV]])
        self.wVE = np.block([[wVE, np.zeros((NV,NE),dtype=dtype)],[np.zeros((NV,NE),dtype=dtype), wVE]])
        self.wVP = np.block([[wVP, np.zeros((NV,NP),dtype=dtype)],[np.zeros((NV,NP),dtype=dtype), wVP]])
        self.wVS = np.block([[wVS, np.zeros((NV,NS),dtype=dtype)],[np.zeros((NV,NS),dtype=dtype), wVS]])
        self.wVV = np.block([[wVV, np.zeros((NV,NV),dtype=dtype)],[np.zeros((NV,NV),dtype=dtype), wVV]])
        
        # Connectivity between P and PE (PE-P)
        wBE = np.zeros((2,2*NE), dtype=dtype)
        wBE[0,:NE] = mBE * bool_pPE/N_nPE - mBE * bool_nPE/N_pPE
        wBE[1,NE:] = mBE * bool_pPE/N_nPE - mBE * bool_nPE/N_pPE 
        
        wDB = np.zeros((2*NE,2), dtype=dtype)
        wDB[:NE,0] = 1
        wDB[NE:,1] = 1
        
        wPB = np.zeros((2*NP,2), dtype=dtype)
        wPB[:NP,0] = 1 * P_Pred_flag
        wPB[NP:,1] = 1 * P_Pred_flag
        
        wSB = np.zeros((2*NS,2), dtype=dtype)
        wSB[:NS,0] = 1 * S_Pred_flag
        wSB[NS:,1] = 1 * S_Pred_flag
        
        wVB = np.zeros((2*NV,2), dtype=dtype)
        wVB[:NV,0] = 1 * V_Pred_flag
        wVB[NV:,1] = 1 * V_Pred_flag
        
        self.wBE = wBE 
        self.wEB = np.zeros((2*NE,2), dtype=dtype)
        self.wDB = wDB 
        self.wPB = wPB 
        self.wSB = wSB 
        self.wVB = wVB 
        
        # Connectivity between line attractor and PE (L-PE)
        wLE = np.zeros((2,2*NE), dtype=dtype)
        if not isinstance(mLE, np.ndarray):
            wLE[0,:NE] = mLE * bool_pPE/N_nPE - mLE * bool_nPE/N_pPE
            wLE[1,NE:] = mLE * bool_pPE/N_nPE - mLE * bool_nPE/N_pPE 
        else:
            wLE[0,:NE] = mLE[0] * bool_pPE/N_nPE - mLE[0] * bool_nPE/N_pPE
            wLE[0,NE:] = mLE[1] * bool_pPE/N_nPE - mLE[1] * bool_nPE/N_pPE
            wLE[1,:NE] = mLE[1] * bool_pPE/N_nPE - mLE[1] * bool_nPE/N_pPE
            wLE[1,NE:] = mLE[0] * bool_pPE/N_nPE - mLE[0] * bool_nPE/N_pPE
        
        self.wLE = wLE

        # Connectivity between P and line attractor network (L-P)
        self.wBL = np.eye(2, dtype=dtype)
            
        # Connectivity between P and P 
        if not isinstance(mBB, np.ndarray):
            self.wBB = np.array([[0, mBB],[mBB, 0]], dtype=dtype)
        else:
            self.wBB = np.array([[0, mBB[0]],[mBB[1], 0]], dtype=dtype)
        
        # Connectivity between P and  Output neurons
        self.w_out_B  = np.eye(2, dtype=dtype) # not used anymore 
        
        # Connectivty between PE and Output neurons
        w_out_E = np.zeros((2,2*NE), dtype=dtype)
        w_out_E[0,:NE] = - m_out * bool_pPE/N_nPE + m_out * bool_nPE/N_pPE
        w_out_E[1,NE:] = - m_out * bool_pPE/N_nPE + m_out * bool_nPE/N_pPE 
        
        self.w_out_E = w_out_E
        
        
class InputStructure:
    def __init__(self, Neurons, InPar, cue_flag = 1):
            
        Xsoma = InPar.inp_ext_soma
        Xdend = InPar.inp_ext_dend 
        
        self.cue_flag = cue_flag
        
        # Assign selectivity (basically the "columns")
        NCells = Neurons.NCells
        NE, NP, NS, NV = np.int32(NCells/2)
       
        self.Selectivity_E = np.concatenate((np.zeros(NE), np.ones(NE)))
        self.Selectivity_D = np.concatenate((np.zeros(NE), np.ones(NE)))
        self.Selectivity_P = np.concatenate((np.zeros(NP), np.ones(NP)))
        self.Selectivity_S = np.concatenate((np.zeros(NS), np.ones(NS)))
        self.Selectivity_V = np.concatenate((np.zeros(NV), np.ones(NV)))
       
        # External, fixed inputs
        inp_ext_soma = np.zeros(sum(NCells))
        inp_ext_dend = np.zeros(NCells[0])
        
        inp_ext_soma[:2*NE] = np.tile(Xsoma[:NE],2)
        inp_ext_soma[2*NE:2*(NE+NP)] = np.tile(Xsoma[NE:(NE+NP)],2)
        inp_ext_soma[2*(NE+NP):2*(NE+NP+NS)] = np.tile(Xsoma[(NE+NP):(NE+NP+NS)],2)
        inp_ext_soma[2*(NE+NP+NS):] = np.tile(Xsoma[(NE+NP+NS):],2)
        inp_ext_dend = np.tile(Xdend,2)
        
        self.inp_ext_soma: dtype = inp_ext_soma
        self.inp_ext_dend: dtype = inp_ext_dend 
        
        # Flag to indicate which neurons receive stimulus (visual)
        ind_break = np.cumsum(np.int32(NCells/2),dtype=np.int32)[:-1]
        neurons_visual_P, neurons_visual_S, neurons_visual_V = np.split(InPar.neurons_visual, ind_break)[1:]
        
        self.Flag_visual_P = np.tile(neurons_visual_P, 2)
        self.Flag_visual_S = np.tile(neurons_visual_S, 2)
        self.Flag_visual_V = np.tile(neurons_visual_V, 2)
        
        
class Stimulation:
    def __init__(self, distribution_flag, stimuli = [], uniform_dist_1 = np.array([1,2,3,4,5]), 
                 uniform_dist_2 = np.array([5,6,7,8,9]), SD = dtype(1)):
        
        self.SD: dtype = SD
        self.distribution_flag = distribution_flag
        self.overlap = np.intersect1d(uniform_dist_1,uniform_dist_2)
        
        if len(stimuli)==0:
            
            N_stim = len(distribution_flag)
            N_uni_1 = len(uniform_dist_1)
            N_uni_2 = len(uniform_dist_2)
            
            rand_num_1 = np.random.choice(uniform_dist_1, size=N_stim, p=np.ones(N_uni_1)/N_uni_1)
            rand_num_2 = np.random.choice(uniform_dist_2, size=N_stim, p=np.ones(N_uni_2)/N_uni_2)
            
            stimuli = rand_num_1 * (distribution_flag==0) + rand_num_2 * (distribution_flag==1)

        self.stimuli = stimuli

        
class Activity_Zero:
    def __init__(self, NeuPar, InPar, StimPar,  P1, P2, r0 = dtype([0,0,0,0,0])):

        NCells = NeuPar.NCells
        Nb = np.cumsum(NCells, dtype=np.int32)
    
        if len(r0)<sum(NCells):
            self.rE0 = np.repeat(r0[0],NCells[0])
            self.rP0 = np.repeat(r0[1],NCells[1])
            self.rS0 = np.repeat(r0[2],NCells[2])
            self.rV0 = np.repeat(r0[3],NCells[3])
            self.rD0 = np.repeat(r0[4],NCells[0])
        else:
            self.rE0, self.rP0, self.rS0, self.rV0 , self.rD0  = np.split(r0,Nb)
        
        # if InPar.cue_flag==1:
        #     distribution_flag = StimPar.distribution_flag
        #     self.rB0 = np.array([P1*(distribution_flag[0]==0), P2*(distribution_flag[0]==1)], dtype=dtype) 
        # else:
        #     self.rB0 = np.array([0, 0], dtype=dtype)
        
        distribution_flag = StimPar.distribution_flag
        self.rB0 = np.array([P1*(distribution_flag[0]==0), P2*(distribution_flag[0]==1)], dtype=dtype) 
        
        self.rL0 = np.array([P1, P2], dtype=dtype)   

        
class Simulation(NamedTuple): 
       
       dt: dtype = dtype(0.1)
       stim_duration: dtype = dtype(1000.0) 


# %% Functions

@njit(cache=True)
def drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV,
         wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV, wLE, wBB, wBE, wBL,
         wEB, wDB, wPB, wSB, wVB, rE, rD, rP, rS, rV, rL, rB,
         Cue, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V): 
    
    # Line attractor neurons
    drL = tau_inv_E * (wLE @ rE)
    
    # Prediction neurons
    drB = tau_inv_E * (-rB + wBB @ rB + wBE @ rE + wBL @ rL + Cue)
    
    # PE network
    drE = tau_inv_E * (-rE + wED @ rD + wEP @ rP + wEB @ rB + Stim_E)
    drD = tau_inv_E * (-rD + wDE @ rE + wDS @ rS + wDB @ rB + Stim_D)
    drP = tau_inv_I * (-rP + wPE @ rE + wPP @ rP + wPS @ rS + wPV @ rV + wPB @ rB + Stim_P)
    drS = tau_inv_I * (-rS + wSE @ rE + wSP @ rP + wSS @ rS + wSV @ rV + wSB @ rB + Stim_S)
    drV = tau_inv_I * (-rV + wVE @ rE + wVP @ rP + wVS @ rS + wVV @ rV + wVB @ rB + Stim_V)
    
    return drE, drD, drP, drS, drV, drL, drB


def NetworkDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV,
                    wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV, wLE, wBB, wBE, wBL,
                    wEB, wDB, wPB, wSB, wVB, rE, rD, rP, rS, rV, rL, rB,
                    Cue, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V, dt):
    
    rE0 = rE.copy()
    rD0 = rD.copy()
    rP0 = rP.copy()
    rS0 = rS.copy()
    rV0 = rV.copy()
    rL0 = rL.copy()
    rB0 = rB.copy()
    
    drE1, drD1, drP1, drS1, drV1, drL1, drB1 = drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV,
                                                    wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV, wLE, wBB, wBE, wBL,
                                                    wEB, wDB, wPB, wSB, wVB, rE0, rD0, rP0, rS0, rV0, rL0, rB0,
                                                    Cue, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V)
        
    rE0[:] += dt * drE1
    rD0[:] += dt * drD1
    rP0[:] += dt * drP1
    rS0[:] += dt * drS1
    rV0[:] += dt * drV1
    rL0[:] += dt * drL1
    rB0[:] += dt * drB1
    
    drE2, drD2, drP2, drS2, drV2, drL2, drB2 = drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV,
                                                    wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV, wLE, wBB, wBE, wBL,
                                                    wEB, wDB, wPB, wSB, wVB, rE0, rD0, rP0, rS0, rV0, rL0, rB0,
                                                    Cue, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V)
    
    
    rE[:] += dt/2 * (drE1 + drE2)
    rD[:] += dt/2 * (drD1 + drD2)
    rP[:] += dt/2 * (drP1 + drP2) 
    rS[:] += dt/2 * (drS1 + drS2) 
    rV[:] += dt/2 * (drV1 + drV2) 
    rL[:] += dt/2 * (drL1 + drL2)
    rB[:] += dt/2 * (drB1 + drB2)
    
    rE[rE<0] = 0
    rP[rP<0] = 0
    rS[rS<0] = 0
    rV[rV<0] = 0
    rD[rD<0] = 0
    rL[rL<0] = 0
    rB[rB<0] = 0

    return


def RunPredNet(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder: str, fln: str = ''):
    
    ### Neuron parameters for PE network
    NCells = NeuPar.NCells
    ind_break = np.cumsum(NCells[1:],dtype=np.int32)[:-1]
    
    tau_inv_E = NeuPar.tau_inv_E
    tau_inv_I = NeuPar.tau_inv_I
    
    ### Network parameters
    wEP = NetPar.wEP
    wED = NetPar.wED
    wDE = NetPar.wDE
    wDS = NetPar.wDS 
    wPE = NetPar.wPE
    wPP = NetPar.wPP
    wPS = NetPar.wPS
    wPV = NetPar.wPV
    wSE = NetPar.wSE
    wSP = NetPar.wSP
    wSS = NetPar.wSS
    wSV = NetPar.wSV
    wVE = NetPar.wVE
    wVP = NetPar.wVP
    wVS = NetPar.wVS
    wVV = NetPar.wVV
    wLE = NetPar.wLE
    wBB = NetPar.wBB
    wBE = NetPar.wBE
    wBL = NetPar.wBL
    wEB = NetPar.wEB
    wDB = NetPar.wDB
    wPB = NetPar.wPB
    wSB = NetPar.wSB
    wVB = NetPar.wVB
    
    ## Stimulation protocol & Inputs
    SD = StimPar.SD
    stimuli = iter(StimPar.stimuli)
    overlap = StimPar.overlap
    distribution_flag = StimPar.distribution_flag
    NStim = np.int32(len(StimPar.stimuli))
    
    inp_ext_soma = InPar.inp_ext_soma
    inp_ext_dend = InPar.inp_ext_dend
    
    inp_ext_E = inp_ext_soma[:NCells[0]]
    inp_ext_P, inp_ext_S, inp_ext_V = np.split(inp_ext_soma[NCells[0]:], ind_break)
    inp_ext_D = inp_ext_dend
    
    Selectivity_E = InPar.Selectivity_E
    Selectivity_D = InPar.Selectivity_D
    Selectivity_P = InPar.Selectivity_P
    Selectivity_S = InPar.Selectivity_S
    Selectivity_V = InPar.Selectivity_V
    cue_flag = InPar.cue_flag
    
    Flag_visual_P = InPar.Flag_visual_P
    Flag_visual_S = InPar.Flag_visual_S
    Flag_visual_V = InPar.Flag_visual_V
      
    ### Simulation parameters
    dt = SimPar.dt
    stim_duration = SimPar.stim_duration
    num_time_steps = np.int32(stim_duration/dt)    
    
    ### Initial activity levels
    rE = RatePar.rE0
    rD = RatePar.rD0
    rP = RatePar.rP0
    rS = RatePar.rS0
    rV = RatePar.rV0
    rL = RatePar.rL0
    rB = RatePar.rB0
    
    ### Initialisation
    Stim_E = np.zeros(NCells[0], dtype=dtype)
    Stim_P = np.zeros(NCells[1], dtype=dtype)
    Stim_S = np.zeros(NCells[2], dtype=dtype)
    Stim_V = np.zeros(NCells[3], dtype=dtype)
    Stim_D = np.zeros(NCells[0], dtype=dtype)
    
    Stim_Mean_E = np.zeros(NCells[0], dtype=dtype)
    Stim_Mean_P = np.zeros(NCells[1], dtype=dtype)
    Stim_Mean_S = np.zeros(NCells[2], dtype=dtype)
    Stim_Mean_V = np.zeros(NCells[3], dtype=dtype)
    Stim_Mean_D = np.zeros(NCells[0], dtype=dtype)
    
    noise_E = np.zeros((NCells[0], num_time_steps), dtype=dtype)
    noise_P = np.zeros((NCells[1], num_time_steps), dtype=dtype)
    noise_S = np.zeros((NCells[2], num_time_steps), dtype=dtype)
    noise_V = np.zeros((NCells[3], num_time_steps), dtype=dtype)
    noise_D = np.zeros((NCells[0], num_time_steps), dtype=dtype)
    
    ### Path & file for data to be stored
    path = 'Results/Data/' + folder
    if not os.path.exists(path):
        os.mkdir(path)

    fp1 = open(path +'/Data_StaticPredNet_PE_' + fln + '.dat','w') 
    fp2 = open(path +'/Data_StaticPredNet_P_' + fln + '.dat','w') 
    
    ### Main loop
    for s in range(NStim):
        
        print('Stimuli', str(s+1), '/', str(NStim))
        
        stim = next(stimuli)
        stim_type = distribution_flag[s]
        noise_E[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        noise_P[:] = np.random.normal(0,SD,size=(NCells[1], num_time_steps))
        noise_S[:] = np.random.normal(0,SD,size=(NCells[2], num_time_steps))
        noise_V[:] = np.random.normal(0,SD,size=(NCells[3], num_time_steps))
        noise_D[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        
        Cue = np.zeros((2, num_time_steps), dtype=dtype)
        if cue_flag==1:
            Cue[1-stim_type,:] = dtype(-10)
            
        if ((stim in overlap) and (cue_flag==0)):
            Stim_Mean_E[:] = inp_ext_E + stim 
            Stim_Mean_P[:] = inp_ext_P + stim * (Flag_visual_P==1)
            Stim_Mean_S[:] = inp_ext_S + stim * (Flag_visual_S==1)
            Stim_Mean_V[:] = inp_ext_V + stim * (Flag_visual_V==1)
        else:
            Stim_Mean_E[:] = inp_ext_E + stim * (Selectivity_E==stim_type)
            Stim_Mean_P[:] = inp_ext_P + stim * (Selectivity_P==stim_type) * (Flag_visual_P==1)
            Stim_Mean_S[:] = inp_ext_S + stim * (Selectivity_S==stim_type) * (Flag_visual_S==1)
            Stim_Mean_V[:] = inp_ext_V + stim * (Selectivity_V==stim_type) * (Flag_visual_V==1)
        
        Stim_Mean_D[:] = inp_ext_D
        
        for tstep in range(num_time_steps):

            Stim_E[:] = Stim_Mean_E + noise_E[:, tstep]
            Stim_D[:] = Stim_Mean_D + noise_D[:, tstep]
            Stim_P[:] = Stim_Mean_P + noise_P[:, tstep]
            Stim_S[:] = Stim_Mean_S + noise_S[:, tstep]
            Stim_V[:] = Stim_Mean_V + noise_V[:, tstep]
            
            NetworkDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV,
                             wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV, wLE, wBB, wBE, wBL,
                             wEB, wDB, wPB, wSB, wVB, rE, rD, rP, rS, rV, rL, rB,
                             Cue[:,tstep], Stim_E, Stim_D, Stim_P, Stim_S, Stim_V, dt)
            
            if (tstep % 10 == 0):
                fp1.write("%f" % (s * stim_duration + (tstep+1) * dt))
                for i in range(NCells[0]):
                    fp1.write(" %f" % rE[i])
                for i in range(NCells[1]):
                    fp1.write(" %f" % rP[i])
                for i in range(NCells[2]):
                    fp1.write(" %f" % rS[i])
                for i in range(NCells[3]):
                    fp1.write(" %f" % rV[i])
                for i in range(NCells[0]):
                    fp1.write(" %f" % rD[i])
                fp1.write("\n") 
                
                fp2.write("%f" % (s * stim_duration + (tstep+1) * dt))
                for i in range(2):
                    fp2.write(" %f" % rL[i])
                for i in range(2):
                    fp2.write(" %f" % rB[i])
                fp2.write("\n") 
                
    fp1.closed
    fp2.closed
    
    return