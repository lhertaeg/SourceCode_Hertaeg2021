#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 14 2021

@author: loreen.hertaeg
"""

# %% import packages

import numpy as np
from Default_Parameters import Default_Connectivity, Default_LearningRates

dtype = np.float32

# %% Functions
    
def RateDynamics(tau_inv_E, tau_inv_I, W, r, Stim, dt):
    
    # Initialise
    N = len(r)
    r0 = r.copy()
    dr1 = np.zeros(N, dtype=dtype)
    dr2 = np.zeros(N, dtype=dtype)
    
    # RK 2nd order
    dr1 = -r0 + W @ r0 + Stim
    dr1[:4] *= tau_inv_E 
    dr1[4:] *= tau_inv_I 

    r0[:4] += dt * dr1[:4]
    r0[4:] += dt * dr1[4:]
    
    dr2 = -r0 + W @ r0 + Stim
    dr2[:4] *= tau_inv_E 
    dr2[4:] *= tau_inv_I
    
    r += dt/2 * (dr1 + dr2)
    
    # Rectify
    r[r<0] = 0

    return


def Test_Static_MFN(W, VS, VV, stim_low, stim_high, stim_SD, stim_duration, fixed_input, tau_inv, dt, folder, fln):
    
    tau_inv_E, tau_inv_I  = tau_inv
    neurons_visual = np.array([1, 1, 0, 0, 1, 0, VS, VV])
    neurons_motor = np.array([0, 0, 1, 1, 0, 1, 1-VS, 1-VV])
    
    stim_motor = np.array([0, stim_high, 0, stim_high, 0, stim_low, 0], dtype=dtype)
    stim_visual = np.array([0, stim_high, 0, stim_low, 0, stim_high, 0], dtype=dtype)
    
    N = len(neurons_visual)
    NStim = len(stim_visual) 
    num_time_steps = np.int32(stim_duration/dt)
    
    r = np.zeros(N, dtype=dtype)
    noise = np.zeros((N,num_time_steps),dtype=dtype)
    
    path = 'Results/Data/' + folder
    fp = open(path +'/Data_StaticNetwork_MFN_' + str(fln) + '.dat','w')   
    
    # main loop
    for s in range(NStim):
        
        V = stim_visual[s]
        M = stim_motor[s]
        noise[:] = np.random.normal(0, stim_SD, size=(N, num_time_steps))
        
        for tstep in range(num_time_steps):
              
            # Numerical integration of mean-field rate dynamics
            Stim = fixed_input + V * neurons_visual + M * neurons_motor + noise[:,tstep]
            RateDynamics(tau_inv_E, tau_inv_I, W, r, Stim, dt)
            
            # write in file
            fp.write("%f" % (s*stim_duration + (tstep+1)*dt))
            for i in range(N):
                fp.write(" %f" % r[i])
            fp.write("\n") 
        
    fp.closed
    return
 

def MeanField_Connectivity():
    
    # default parameters 
    weights_mean, _, _, _ = Default_Connectivity()
    
    # create a weight matrix that accounts for 2 types of PE neurons (nPE, pPE)
    # and two types of PVs (PV recieving V and PV receiving M)
    
    # Order: nPE, pPE, Pv, Pm, S, V
    Weights = np.repeat(weights_mean, [2, 2, 2, 1, 1], axis=0)
    Weights = np.repeat(Weights, [2, 2, 2, 1, 1], axis=1)
    
    # Ensure that total weights are in line with values given in weights_mean
    Weights[:,:2] *= 0.5
    Weights[0,3] = 0
    Weights[1,2] = 0
    Weights[:,4:6] *= 0.5    
    Weights[2,6] = -0.5
    Weights[3,6] = -0.5
    
    return Weights


def Define_Weights_Optimized():
    
    # default learning rates (non-zero elements define the weights 
    # that will be optimized to satisfy the balance equations)
    eta_mean, _ = Default_LearningRates()
    
    # create a matrix that accounts for 2 types of PE neurons (nPE, pPE)
    # and two types of PVs (PV recieving V and PV receiving M)
    optimize_flag = 1*(eta_mean!=0)
    optimize_flag = np.repeat(optimize_flag, [2, 2, 2, 1, 1], axis=0)
    optimize_flag = np.repeat(optimize_flag, [2, 2, 2, 1, 1], axis=1)
    
    # Reduce to number of balance equations that need to be satisfied
    optimize_flag[0,5] = 0
    optimize_flag[1,4] = 0
    optimize_flag[4,7] = 0
    optimize_flag[5,6] = 0
    
    return optimize_flag
    
  
