"""
PE mean field networks
"""

# %% Import

import numpy as np
import pickle

from Default_Parameters import Default_NeuronPara
from Functions_Save import Create_PathFilePickle, SaveInput2PC_MFN
from Functions_Analysis import Solve_Balance_PE, Activity_Test_MFN, Inputs_to_PCs_MFN
from Functions_MeanFieldNet import MeanField_Connectivity, Define_Weights_Optimized, Test_Static_MFN

dtype = np.float32

# %% Functions 

def run_balance(folder, VS, VV):
    
    ### Data structure for storage 
    fln_data = 'Data_Optimal_Parameters_MFN_' + str(VS) + str(VV)
    
    Weights = MeanField_Connectivity()
    optimize_flag = Define_Weights_Optimized()
    
    path_data, filename_data = Create_PathFilePickle(folder, fln_data)

    ### Solve equations for heterogeneous PE circuits
    Weights[optimize_flag!=0], fopt = Solve_Balance_PE(Weights, optimize_flag, VS, VV, path_data, filename_data, 1)

    if fopt>1e-5:

        optimize_flag[0,5] = 1
        optimize_flag[1,4] = 1
        optimize_flag[4,7] = 1
        optimize_flag[5,6] = 1
    
        Weights[optimize_flag!=0], fopt = Solve_Balance_PE(Weights, optimize_flag, VS, VV, path_data, filename_data, 1) 
        
    ### run simulations to extract excitatory and inhibitory inputs into excitatory neurons
    r_target = dtype([1, 1, 0, 0, 4, 4, 4, 4])
    fln_inputs = fln_data + '_Inputs' 
    
    # Network architecture, initialise parameters and input structure
    path_data, filename_data = Create_PathFilePickle(folder, fln_data)
    
    with open(filename_data,'rb') as f:
        xopt, Weights, optimize_flag, _, _ = pickle.load(f)
        
    Weights[optimize_flag!=0] = xopt
    _, tau_E_inv, tau_I_inv = Default_NeuronPara()
    
    # Run static mean-field network
    stim_low, stim_high, stim_SD = 0, 1, 1e-5 
    stim_duration, dt = dtype(1000), dtype(0.1)
    fixed_input = (np.eye(8) - Weights) @ r_target
    
    Test_Static_MFN(Weights, VS, VV, stim_low, stim_high, stim_SD, stim_duration, fixed_input, 
                    [tau_E_inv, tau_I_inv], dt, folder, fln_inputs)
    
    [E_nE_phases, I_nE_phases, E_pE_phases, I_pE_phases, 
     E_nD_phases, I_nD_phases, E_pD_phases, I_pD_phases] = Inputs_to_PCs_MFN(folder, fln_inputs, Weights, stim_duration, stim_high)
    
    SaveInput2PC_MFN(Weights, stim_duration, stim_high, E_nE_phases, I_nE_phases, E_pE_phases, 
                     I_pE_phases, E_nD_phases, I_nD_phases, E_pD_phases, I_pD_phases, folder, fln_inputs)
    
        

def run_example(folder, VS, VV):

    r_target = dtype([1, 1, 0, 0, 4, 4, 4, 4])
    
    ### Data structure for storage and loading
    fln_load = 'Data_Optimal_Parameters_MFN_' + str(VS) + str(VV)
    fln_save = fln_load + '_Example'
    
    ### Network architecture, initialise parameters and input structure
    path_data, filename_data = Create_PathFilePickle(folder, fln_load)
    
    with open(filename_data,'rb') as f:
        xopt, Weights, optimize_flag, _, _ = pickle.load(f)
        
    Weights[optimize_flag!=0] = xopt
    _, tau_E_inv, tau_I_inv = Default_NeuronPara()
    
    ### Run static mean-field network
    stim_low, stim_high, stim_SD = 0, 5, 1
    stim_duration, dt = dtype(1000), dtype(0.1)
    fixed_input = (np.eye(8) - Weights) @ r_target
    
    Test_Static_MFN(Weights, VS, VV, stim_low, stim_high, stim_SD, stim_duration, fixed_input, 
                    [tau_E_inv, tau_I_inv], dt, folder, fln_save)
    
            

def run_difference(folder, VS, VV):
    
    ### Data structure for plotting
    fln_save4plot = 'Data_PE_MFN_' + str(VS) + str(VV)
    _, filename_save4plot = Create_PathFilePickle(folder, fln_save4plot)
    
    ### Data structure for storage and loading
    fln_load = 'Data_Optimal_Parameters_MFN_' + str(VS) + str(VV)
    fln_save = fln_load + '_Temporary'
    
    ### Network architecture, initialise parameters and input structure
    r_target = dtype([1, 1, 0, 0, 4, 4, 4, 4])
    path_data, filename_data = Create_PathFilePickle(folder, fln_load)
    
    with open(filename_data,'rb') as f:
        xopt, Weights, optimize_flag, _, _ = pickle.load(f)
        
    Weights[optimize_flag!=0] = xopt
    _, tau_E_inv, tau_I_inv = Default_NeuronPara()
    
    ### Input parameters
    stim_high, stim_SD = dtype(5), dtype(1)
    stim_duration, dt = dtype(1000), dtype(0.1)
    fixed_input = (np.eye(8) - Weights) @ r_target
    
    stim_low_all = np.linspace(0, 5, 6, dtype=dtype) 
    RE_MM = np.zeros((2,len(stim_low_all)),dtype=dtype)
    RE_PB = np.zeros((2,len(stim_low_all)),dtype=dtype)
    
    for i in range(len(stim_low_all)):
        
        stim_low = stim_low_all[i]
        
        Test_Static_MFN(Weights, VS, VV, stim_low, stim_high, stim_SD, stim_duration, fixed_input, 
                        [tau_E_inv, tau_I_inv], dt, folder, fln_save)
        
        R_phases = Activity_Test_MFN(folder, fln_save, stim_duration)
        RE_MM[:,i] = R_phases[:2,2] - R_phases[:2,0]
        RE_PB[:,i] = R_phases[:2,3] - R_phases[:2,0]
    
    ### Save data for plotting purposes
    with open(filename_save4plot,'wb') as f:
                pickle.dump([stim_low_all, RE_MM, RE_PB],f)
    

def run_perturbation(folder_load, folder_save, VS, VV):
    
    ### Network architecture, neuron parameters and input structure
    fln_load = 'Data_Optimal_Parameters_MFN_' + str(VS) + str(VV)
    _, filename_data = Create_PathFilePickle(folder_load, fln_load)
    
    with open(filename_data,'rb') as f:
        xopt, Weights, optimize_flag, _, _ = pickle.load(f)
        
    Weights[optimize_flag!=0] = xopt
    _, tau_E_inv, tau_I_inv = Default_NeuronPara()
    
    ### Input & stimulation parameters
    stim_low, stim_high, stim_SD = 0, 3, 1
    stim_duration, dt = dtype(1000), dtype(0.1)
    stim_extra_all = np.linspace(-1.5,1.5,11)
    
    r_target = dtype([1, 1, 0, 0, 4, 4, 4, 4])
    fixed_input = (np.eye(8) - Weights) @ r_target
    Input_soma_phases = np.array([0, stim_high, 0, stim_high], dtype=dtype)
    
    ### For readability ...
    wnED, wpED = Weights[0,2], Weights[1,3]
    wnEPv, wnEPm = Weights[0,4:6]
    wpEPv, wpEPm = Weights[1,4:6]
    
    ### Data structure for data storage
    fln_save = fln_load + '_Temporary'
    fln_save4plot = 'Data_Manipulation_MFN_' + str(VS) + str(VV)
    _, filename_save4plot = Create_PathFilePickle(folder_save, fln_save4plot)
    
    ### Run 
    N = len(r_target)
    Inputs_nPE = np.zeros((4,len(stim_extra_all),N))
    Inputs_pPE = np.zeros((4,len(stim_extra_all),N))
    
    for j in range(len(stim_extra_all)):
        
        stim_extra = stim_extra_all[j]
        print('Manipulation: ' + str(stim_extra))
    
        # manipulations
        for i in range(N):
            
            print('Target: ' + str(i))
            
            manipulation = np.zeros(N)
            manipulation[i] = stim_extra
            fixed_input_manipulated = fixed_input + manipulation
            
            Test_Static_MFN(Weights, VS, VV, stim_low, stim_high, stim_SD, stim_duration, fixed_input_manipulated, 
                            [tau_E_inv, tau_I_inv], dt, folder_save, fln_save)
            
            R_phases = Activity_Test_MFN(folder_save, fln_save, stim_duration)
        
            Rates_Phases_nD = R_phases[2,:] 
            Rates_Phases_pD = R_phases[3,:] 
            Rates_Phases_PVv = R_phases[4,:] 
            Rates_Phases_PVm = R_phases[5,:]  
                        
            input_nPE = fixed_input[0] + Input_soma_phases
            input_pPE = fixed_input[1] + Input_soma_phases
            Inputs_nPE[:,j,i] = input_nPE + wnED * Rates_Phases_nD + wnEPv * Rates_Phases_PVv + wnEPm * Rates_Phases_PVm
            Inputs_pPE[:,j,i] = input_pPE + wpED * Rates_Phases_pD + wpEPv * Rates_Phases_PVv + wpEPm * Rates_Phases_PVm
                
    ### Save data
    with open(filename_save4plot,'wb') as f:
        pickle.dump([stim_extra_all, Inputs_nPE, Inputs_pPE],f)