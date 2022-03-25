"""
PE networks, connected to attractor network
"""

# %% Import

import numpy as np

from Default_Parameters import Default_PredProcPara
from Functions_Save import LoadNetworkPara, LoadAnalysisActivity, SaveActivityPredProcNet, SaveActivityBiasTowardsMean
from Functions_Save import SaveLearningRate, SaveErrorEnvironments
from Functions_Analysis import Output_PredNet, Output_per_stimulus_PredNet, Output_ModLearnRate
from Functions_PredNet import RunPredNet, Neurons, Network, InputStructure, Stimulation, Simulation, Activity_Zero

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %%

def run_modulate_leaning(folder_data, folder, fln_save_data):
    
    fln_load_after = 'Example_Target_Input_After'
    
    m_out, mBE, mBB, mLE, w_pred, w_classic, eta, stimuli_weak, stimuli_strong = Default_PredProcPara()
    N_stimuli, N_ini = 2, 500 
    
    ### Load data
    NeuPar_PE, NetPar_PE, InPar_PE, _, _, _, _ = LoadNetworkPara(folder_data, fln_load_after)
    _, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder_data, fln_load_after)
    
    ### Neuron parameters
    NeuPar = Neurons()
    
    ### Network parameters
    NetPar = Network(NeuPar, NetPar_PE, mBE, mBB, mLE, m_out, bool_nPE, bool_pPE, InPar_PE.neurons_motor)
    
    ### Simulation parameters
    SimPar = Simulation()
    
    ### Input structure
    InPar = InputStructure(NeuPar, InPar_PE)
    
    mean_statistics_weak = np.mean(stimuli_weak)
    mean_statistics_strong = np.mean(stimuli_strong)
    
    ### Run network for all stimuli of distribution
    dR_mod  = np.zeros(len(stimuli_weak))
    dR_classic = np.zeros(len(stimuli_weak))
    
    for i in range(len(stimuli_weak)):
        
        distribution_flag = np.zeros(N_stimuli, dtype=np.int32)
        stimuli = np.zeros_like(distribution_flag)
        stimuli[:] = stimuli_weak[i]
          
        StimPar = Stimulation(distribution_flag, stimuli)
   
        ### Initial activity levels
        RatePar = Activity_Zero(NeuPar, InPar, StimPar, mean_statistics_weak, mean_statistics_strong)
        
        ### Run network 
        RunPredNet(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, fln_save_data)
            
        ### Analysis
        paras = [w_pred, w_classic, eta, N_ini]
        R_mod, R_classic, dR_mod[i], dR_classic[i] = Output_ModLearnRate(NeuPar, StimPar, bool_nPE, bool_pPE, 
                                                                         paras, folder, fln_save_data)
        
        ### Save data
        SaveLearningRate(R_mod, R_classic, dR_mod, dR_classic, folder, fln_save_data + '_' +  str(i))



def run_generalise(folder_data, folder, fln_save_data):
    
    fln_load_after = 'Example_Target_Input_After'
    fln_save_data_uncoupled = fln_save_data + '_1'
    fln_save_data_coupled = fln_save_data + '_2'
    
    m_out, mBE, mBB, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()

    ### Load data
    NeuPar_PE, NetPar_PE, InPar_PE, _, _, _, _ = LoadNetworkPara(folder_data, fln_load_after)
    _, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder_data, fln_load_after)
    
    ### Neuron parameters
    NeuPar = Neurons()
    
    ### Network parameters
    mLE = 1e-2,
    NetPar_uncoupled = Network(NeuPar, NetPar_PE, mBE, mBB, mLE, m_out, bool_nPE, bool_pPE, InPar_PE.neurons_motor)

    mLE = dtype([1e-2, 1e-2])
    NetPar_coupled = Network(NeuPar, NetPar_PE, mBE, mBB, mLE, m_out, bool_nPE, bool_pPE, InPar_PE.neurons_motor)
    
    ### Simulation parameters
    SimPar = Simulation() # stim_duration=1000.0
    
    ### Inputs 
    InPar = InputStructure(NeuPar, InPar_PE)
    
    mean_statistics_weak = np.mean(stimuli_weak)
    mean_statistics_strong = np.mean(stimuli_strong)
    
    ### Run for several distributions
    Ratio_AB = np.array([0.0,0.3], dtype=dtype) #np.linspace(0, 50, 6, dtype=np.int32)/100 
    Error_uncoupled = np.zeros((2,len(Ratio_AB)))
    Error_coupled = np.zeros((2,len(Ratio_AB)))
    
    for j in range(len(Ratio_AB)):
        
        ### Stimulation 
        ratio = Ratio_AB[j]
        N_stimuli =  15 + 2 # training + test
    
        stimuli = np.random.choice((stimuli_weak[-1],stimuli_strong[-1]), size=N_stimuli, p=[1-ratio, ratio])
        stimuli[-2:] = np.array([stimuli_weak[-1],stimuli_strong[-1]], dtype=dtype)
        distribution_flag = 0 * (stimuli==stimuli_weak[-1]) + 1 * (stimuli==stimuli_strong[-1])
  
        StimPar = Stimulation(distribution_flag, stimuli)

        ### Initial activity levels
        RatePar_coupled = Activity_Zero(NeuPar, InPar, StimPar, mean_statistics_weak, mean_statistics_strong)
        RatePar_uncoupled = Activity_Zero(NeuPar, InPar, StimPar, mean_statistics_weak, mean_statistics_strong) 
    
        ### Run network (Adaptation to new environement)
        RunPredNet(NeuPar, NetPar_uncoupled, InPar, StimPar, SimPar, RatePar_uncoupled, folder, fln_save_data_uncoupled)
        RunPredNet(NeuPar, NetPar_coupled, InPar, StimPar, SimPar, RatePar_coupled, folder, fln_save_data_coupled)
        
        ### Error
        Error_uncoupled[:,j] = 0.5 * (RatePar_uncoupled.rL0 - np.array([stimuli_weak[-1], stimuli_strong[-1]]))**2
        Error_coupled[:,j] = 0.5 * (RatePar_coupled.rL0 - np.array([stimuli_weak[-1], stimuli_strong[-1]]))**2
        
    ### Save data
    SaveErrorEnvironments(Ratio_AB, Error_coupled, Error_uncoupled, folder, fln_save_data)
    
    

def run_switching_attarctors(folder_data, folder, fln_save_data):
    
    fln_load_after = 'Example_Target_Input_After'
    m_out, mBE, mBB, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
        
    ### Load data
    NeuPar_PE, NetPar_PE, InPar_PE, _, _, _, _ = LoadNetworkPara(folder_data, fln_load_after)
    _, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder_data, fln_load_after)
    
    ### Neuron parameters
    NeuPar = Neurons()
    
    ### Network parameters
    NetPar = Network(NeuPar, NetPar_PE, mBE, mBB, mLE, m_out, bool_nPE, bool_pPE, InPar_PE.neurons_motor)
    
    ### Simulation parameters
    SimPar = Simulation()
    
    ### Stimulation protocol & Inputs
    InPar = InputStructure(NeuPar, InPar_PE, cue_flag = 0)
    
    mean_statistics_weak = np.mean(stimuli_weak)
    mean_statistics_strong = np.mean(stimuli_strong)
    
    N_stimuli =  25 # 10
    distribution_flag = np.zeros(N_stimuli, dtype=np.int32)
    distribution_flag = np.array(([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0]),dtype=dtype)
    
    stimuli = np.zeros_like(distribution_flag)
    stimuli = np.array(([1,3,4,5,2,2,4,6,7,7,9,5,8,6,4,2,3,1,8,7,6,5,2,1,2]),dtype=dtype)
     
    StimPar = Stimulation(distribution_flag, stimuli)
    
    num_time_steps = np.int32(SimPar.stim_duration/SimPar.dt/10) # because r is recorded every 10 steps
    stimulus_sequence = np.zeros((2, N_stimuli * num_time_steps))
    stimulus_sequence[0,:] = np.repeat(stimuli * (stimuli<=5), num_time_steps)
    stimulus_sequence[1,:] = np.repeat(stimuli * (stimuli>=5), num_time_steps)
   
    ### Initial activity levels
    RatePar = Activity_Zero(NeuPar, InPar, StimPar, mean_statistics_weak, mean_statistics_strong) 
    
    ### Run network 
    RunPredNet(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, fln_save_data)
        
    ### Analysis
    t, RE, L, B, Out = Output_PredNet(NeuPar, NetPar, stimulus_sequence, folder, fln_save_data)
    Out_per_stimulus = Output_per_stimulus_PredNet(t, Out, StimPar, SimPar)
    
    ### Save data
    SaveActivityPredProcNet(t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar,  folder, fln_save_data)
      
    

def run_vanishing_bias(folder_data, folder, fln_save_data):
    
    N_transistion = 30 
    fln_load_after = 'Example_Target_Input_After'
    m_out, mBE, mBB, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
    
    ### Load data
    NeuPar_PE, NetPar_PE, InPar_PE, _, _, _, _ = LoadNetworkPara(folder_data, fln_load_after)
    _, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder_data, fln_load_after)
    
    ### Neuron parameters
    NeuPar = Neurons()
    
    ### Network parameters
    NetPar = Network(NeuPar, NetPar_PE, mBE, mBB, mLE, m_out, bool_nPE, bool_pPE, InPar_PE.neurons_motor)
    
    ### Simulation parameters
    SimPar = Simulation()
    
    ### Stimulation protocol & Inputs
    InPar = InputStructure(NeuPar, InPar_PE)
    
    mean_statistics_weak = np.mean(stimuli_weak)
    mean_statistics_strong = np.mean(stimuli_strong)
    
    N_stimuli = 100 
    distribution_flag = np.zeros(N_stimuli, dtype=np.int32)
    stimuli = np.zeros_like(distribution_flag)
    stimuli = np.random.choice(stimuli_weak, size=N_stimuli)
    stimuli[N_transistion:] = dtype(5)
      
    StimPar = Stimulation(distribution_flag, stimuli)
    
    num_time_steps = np.int32(SimPar.stim_duration/SimPar.dt/10) # because r is recorded every 10 steps
    stimulus_sequence = np.zeros((2, N_stimuli * num_time_steps))
    stimulus_sequence[0,:] = np.repeat(stimuli * (distribution_flag==0), num_time_steps)
    stimulus_sequence[1,:] = np.repeat(stimuli * (distribution_flag==1), num_time_steps)
   
    ### Initial activity levels
    RatePar = Activity_Zero(NeuPar, InPar, StimPar, mean_statistics_weak, mean_statistics_strong)
    
    ### Run network 
    RunPredNet(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, fln_save_data)
        
    ### Analysis
    t, RE, L, B, Out = Output_PredNet(NeuPar, NetPar, stimulus_sequence, folder, fln_save_data)
    Out_per_stimulus = Output_per_stimulus_PredNet(t, Out, StimPar, SimPar)
    
    ### Save data
    SaveActivityPredProcNet(t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar, folder, fln_save_data)
        
    

def run_contraction_bias(folder_data, folder, fln_save_data):

    fln_load_after = 'Example_Target_Input_After'
    m_out, mBE, mBB, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
  
    ### Load data
    NeuPar_PE, NetPar_PE, InPar_PE, _, _, _, _ = LoadNetworkPara(folder_data, fln_load_after)
    _, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder_data, fln_load_after)
    
    ### Neuron parameters
    NeuPar = Neurons()
    
    ### Network parameters
    NetPar = Network(NeuPar, NetPar_PE, mBE, mBB, mLE, m_out, bool_nPE, bool_pPE, InPar_PE.neurons_motor)
    
    ### Simulation parameters
    SimPar = Simulation()
    
    ### Stimulation protocol & Inputs
    InPar = InputStructure(NeuPar, InPar_PE)
    
    mean_statistics_weak = np.mean(stimuli_weak)
    mean_statistics_strong = np.mean(stimuli_strong)
    
    ### Run network for 'weak' stimuli
    distribution_flag = np.array([0])
    Out_weak = np.zeros(len(stimuli_weak))
    
    for i in range(len(stimuli_weak)):
        
        ## Stimulus
        stimuli = np.array([stimuli_weak[i]], dtype=dtype)
        StimPar = Stimulation(distribution_flag, stimuli)
        
        num_time_steps = np.int32(SimPar.stim_duration/SimPar.dt/10) # because r is recorded every 10 steps
        stimulus = np.zeros((2, num_time_steps))
        stimulus[0,:] = np.repeat(stimuli * (distribution_flag==0), num_time_steps)
        stimulus[1,:] = np.repeat(stimuli * (distribution_flag==1), num_time_steps)
        
        ## Initial activity levels
        RatePar = Activity_Zero(NeuPar, InPar, StimPar, mean_statistics_weak, mean_statistics_strong)
        
        ## Run static network 
        RunPredNet(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, fln_save_data + '_weak_' + str(i))
        
        ## Analysis
        t, RE, L, B, Out = Output_PredNet(NeuPar, NetPar, stimulus, folder, fln_save_data + '_weak_' + str(i))
        Out_per_stimulus = Output_per_stimulus_PredNet(t, Out, StimPar, SimPar)
        Out_weak[i] = Out_per_stimulus[0,0]
        
        ## Save data
        SaveActivityPredProcNet(t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar, folder, fln_save_data + '_weak_' + str(i))
    
    ### Run network for 'strong' stimuli
    distribution_flag = np.array([1])
    Out_strong = np.zeros(len(stimuli_strong))
    
    for i in range(len(stimuli_strong)):
        
        ## Stimulus
        stimuli = np.array([stimuli_strong[i]], dtype=dtype)
        StimPar = Stimulation(distribution_flag, stimuli)
        
        num_time_steps = np.int32(SimPar.stim_duration/SimPar.dt/10) # because r is recorded every 10 steps
        stimulus = np.zeros((2, num_time_steps))
        stimulus[0,:] = np.repeat(stimuli * (distribution_flag==0), num_time_steps)
        stimulus[1,:] = np.repeat(stimuli * (distribution_flag==1), num_time_steps)
        
        ## Initial activity levels
        RatePar = Activity_Zero(NeuPar, InPar, StimPar, mean_statistics_weak, mean_statistics_strong)
        
        ## Run static network 
        RunPredNet(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, fln_save_data + '_strong_' + str(i))
        
        ## Analysis
        t, RE, L, B, Out = Output_PredNet(NeuPar, NetPar, stimulus, folder, fln_save_data + '_strong_' + str(i))
        Out_per_stimulus = Output_per_stimulus_PredNet(t, Out, StimPar, SimPar)
        Out_strong[i] = Out_per_stimulus[1,0]
        
        ## Save data
        SaveActivityPredProcNet(t, RE, L, B, Out, Out_per_stimulus, StimPar, SimPar, folder, fln_save_data + '_strong_' + str(i))
    
    ### Save data for plotting
    SaveActivityBiasTowardsMean(stimuli_weak, stimuli_strong, Out_weak, Out_strong, folder, fln_save_data)
        