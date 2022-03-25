"""
PE networks
"""

# %% Import

import numpy as np
import copy

from Default_Parameters import Default_Connectivity, Default_LearningRates
from Functions_Network import Neurons, Network, Activity_Zero, InputStructure, Stimulation, Simulation, Learning
from Functions_Network import RunStaticNetwork, RunPlasticNetwork
from Functions_Save import SaveNetworkPara, LoadNetworkPara, SaveAnalysisActivity, LoadAnalysisActivity
from Functions_Save import SaveInputsPCs, SaveGeneralisability
from Functions_Save import SaveRobustness, SaveNumbersDepInp
from Functions_Analysis import Activity_relative_to_BL, Classify_PCs, Inputs_to_PCs, Activity_Phases


dtype = np.float32

# %%

def run_static_network_before(folder, fln):

    ### Neuron parameters
    NeuPar = Neurons()
    
    ### Network parameters
    weights_mean, conn_prob, weight_name, temp_name = Default_Connectivity()
    NetPar = Network(NeuPar, weights_mean, conn_prob, weight_name, temp_name)
    
    # create assemblies in connections from PV to E (divide PV into subpopulatione: one receiving sensory input, one receiving predictions)
    Part = np.int(NeuPar.NCells[0]/3)
    NP = np.int(NeuPar.NCells[1]/2)
    
    NetPar.wEP[:Part,:NP] *= 0.5 
    NetPar.wEP[:Part,NP:] *= 1.5  
    NetPar.wEP[Part:2*Part,:NP] *= 1.5 
    NetPar.wEP[Part:2*Part,NP:] *= 0.5 
    
    ### Stimulation protocol & inputs
    Xternal = dtype([5, 5, 5, 5, 0])
    SD, stim_high = dtype(1.5), dtype(5)
    VP, VS, VV = dtype(0.5), dtype(0.7), dtype(0.3)
    
    StimPar = Stimulation(stim_high, SD=SD)
    InPar = InputStructure(NeuPar, NetPar, VP=VP, VS=VS, VV=VV, Xternal=Xternal)
    
    ### Initial activity levels
    RatePar = Activity_Zero(NeuPar)
    
    ### Simulation parameters
    SimPar = Simulation()
    
    ### Run static network (Testing)
    RunStaticNetwork(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, fln)
    
    ### Learning parameters (for later)
    eta_mean, eta_name = Default_LearningRates()
    LearnPar = Learning(NeuPar, eta_mean, eta_name)
    
    ### Analyse E activity
    t, RE, RP, RS, RV  = Activity_relative_to_BL(NeuPar, StimPar, SimPar, folder, fln)
    bool_nPE, bool_pPE = Classify_PCs(t, RE, SimPar)
    
    ### Save data
    SaveNetworkPara(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, LearnPar, folder, fln)
    SaveAnalysisActivity(t, RE, RP, RS, RV, bool_nPE, bool_pPE, folder, fln)

    ############# Inputs #############
    fln_input = 'Data_Inputs2PCs_Network_Before'

    ### Stimulation protocol
    SD, stim_high = dtype(1.5), dtype(5)
    StimPar = Stimulation(stim_high, SD=SD, num_stim = 5, flg_test = 0)
    StimPar.stim_visual = np.array([0,1,0,3,0,5,0,2,0,4], dtype=dtype)
    StimPar.stim_motor = StimPar.stim_visual
    
    ## Run static network
    RunStaticNetwork(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, 'Inputs2PCs_' + fln)
    
    ### Analysis
    t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M = Inputs_to_PCs(NeuPar, NetPar, InPar, 
                                                                                    StimPar, SimPar, folder, fln)
    ### Save data
    SaveInputsPCs(t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M, folder, fln_input)
        


def run_plastic_network(folder, fln, plasticity_type):
            
    fln_save = 'Example_' + fln
    
    ### Load data to define neuron, network parameters, etc.
    NeuPar, NetPar, InPar, StimPar_Test, SimPar_Test, RatePar, LearnPar = LoadNetworkPara(folder, 'Example_Before')
    
    ### Stimulation protocol (for training)
    num_stim = np.int32(350)
    SD, stim_high = dtype(1.5), dtype(5)

    StimPar_Training = Stimulation(stim_high, SD=SD, num_stim=num_stim, flg_test=0)
    
    ### Simulation parameters
    SimPar_Training = Simulation(dt=0.2)
    
    ### Run plastic network (Learning/Training)
    RunPlasticNetwork(NeuPar, NetPar, InPar, StimPar_Training, SimPar_Training, LearnPar, RatePar, 
                      folder, fln_save, plasticity_type=plasticity_type)
    
    ### Save network after learning
    SaveNetworkPara(NeuPar, NetPar, InPar, StimPar_Training, SimPar_Training, RatePar, LearnPar, folder, fln_save)
    
    ### Run static network (Testing)
    RunStaticNetwork(NeuPar, NetPar, InPar, StimPar_Test, SimPar_Test, RatePar, folder, fln_save)
    
    ### Analyse E activity
    t, RE, RP, RS, RV  = Activity_relative_to_BL(NeuPar, StimPar_Test, SimPar_Test, folder, fln_save)
    bool_nPE, bool_pPE = Classify_PCs(t, RE, SimPar_Test)
    
    ### Save data
    SaveAnalysisActivity(t, RE, RP, RS, RV, bool_nPE, bool_pPE, folder, fln_save)
    
    ############# Inputs #############
    fln_input = 'Data_Inputs2PCs_Network_' + fln
    SimPar = Simulation()

    ### Stimulation protocol
    SD, stim_high = dtype(1.5), dtype(5)
    StimPar = Stimulation(stim_high, SD=SD, num_stim = 5, flg_test = 0)
    StimPar.stim_visual = np.array([0,1,0,3,0,5,0,2,0,4], dtype=dtype)
    StimPar.stim_motor = StimPar.stim_visual
    
    ## Run static network
    RunStaticNetwork(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, 'Inputs2PCs_' + fln_save)
    
    ### Analysis
    t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M = Inputs_to_PCs(NeuPar, NetPar, InPar, 
                                                                                    StimPar, SimPar, folder, fln_save)
    ### Save data
    SaveInputsPCs(t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, V, M, folder, fln_input)
 

def run_generalise(folder, plasticity_type):    
    
    ### Path & file for storage and loading
    if plasticity_type==0:
        fln = 'Target_Rate_After'
    elif plasticity_type==1:
        fln = 'Target_Input_After'
    
    fln_load = 'Example_' + fln
    fln_data = 'Data_Generalisability_Network_' + fln

    ### Load data
    NeuPar, NetPar, InPar, _, _, RatePar, _ = LoadNetworkPara(folder, fln_load)
    _, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_load)
    
    ### Simulation parameters
    SimPar = Simulation()
    
    ### Run static network for various stimulation stengths
    SD, stim_max_training = dtype(1.5), dtype(5)
    stimuli = np.arange(1, 2*stim_max_training+1, 1, dtype=dtype)
    
    FB = np.zeros((len(stimuli), NeuPar.NCells[0]))
    
    for i in range(len(stimuli)):
    
        ## Stimulation protocol 
        stim_high = stimuli[i]
        StimPar = Stimulation(stim_high, SD=SD)
        
        ## Run static network
        RunStaticNetwork(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, 'Temporary_' + fln)
        
        ## Analyse E activity
        t, RE, RP, RS, RV  = Activity_relative_to_BL(NeuPar, StimPar, SimPar, folder, 'Temporary_' + fln)
        E_Phases, P_Phases, S_Phases, V_Phases  = Activity_Phases(t, RE, RP, RS, RV, SimPar)
        
        ## FB-BL activity vs. stimulus strength
        FB[i, :] = E_Phases[:, 1]
        
    ### Save data
    SaveGeneralisability(stimuli, FB, stim_max_training, folder, fln_data)
        


def run_robustness(folder, plasticity_type):

    for perturbation in [-2,2]:
    
        ### Path & file for storage and loading
        if plasticity_type==0:
            fln = 'Target_Rate_After'
        elif plasticity_type==1:
            fln = 'Target_Input_After'
        
        fln_load = 'Example_' + fln
        fln_data = 'Data_Robustness_Network_' + fln + '_' + (('suppress')*(perturbation<0) + ('stimulate')*(perturbation>0))
            
        ### Load data
        NeuPar_ctrl, _, _, _, SimPar_ctrl, RatePar_ctrl, _ = LoadNetworkPara(folder, 'Example_' + fln)
        NeuPar, NetPar, InPar_Zero, _, _, RatePar, _ = LoadNetworkPara(folder, fln_load)
        _, _, _, _, _, bool_nPE, bool_pPE = LoadAnalysisActivity(folder, fln_load)
        
        ### Simulation parameters
        SimPar = Simulation()
        
        ### Stimulation protocol & Inputs
        SD, stim_high = dtype(1.5), dtype(5)
        StimPar = Stimulation(stim_high, SD=SD)
        
        ### Anlayse E unperturbed E activity
        t, RE, RP, RS, RV  = Activity_relative_to_BL(NeuPar_ctrl, StimPar, SimPar_ctrl, folder, 'Example_' + fln)
        E_Ctrl, _, _, _  = Activity_Phases(t, RE, RP, RS, RV, SimPar_ctrl)
        
        ### Run static network when neurons/compartments are perturbed
        BL = np.zeros((5, NeuPar.NCells[0]))
        FB = np.zeros((5, NeuPar.NCells[0]))
        
        for i in range(5): # E, PV, SOM, VIP, D
        
            ## Define modulation/perturbation
            InPar = copy.deepcopy(InPar_Zero) 
            
            if i<4:
                manipulate_who = np.zeros(4)
                manipulate_who[i] = 1
                InPar.inp_ext_soma += perturbation * np.repeat(manipulate_who, NeuPar.NCells)
            else:
                InPar.inp_ext_dend += perturbation
            
            ## Run static network
            RunStaticNetwork(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, 'Temporary_' + fln)
            
            ## Analyse E activity
            t, RE, RP, RS, RV  = Activity_relative_to_BL(NeuPar, StimPar, SimPar, folder, 'Temporary_' + fln)
            E_Phases, P_Phases, S_Phases, V_Phases  = Activity_Phases(t, RE, RP, RS, RV, SimPar)
            
            FB[i, :] = E_Phases[:, 1]
            
        ### Save data
        SaveRobustness(FB, BL, E_Ctrl, folder, fln_data) # BL does not matter here
        
        
def run_input_distribution(folder_load, folder_save, flag_IN):
        
        fln_load = 'Example_Target_Input_After'
        V_fraction = np.linspace(0.1,0.9,5)
        Num_nPE = np.zeros(len(V_fraction))
        Num_pPE = np.zeros(len(V_fraction))
        
        if flag_IN==0:
            fln_pickle = 'DependenceInput_PV'
        elif flag_IN==1:
            fln_pickle = 'DependenceInput_SOM'
        elif flag_IN==2:
            fln_pickle = 'DependenceInput_VIP'
        
        for i in range(len(V_fraction)):
            
            fln_save = fln_pickle + '_' + str(i)
  
            ### Load data to define neuron, network parameters, etc.
            _, _, _, StimPar_Test, SimPar_Test, _, _ = LoadNetworkPara(folder_load, 'Example_Before')
            NeuPar, NetPar, InPar, _, _, RatePar, LearnPar = LoadNetworkPara(folder_load, fln_load)
            
            ### Adjust input distribution
            InPar_new = copy.deepcopy(InPar) 
            
            # XXXX 
            NCells = NeuPar.NCells
            Nb = np.cumsum(NCells, dtype=np.int32) 
            N = NCells[flag_IN+1]
            num = int(V_fraction[i]*N)
            
            vis = np.array([1] * num + [0] * (N - num), dtype=dtype)
            InPar_new.neurons_visual[Nb[flag_IN]:Nb[flag_IN+1]] = vis
            InPar_new.neurons_motor[Nb[flag_IN]:Nb[flag_IN+1]] = 1 - vis
            
            ### Run static network (Testing)
            RunStaticNetwork(NeuPar, NetPar, InPar_new, StimPar_Test, SimPar_Test, RatePar, folder_save, fln_save)
            
            ### Analyse E activity
            t, RE, RP, RS, RV  = Activity_relative_to_BL(NeuPar, StimPar_Test, SimPar_Test, folder_save, fln_save)
            bool_nPE, bool_pPE = Classify_PCs(t, RE, SimPar_Test)
            
            ### Save data
            SaveAnalysisActivity(t, RE, RP, RS, RV, bool_nPE, bool_pPE, folder_save, fln_pickle + '_' + str(i))
     
            ### extract number of nPE and pPE neurons
            Num_nPE[i] = 100 * sum(bool_nPE==1)/NeuPar.NCells[0]
            Num_pPE[i] = 100 * sum(bool_pPE==1)/NeuPar.NCells[0]
            
        ### Save numbers
        SaveNumbersDepInp(V_fraction, Num_nPE, Num_pPE, folder_save, fln_pickle)
    
     
            
def run_initial_weights(folder_load, folder_save, flag_ini):
    
    if flag_ini==0:
        fln_save = 'DependenceInitialConnectivity_nPE'
    elif flag_ini==1:
        fln_save = 'DependenceInitialConnectivity_pPE' 
    elif flag_ini==2:
        fln_save = 'DependenceInitialConnectivity_PE'

    ### Load data to define neuron, network parameters, etc.
    NeuPar, _, _, StimPar_Test, SimPar_Test, RatePar, LearnPar = LoadNetworkPara(folder_load, 'Example_Before')
    
    ### Adjust weights- inspired my phase space derived from MF network
    weights_mean, conn_prob, weight_name, temp_name = Default_Connectivity()
    
    uni_low = 0.5 * np.ones_like(weights_mean)
    uni_high = 1.5 * np.ones_like(weights_mean)
    
    if flag_ini==0: # nPE
        uni_low[0,2] = 2/abs(weights_mean[0,2]) # wEP
        uni_high[0,2] = 4/abs(weights_mean[0,2]) # wEP
        uni_low[2,3] = 0.5/abs(weights_mean[2,3]) # wPS
        uni_high[2,3] = 1/abs(weights_mean[2,3]) # wPS
        uni_low[2,4] = 1.5/abs(weights_mean[2,3]) # wPV
        uni_high[2,4] = 2.5/abs(weights_mean[2,3]) # wPV
        
    elif flag_ini==1: # pPE
        uni_low[0,2] = 2.5/abs(weights_mean[0,2]) # wEP
        uni_high[0,2] = 8/abs(weights_mean[0,2]) # wEP
        uni_low[2,3] = 1.5/abs(weights_mean[2,3]) # wPS
        uni_high[2,3] = 2.5/abs(weights_mean[2,3]) # wPS
        uni_low[2,4] = 0.5/abs(weights_mean[2,3]) # wPV
        uni_high[2,4] = 1/abs(weights_mean[2,3]) # wPV
        
    elif flag_ini==2: # both
        uni_low[0,2] = 2.5/abs(weights_mean[0,2]) # wEP
        uni_high[0,2] = 8/abs(weights_mean[0,2]) # wEP
        uni_low[2,3] = 1/abs(weights_mean[2,3]) # wPS
        uni_high[2,3] = 2.5/abs(weights_mean[2,3]) # wPS
        uni_low[2,4] = 0.5/abs(weights_mean[2,3]) # wPV
        uni_high[2,4] = 2/abs(weights_mean[2,3]) # wPV
    
    NetPar = Network(NeuPar, weights_mean, conn_prob, weight_name, temp_name, uni_low=uni_low, uni_high=uni_high)
        
    ### Inputs
    InPar = InputStructure(NeuPar, NetPar, VP=dtype(0.5), VS=dtype(1), VV=dtype(0), Xternal=dtype([5, 5, 5, 5, 0]))
    
    ### Stimulation protocol (for training)
    num_stim = np.int32(350)
    SD, stim_high = dtype(1.5), dtype(5)

    StimPar_Training = Stimulation(stim_high, SD=SD, num_stim=num_stim, flg_test=0)
    
    ### Simulation parameters
    SimPar_Training = Simulation(dt=0.2)
    
    ### Save network before learning
    SaveNetworkPara(NeuPar, NetPar, InPar, StimPar_Training, SimPar_Training, RatePar, LearnPar, folder_save, fln_save + '_Before')
    
    ### Run plastic network (Learning/Training)
    RunPlasticNetwork(NeuPar, NetPar, InPar, StimPar_Training, SimPar_Training, LearnPar, RatePar, 
                      folder_save, fln_save)
    
    ### Save network after learning
    SaveNetworkPara(NeuPar, NetPar, InPar, StimPar_Training, SimPar_Training, RatePar, LearnPar, folder_save, fln_save)
    
    ### Run static network (Testing)
    RunStaticNetwork(NeuPar, NetPar, InPar, StimPar_Test, SimPar_Test, RatePar, folder_save, fln_save)
    
    ### Analyse E activity
    t, RE, RP, RS, RV  = Activity_relative_to_BL(NeuPar, StimPar_Test, SimPar_Test, folder_save, fln_save)
    bool_nPE, bool_pPE = Classify_PCs(t, RE, SimPar_Test)
    
    ### Save data
    SaveAnalysisActivity(t, RE, RP, RS, RV, bool_nPE, bool_pPE, folder_save, fln_save)
   