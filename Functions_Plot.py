#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 14 2021

@author: loreen.hertaeg
"""

# %% import packages

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
#import pickle

from matplotlib import gridspec #, rc
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Functions_Save import LoadInput2PC_MFN
from Functions_Analysis import PathwaysCont

dtype = np.float32

# %% Colors

Col_Rate_E = '#9E3039'
Col_Rate_nE = '#955F89'
Col_Rate_nD = '#BF9BB8' 
Col_Rate_pE = '#CB9173' 
Col_Rate_pD = '#DEB8A6' 
Col_Rate_PVv = '#508FCE' 
Col_Rate_PVm = '#2B6299' 
Col_Rate_SOM = '#79AFB9' 
Col_Rate_VIP = '#39656D' 
Col_Input_V = '#B54C03' #'#D7A251' 
Col_Input_M = '#218380'

Col_dMM = '#CDCAE2'
Col_hMM = '#F3D4A5'

Col_Exc = '#BB3A44'
Col_Inh = '#2D669F'

PC_cmap = LinearSegmentedColormap.from_list(name='PC_cmap', colors=['#212121','#DBB3B7',Col_Rate_E])
Perturbations = LinearSegmentedColormap.from_list(name='Perturbations', colors=[(0, '#4D60C6'),
                                                                                (5/12, '#C8D5EE'), 
                                                                                (0.5,'#A9A19E'), #ADADAD
                                                                                (7/12,'#E9CEBE'),
                                                                                (1,'#B34139')])

cmap_weak = mpl.colors.LinearSegmentedColormap.from_list('Weak', ['#A32729', '#EDA12B'])
cmap_strong = mpl.colors.LinearSegmentedColormap.from_list('Strong', ['#414288', '#32965D'])


# %% Functions


def Plot_Colorbar(ax_cbar, orientation, title, cmap=None, vmin=-1, vmax=1):

    if cmap is None:
        col_map = Perturbations
    else:
        col_map = cmap

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=col_map),
                      cax=ax_cbar, orientation=orientation)
    
    
    ax_cbar.tick_params(size=1.0,pad=2.0)
    ax_cbar.tick_params(axis='both', which='both', labelsize=5)
    ax_cbar.set_title(title, fontsize=5, pad=2)
    cb.outline.set_visible(False)
    

def Plot_Numbers_DependenceInput(V_fraction, Num_nPE, Num_pPE, folder, fln, title=None, leg_flag=False,
                                 ylab_flag=1, xlab_flag=1, figsize=(4,4), fs=7, ms=1, lw=1, axs=None):
    
    if axs==None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    else:
        ax = axs
        
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=1)
    
    ax.plot(V_fraction*100, Num_nPE,'.-',color=Col_Rate_nE, lw=lw, markersize=ms, label='nPE')
    ax.plot(V_fraction*100, Num_pPE,'.-',color=Col_Rate_pE, lw=lw, markersize=ms, label='pPE')
    
    if xlab_flag==1:
        ax.set_xlabel('# Neurons receiving sensory input (%)',fontsize=fs)
    if ylab_flag==1:
        ax.set_ylabel('# Cells (%)',fontsize=fs)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    if title is not None:
        #ax.set_title(title, fontsize=fs, pad=-2)
        ax.text(0.4,1, title, fontsize=fs, transform=ax.transAxes)
    if leg_flag:
        ax.legend(loc=[0.2,0.5], fontsize=fs-1, handlelength=1, frameon=False)
    
    sns.despine(ax=ax)
    
    if axs==None:
        PathFig = 'Results/Figures/' + folder + '/'
        plt.savefig(PathFig + fln + '_Numbers.png', bbox_inches='tight', transparent=True, dpi=300)
        

def Plot_ModulationLearnRates(stimuli_weak, R_mod, R_classic, dR_mod, dR_classic, N_ini, folder, fln, fs=7, ms=1, lw=1, axs=None):
    
    if axs==None:
        fig, ax = plt.subplots(1,1,tight_layout=True)
    else:
        ax = axs
        
    ax.plot(stimuli_weak, dR_mod/dR_classic, '.-', color = cmap_weak(0.5), lw=lw, markersize=ms)
    ax.set_xlabel('Stimulus strength', fontsize=fs)
    ax.set_ylabel('Ratio initial \nspeed (mod / fix)', fontsize=fs)
    #ax.set_ylabel('Mod / fixed', fontsize=fs)
    #ax.set_title('Ratio initial speed', fontsize=fs, pad=0)
    ax.set_yticks([1.2,1.7])
    ax.locator_params(axis='x', nbins=3)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    sns.despine(ax=ax)

    ax_in = inset_axes(ax, width="40%",height="40%", loc=1)
    ax_in.axvspan(0, N_ini, color='k', alpha=0.1)
    ax_in.plot(R_mod, color='k', label='mod',lw=lw)
    ax_in.plot(R_classic, ':', color='k', label='fix',lw=lw)
    # ax_in.plot([0,0],[2.5,5],'k-',lw=lw)
    # ax_in.plot([0,2000],[2.5,2.5],'k-',lw=lw)
    ax_in.set_xlim([0,1000])
    ax_in.tick_params(axis='both', which='both', labelsize=fs-1)
    ax_in.set_xticks([]), ax_in.set_yticks([])
    ax_in.legend(loc=[0.5,0], fontsize=fs-1, frameon=False, handlelength=1)
    ax_in.set_xlabel('Time', fontsize=fs-1, labelpad=2) 
    ax_in.set_ylabel('Rate', fontsize=fs-1, labelpad=2)
    sns.despine(ax=ax_in)#, left=True, bottom=True)
    
    if axs==None:
       PathFig = 'Results/Figures/' + folder + '/'
       plt.savefig(PathFig + 'Plot_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300) 


def Plot_CoupledEnvironments(Ratio_AB, Error_coupled, Error_uncoupled, folder, fln, fs=7, ms=1, lw=1, axs=None):
    
    if axs==None:
        fig, ax = plt.subplots(1,1,tight_layout=True)
    else:
        ax = axs
    
    ax.plot(Ratio_AB*100, Error_coupled[1,:]/Error_coupled[0,:],'k-', lw=lw, label='Coupled')
    ax.plot(Ratio_AB*100, Error_uncoupled[1,:]/Error_uncoupled[0,:],'k:', lw=lw, label='Uncoupled')
    
    # ax.plot(Ratio_AB*100, Error_coupled[0,:],'-', color='#E08DAC', lw=lw, label='Class A, coupled')
    # ax.plot(Ratio_AB*100, Error_coupled[1,:],'-', color='#6A7FDB', lw=lw, label='Class B, coupled')
    # ax.plot(Ratio_AB*100, Error_uncoupled[0,:],'--', color='#E08DAC', lw=lw, label='Class A, uncoupled')
    # ax.plot(Ratio_AB*100, Error_uncoupled[1,:],'--', color='#6A7FDB', lw=lw, label='Class B, uncoupled')
    
    ax.locator_params(axis='y', nbins=4)
    ax.locator_params(axis='x', nbins=4)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    ax.legend(loc=3, frameon=False, fontsize=fs, handlelength=1)
    ax.set_ylabel('Performance error \n(dist. II / dist. I) ', fontsize=fs)
    ax.set_xlabel('# Stimuli from dist. II\nin training (%)', fontsize=fs)
    #ax.set_title('Ratio of errors', fontsize=fs, pad=0)
    sns.despine(ax=ax)
    
    if axs==None:
       PathFig = 'Results/Figures/' + folder + '/'
       plt.savefig(PathFig + 'Plot_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300) 
    
   
def Plot_SwitchAttractor(StimPar, SimPar, t, Pred, RE, bool_nPE, bool_pPE, dist1, dist2, 
                         folder, fln, fs=7, ms=1, lw=1, axs=None):
    
    ### Pre-processing
    stimuli = StimPar.stimuli
    N_stimuli = len(stimuli)
    distribution_flag = StimPar.distribution_flag
    stim_duration = SimPar.stim_duration
    
    mean_dist1 = np.mean(dist1)
    mean_dist2 = np.mean(dist2)

    Pred_final = np.zeros((2, N_stimuli))
    
    for i in range(N_stimuli):
        Pred_final[:,i] = np.mean(Pred[:, (t>(0.5+i)*stim_duration) & (t<(1+i)*stim_duration)], 1) 
    
    ### Plot
    if axs==None:
        plt.figure(figsize=(10,10),tight_layout=True)
    else:
        ax = axs
    
    for i in range(N_stimuli):
        
        if distribution_flag[i]==0:
            col = cmap_weak((stimuli[i]-1)/4)
        elif distribution_flag[i]==1:
            col = cmap_strong((stimuli[i]-5)/4)
        
        if stimuli[i]==5:
            ax.plot(i, 9, 'kx', ms=ms+1)
    
        ax.plot([i,i+1],[9.5,9.5],color=col,lw=lw+1.5)
        
    ax.plot(np.arange(0.5, N_stimuli, 1), Pred_final[0,:],'-o', color=cmap_weak(0.5), markersize=ms, lw=lw, label='Prediction neuron 1')
    ax.plot(np.arange(0.5, N_stimuli, 1), Pred_final[1,:],'-o', color=cmap_strong(0.5), markersize=ms, lw=lw, label='Prediction neuron 2')
    #ax.axhline(5, color='k',ls=':', zorder=0, lw=lw)#, alpha=0.5)
    ax.axhline(mean_dist1,color=cmap_weak(0.5),ls=':', zorder=0, lw=lw, label='Mean of dist. I')#, alpha=0.5)
    ax.axhline(mean_dist2,color=cmap_strong(0.5),ls=':', zorder=0, lw=lw, label='Mean of dist. II')#, alpha=0.5)
    
    winner = 0 * (Pred_final[0,:]>Pred_final[1,:]) + 1 * (Pred_final[0,:]<Pred_final[1,:])
    diff_winner = np.diff(winner)
    
    for i in range(len(diff_winner)):
        if (diff_winner[i-1]!=0):
            ax.plot(i, 9, 'kv', ms=ms)
    
    for i in range(len(distribution_flag)):
        if distribution_flag[i]==0:
            col = '#F9EFE9' #cmap_weak(0.5)
        elif distribution_flag[i]==1:
            col = '#EAF0EF' #cmap_strong(0.5)
        ax.axvspan(i,i+1,color=col, zorder=0) #, alpha=0.1)
        
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=3)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    
    ax.set_xlabel('# Trials', fontsize=fs)
    ax.set_ylabel('Activity of pred. \nneurons (1/s)', fontsize=fs) # 'Activity (1/s)'
    
    ax.plot(np.nan, np.nan, 'kx', ms=ms+1, label='Stimulus 5')
    ax.plot(np.nan, np.nan, 'kv', ms=ms+1, label='Switch')
    ax.legend(loc=[1.05,0.05], fontsize=fs, frameon=False)
    
    ax.set_xlim([0,N_stimuli])
    ax.set_ylim([-0.3,9.5])
    
    sns.despine(ax=ax)

    if axs==None:
        PathFig = 'Results/Figures/' + folder + '/'
        plt.savefig(PathFig + 'Plot_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300)  
    

def Plot_TransitionNewDistribution(StimPar, Out_per_stimulus, N_transistion, folder, fln, fs=7, ms=1, lw=1, axs=None):
    
    ### Pre-processing
    Out = np.max(Out_per_stimulus,0)
    stimuli = StimPar.stimuli
    N_stimuli = len(stimuli)
    choices = np.unique(stimuli)
    
    ### Plot
    if axs==None:
        fig, ax = plt.subplots(1,1,tight_layout=True)
    else:
        ax = axs
    
    ax.scatter(np.arange(N_stimuli), Out, marker='o', c=stimuli, cmap=cmap_weak, s=ms)
    ax.set_xlim([0,100])
    for i in range(len(choices)):
        ax.axhline(choices[i], xmax=0.3, color=cmap_weak(choices[i]/len(choices)), ls='--', alpha=1, lw=lw)
    ax.axhline(choices[-1], xmin=0.3, color=cmap_weak(choices[-1]/len(choices)), ls='-', alpha=1, lw=lw)
    ax.axvline(N_transistion, color='k', ls=':',alpha=0.5, zorder=0, lw=lw)
    #ax.text(35,2,'Transition', fontsize=fs-1) #,rotation=-90
    
    #ax.locator_params(axis='y', nbins=4)
    #ax.locator_params(axis='x', nbins=4)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    ax.set_xlabel('# Trials', fontsize=fs)
    ax.set_ylabel('Perceived stimulus', fontsize=fs)
    sns.despine(ax=ax)
              
    if axs==None:
       PathFig = 'Results/Figures/' + folder + '/'
       plt.savefig(PathFig + 'Plot_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300)   


def Plot_BiasTowradsMean(stimuli_weak, stimuli_strong, Out_weak, Out_strong, folder, fln, fs=7, ms=1, lw=1, axs=None):
    
    if axs==None:
        fig, ax = plt.subplots(1,1,tight_layout=True)
    else:
        ax = axs
    
    # ax.scatter(np.repeat(stimuli_weak,N_out), R_out_weak, marker='o', c=np.repeat(stimuli_weak,N_out), cmap='autumn')
    # ax.scatter(np.repeat(stimuli_strong,N_out), R_out_strong, marker='o', c=np.repeat(stimuli_strong,N_out), cmap='winter')
    ax.scatter(stimuli_weak, Out_weak, marker='o', c=stimuli_weak, cmap=cmap_weak, s=ms)
    ax.scatter(stimuli_strong, Out_strong, marker='o', c=stimuli_strong, cmap=cmap_strong, s=ms)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, '--', color='k', alpha=0.3, zorder=0, lw=lw)
    
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=3)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    ax.set_xlabel('Received stimulus', fontsize=fs)
    ax.set_ylabel('Perceived stimulus', fontsize=fs)
    ax.set_xlim([0.5,9.5])
    ax.set_ylim([0.5,9.5])
    
    sns.despine(ax=ax) 
    
    if axs==None:
        PathFig = 'Results/Figures/' + folder + '/'
        plt.savefig(PathFig + 'Plot_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300)
      

def Plot_Robustness(FB_1, FB0_1, FB_2, FB0_2, perturbation, filename_plot, idx=[], ylim=[], leg_flg=False, 
                    figsize=(6.4,4.8), fs=7, ms=1, lw=1, range_target=[], G=None):
    
    ### Pre-processing
    if len(idx)==0:
        R_FB_1 = FB_1
        R_FB0_1 = FB0_1
        if FB_2 is not None:
            R_FB_2 = FB_2
            R_FB0_2 = FB0_2
    else:
        R_FB_1 = FB_1[:,idx>0]
        R_FB0_1 = FB0_1[idx>0]
        if FB_2 is not None:
            R_FB_2 = FB_2[:,idx>0]
            R_FB0_2 = FB0_2[idx>0]
            
    if len(range_target)==0:
        idx_targets = np.arange(5)
    else:
        idx_targets = range_target
    
    ### Plot
    if G is None:
        plt.figure(figsize=figsize,tight_layout=True)
        GG = gridspec.GridSpec(1,len(idx_targets))
    else:
        GG = G 
    
    Target = ['E soma','PV','SOM','VIP','E dend']
    
    Col = sns.color_palette('coolwarm', 11) 
    
    if isinstance(perturbation,list):
        col1 = Col[4]*(perturbation[0]<0) + Col[6]*(perturbation[0]>0)
        col1_mean = Col[0]*(perturbation[0]<0) + Col[-1]*(perturbation[0]>0)
        col2 = Col[4]*(perturbation[1]<0) + Col[6]*(perturbation[1]>0)
        col2_mean = Col[0]*(perturbation[1]<0) + Col[-1]*(perturbation[1]>0)
    else:
        if perturbation<0:
            col1 = Col[4]
            col1_mean = Col[0]
        elif perturbation>0:
            col1 = Col[6]
            col1_mean = Col[-1]

    k = 0
    for i in idx_targets:
        
        ax = plt.subplot(GG[0,k])
        ax.locator_params(axis='y', nbins=3)
    
        ax.plot([1,2],[R_FB0_1, R_FB_1[i,:]], '.-', color=col1, alpha=0.3, lw=lw, markersize=ms)
        if FB_2 is not None:
            ax.plot([1,2],[R_FB0_2, R_FB_2[i,:]], '.-', color=col2, alpha=0.3, lw=lw, markersize=ms)
        
        ax.plot([1,2],[np.mean(R_FB0_1), np.mean(R_FB_1[i,:])], '.-', color=col1_mean, lw=lw, label='Inh. pert.', markersize=ms)
        if FB_2 is not None:
            ax.plot([1,2],[np.mean(R_FB0_2), np.mean(R_FB_2[i,:])], '.-', color=col2_mean, lw=lw, label='Exc. pert.', markersize=ms)
        
        ax.set_xticks([1,2])
        ax.set_xlim([0.7,2.3])
        if len(ylim)>0:
            ax.set_ylim(ylim)
        
        ax.tick_params(axis='both', which='both', labelsize=fs)
        ax.tick_params(axis='both', which='both', size=1.0)
        
        if k==0:
            ax.set_ylabel('Rate FP - BL (1/s)', fontsize=fs)
            ax.set_xticklabels(['Ctrl','Pert'], fontsize=fs, rotation=30)
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            
        if k==1:
            if leg_flg:
                ax.legend(loc=2, handlelength=1, fontsize=fs, frameon=False)
            
        ax.text(0.9,-0.7,Target[i], fontsize=fs)
        
        if k>0:
            ax.set_yticks([])
            sns.despine(ax=ax, left=True)
        else:
            sns.despine(ax=ax)
        
        k += 1
    
    ### Save
    if G is None:
        plt.savefig(filename_plot + '.png', bbox_inches='tight', transparent=True, dpi=300)


def Plot_Generalisability(stimuli, FB, stim_max_training, filename_plot, idx=[], figsize=(6.4,4.8), fs=7, ms=1, lw=1, axs=None):
    
    ### Pre-processing
    if len(idx)==0:
        R_FB = FB
    else:
        R_FB = FB[:,idx>0]
    
    ### Plotting
    if axs==None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        ax = axs
    
    ax.locator_params(axis='y', nbins=3)    
    ax.locator_params(axis='x', nbins=3) 

    #plt.plot(stim_all, R_FB,'o',color='k',alpha=0.5)
    N = np.size(R_FB,1)
    R_FB_avg = np.median(R_FB,1)
    R_FB_below = R_FB_avg-np.std(R_FB,1)/np.sqrt(N)
    R_FB_above = R_FB_avg+np.std(R_FB,1)/np.sqrt(N)
    y = np.stack((R_FB_below, R_FB_above), axis=0)
    x = np.tile(stimuli,(2,1))
    
    ax.plot(stimuli, R_FB_avg,'o',color=Col_Rate_E, markersize=ms, lw=lw)
    ax.plot(stimuli, R_FB_above,'_', color=Col_Rate_E, markersize=ms, lw=lw)
    ax.plot(stimuli, R_FB_below,'_', color=Col_Rate_E, markersize=ms, lw=lw)
    ax.plot(x,y,'-', color = Col_Rate_E, lw=lw)
    #ax.axvline(stim_max_training,linestyle='--',color = 'r',zorder=0, lw=lw)
    ax.axvspan(0, stim_max_training, color='k', alpha=0.1)
    ax.tick_params(axis='both', which='both', size=1.0, pad=2.0)
    ax.set_xlabel('Stimulus (1/s)', fontsize=fs)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    #ax.set_ylabel(r'r$_\mathrm{FB}$ - r$_\mathrm{BL}$ (1/s)')
    ax.set_ylabel(' Rate FP (1/s)', fontsize=fs)
    
    #ax.set_ylim([-1,2])
    ax.set_xlim(left=0)
    sns.despine(ax=ax) 
    
    if axs==None:
        plt.savefig(filename_plot +  '.png', bbox_inches='tight', transparent=True, dpi=300)
    
    

def Plot_Inputs_to_PCs(NeuPar, t, Inp_soma_Exc, Inp_soma_Inh, Inp_dend_Exc, Inp_dend_Inh, 
                       V, M, filename_plot, figsize=(6.4,4.8), fs=7, ms=1, lw=1, G=None):
    
    ### Parameters & Pre-processing
    NCells = NeuPar.NCells
    who_rand = np.random.randint(0, NCells[0], size=10)
    Input_soma_pop = np.mean(Inp_soma_Inh + Inp_soma_Exc,1)
    Input_dend_pop = np.mean(Inp_dend_Inh + Inp_dend_Exc,1)
    
    ### Plot
    if G is None:
        plt.figure(figsize=figsize,tight_layout=True)
        GG = gridspec.GridSpec(3,1)
    else:
        GG = G 
        
    ylx, yly = -0.25, 0.5 
    
    ax = plt.subplot(GG[0,0])
    #ax.plot(t/1000,np.mean(Inp_soma_Exc,1),color=Col_Rate_E, lw=lw)
    #ax.plot(t/1000,np.mean(Inp_soma_Inh,1),color='b', lw=lw)
    ax.plot(t/1000,Input_soma_pop,color=Col_Rate_E, lw=lw)
    ax.axhline(0, color='k', ls=':', alpha=0.5, zorder=0, lw=lw)
    ax.set_xticks([])
    ax.set_yticks([0,np.round(np.max(Input_soma_pop),0)])
    ax.set_ylabel('Soma', fontsize=fs)#, labelpad=5)
    ax.yaxis.set_label_coords(ylx, yly)
    #ax.set_title('Inputs to PCs (1/s)', fontsize=fs, pad=5)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    sns.despine(ax=ax, bottom=True)
    
    ax = plt.subplot(GG[1,0])
    #ax.plot(t/1000,np.mean(Inp_dend_Exc,1),color=Col_Rate_E, lw=lw)
    #ax.plot(t/1000,np.mean(Inp_dend_Inh,1),color='b', lw=lw)
    ax.plot(t/1000,Input_dend_pop,color=Col_Rate_E, lw=lw)
    ax.axhline(0, color='k', ls=':', alpha=0.5, zorder=0, lw=lw)
    ax.set_xticks([])
    ax.set_yticks([0,np.round(np.max(Input_soma_pop),0)])
    ax.set_ylabel('Dend', fontsize=fs)#, labelpad=5)
    ax.yaxis.set_label_coords(ylx, yly)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    sns.despine(ax=ax, bottom=True)
    
    ax = plt.subplot(GG[2,0])
    ax.plot(t/1000,V,color=Col_Input_V, lw=lw)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    ax.set_yticks([0,np.round(np.max(Input_soma_pop),0)])
    ax.set_xlabel('Time (s)', fontsize=fs)
    ax.set_ylabel('Stim', fontsize=fs)#, labelpad=5)
    ax.yaxis.set_label_coords(ylx, yly)
    sns.despine(ax=ax)
    
    ### Save
    if G is None:
        plt.savefig(filename_plot +  '.png', bbox_inches='tight', transparent=True, dpi=300)
        
        
def Plot_PC_Activity_Test(t, PC, bool_nPE, bool_pPE, StimPar, SimPar, folder, fln, ylab=True, xlab=True,
                          window_out = dtype(500), fs=7, ms=1, lw=1, dpi = 500, Panel='', G = None):
    
    ### Define phases
    #dt = SimPar.dt
    stim_duration = SimPar.stim_duration
    Test_phases = stim_duration * np.array([1,2,3,4,5,6,7])
    stim_type = StimPar.stim_type
    
    ### Sort PCs according to maximal activity within MM phases
    MM_end = Test_phases[stim_type<0][0]
    isort = np.argsort(np.max(PC[(t>MM_end-window_out) & (t<MM_end),:],0))
    
    PC = PC[:,isort]    
    PC = PC[(t>window_out),:]
    
    ### Extract visual and motor input
    time_steps_per_phase = np.int32(100) # np.int32(stim_duration/dt)
    stim_visual = StimPar.stim_visual 
    stim_motor = StimPar.stim_motor
    V = np.repeat(stim_visual, time_steps_per_phase)
    M = np.repeat(stim_motor, time_steps_per_phase)
    t0 = np.linspace(0, Test_phases[-1], len(V))
    
    ### Plotting
    NE = np.size(PC,1)
    
    if G is None:
        GG = gridspec.GridSpec(9,22)
        GG.update(hspace=0.5, wspace=0.5)
    else:
        GG = G
    
    max_abs_R = np.max(abs(PC),0)
    Mean_max_abs_R = np.nanmean(max_abs_R)
    SD_max_abs_R = np.nanstd(max_abs_R)
    #vc = max(Mean_max_abs_R + SD_max_abs_R, 1)
    vc = np.ceil(max(np.nanmax(max_abs_R), 1))
    cbar_tick = np.round(0.9*vc,0)
    
    ax2 = plt.subplot(GG[:-2,:-1])
    ax2.text(-0.2, 1.2, Panel, fontsize=fs, transform=ax2.transAxes)
    
    cbar_ax = plt.subplot(GG[-1,:-1])
    if xlab:
        BarLabel = 'Rate rel. to BL (1/s)'
    else:
        BarLabel = ''
    
    data = pd.DataFrame(PC.T, columns=np.round(t[(t>500.0)]), index=range(NE))
    ax2 = sns.heatmap(data,ax=ax2, cmap=PC_cmap, xticklabels=False, yticklabels=False, cbar_ax=cbar_ax, 
                      cbar_kws={"orientation": "horizontal",'label':BarLabel, 'ticks': [-cbar_tick,0,cbar_tick]},
                      vmin=-vc,vmax=vc)
    
    ax2.tick_params(axis='both', which='both', size=1.0, pad=2.0)
    #ax2.set_yticks([]) 
    ax2.set_xticks([])
    ax2.invert_yaxis()
    if ylab:
        ax2.set_ylabel('Neurons', fontsize=fs)
    #ax2.axis('off')
    
    axS = plt.subplot(GG[:-2,-1])
    axS.plot([1,2],[range(NE), range(NE)], color='#B1B1B1', alpha=0.5,lw=3)
   
    Id_nPE = np.where(bool_nPE)[0]
    if len(Id_nPE)>0:
        IdxMM = np.argsort(isort)[Id_nPE]
        axS.plot([1,2],[IdxMM,IdxMM],'-',color=Col_Rate_nE,lw=0.5)
        
    Id_pPE = np.where(bool_pPE)[0]
    if len(Id_pPE)>0:
        IdxPB = np.argsort(isort)[Id_pPE]
        axS.plot([1,2],[IdxPB,IdxPB],'-',color=Col_Rate_pE,lw=0.5)
        
    axS.set_ylim([0,NE-1]) 
    axS.set_xlim([1,2])
    axS.set_xticks([]), axS.set_yticks([])
    axS.axis('off')
    
    cbar_ax.tick_params(labelsize=fs)
    cbar_ax.tick_params(size=1.0, pad=2.0)
    cbar_ax.xaxis.label.set_size(fs)
    
    axI = plt.subplot(GG[-2,:-1])
    axI.plot(t0, V, color=Col_Input_V, lw=lw)
    axI.plot(t0, M, color=Col_Input_M, lw=lw)
    axI.set_xlim([window_out, Test_phases[-1]])
    axI.set_xticks([]), axI.set_yticks([])
    sns.despine(ax=axI), plt.axis('off')
    
    if G is None:
        ### Save
        PathFig = 'Results/Figures/' + folder
        if not os.path.exists(PathFig):
            os.mkdir(PathFig)
        
        plt.savefig(PathFig + '/PC_Activity_Test_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=dpi)
        plt.close()
        

def Plot_Manipulation_Space_Summary(filename_save, Inputs_nPE, Inputs_pPE, stim_extra_all, show=0, 
                                    fs=7, ms=1, lw=1, xlab=True, ylab=True, range_target=[],
                                    figsize=(6.4,4.8), ax=None):
    
    N = np.size(Inputs_nPE,2)
    if len(range_target)==0:
        idx_targets=np.arange(N)
    else:
        idx_targets=range_target
    
    Label_all = ['nPE\n(soma)', 'pPE\n(soma)', 'nPE\n(dend)', 'pPE\n(dend)', r'PV$_S$', r'PV$_P$', 'SOM', 'VIP']
    
    if ax==None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        ax = ax
        
    ax.axhline(90, lw=lw, color='k', ls='--')
    ax.axhline(0, lw=lw, color='k', ls='--')
    
    k=0
    for i in idx_targets:
        
        nPE_V, nPE_M = Inputs_nPE[3,:,i]-Inputs_nPE[0,:,i], Inputs_nPE[2,:,i]-Inputs_nPE[0,:,i]
        pPE_V, pPE_M = Inputs_pPE[3,:,i]-Inputs_pPE[0,:,i], Inputs_pPE[2,:,i]-Inputs_pPE[0,:,i]
    
        angle_nPE = np.arctan2(nPE_M,nPE_V) * 180 / np.pi
        angle_pPE = np.arctan2(pPE_M,pPE_V) * 180 / np.pi
        
        if show==1:
            angle_nPE[stim_extra_all<0] = 200
            angle_pPE[stim_extra_all<0] = 200
        elif show==-1:
            angle_nPE[stim_extra_all>0] = 200
            angle_pPE[stim_extra_all>0] = 200
        
        ax.scatter((k+1)*np.ones(len(stim_extra_all)), angle_nPE, marker='s', s=ms**2, c=stim_extra_all, cmap=Perturbations, zorder=50)
        ax.scatter((k+1)*np.ones(len(stim_extra_all)), angle_pPE, marker='o', s=ms**2, c=stim_extra_all, cmap=Perturbations, zorder=50)
        k += 1
        
    ax.axhspan(ymin=90,ymax=180, facecolor='#FFFBEB')
    ax.axhspan(ymin=-90,ymax=0, facecolor='#FFFBEB')
    
    ax.set_ylim([-60,140])
    x_idx = np.arange(1,len(idx_targets)+1)
    ax.set_xlim([x_idx[0]-0.5, x_idx[-1]+0.5])
    
    ax.set_xticks(x_idx) 
    ax.set_xticklabels(Label_all[x_idx[0]-1+idx_targets[0]:x_idx[-1]+idx_targets[0]], fontsize=fs, rotation=30)
    if xlab:
        ax.set_xlabel('Perturbation target', fontsize=fs)
    if ylab:
        ax.set_ylabel(r'Angle $\Theta$', fontsize=fs)
    
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    sns.despine(ax=ax)
    
    if ax==None:
        if show==0:
            plt.savefig(filename_save + '_All_' + Label_all[i] + '.png', bbox_inches='tight', transparent=True, dpi=300)
        elif show==1:
            plt.savefig(filename_save + '_Positive_' + Label_all[i] + '.png', bbox_inches='tight', transparent=True, dpi=300)
        elif show==-1:
            plt.savefig(filename_save + '_Negative_' + Label_all[i] + '.png', bbox_inches='tight', transparent=True, dpi=300)    

       
def Plot_Manipulation_Space_Scheme(filename_save, fs=7, ms=1, lw=1, figsize=(6.4,4.8), ax=None):
        
    if ax==None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        ax = ax
    
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    
    ax.plot(0, 3, marker='s', ms=ms, c='#A9A19E', zorder=10) 
    ax.plot(3, 0, marker='o', ms=ms, c='#A9A19E', zorder=10)
    ax.plot(3, 3, marker='.', ms=ms, c='k', zorder=10) 
    ax.arrow(0, 0, 2.6, 2.6, head_width = 0.2, width = 0.05, facecolor='k', lw=lw)
    
    # draw angle
    r =  1.5 #circle radius
    theta = np.linspace(0,0.25*np.pi,100)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    ax.plot(x1, x2, color='k', lw=lw)
    
    # annotations
    ax.annotate(r'$\Theta$', (0.7, 0.2), fontsize=fs)
    # ax.annotate('nPE', (0.3, 3), fontsize=fs, color='#8C271E')
    # ax.annotate('pPE', (3, 0.3), fontsize=fs, color='#8C271E')
    ax.annotate('OP - BL', (0.2, 4.), fontsize=fs)
    ax.annotate('UP - BL', (2.5, -0.7), fontsize=fs)
    
    # example
    ax.plot(np.nan,np.nan, 's', color='#A9A19E',label=r'nPE: $\Theta=90°$', ms=ms)
    ax.plot(np.nan,np.nan, 'o', color='#A9A19E',label=r'pPE: $\Theta=0°$', ms=ms)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.annotate('Bi-directional\n PE neuron', (-3.9,1.5), fontsize=fs)
    ax.annotate('Bi-directional\n PE neuron', (0.3,-2.6), fontsize=fs)
    
    ax.legend(loc=(-0.15,0.1), frameon=False, fontsize=fs, handletextpad=0.0)
    
    ax.axhspan(ymin=0,ymax=4, xmin=0, xmax=0.5, facecolor='#FFFBEB')
    ax.axvspan(xmin=0,xmax=4, ymin=0,ymax=0.5, facecolor='#FFFBEB',)

    sns.despine(ax=ax)
    
    if ax==None:
        plt.savefig(filename_save + '_Scheme' + '.png', bbox_inches='tight', transparent=True, dpi=300)


def Plot_Manipulation_BL_FB(filename_save, Inputs_nPE, Inputs_pPE, stim_extra_all, range_target=[], range_pert=[], 
                            fs=7, ms=1, lw=1, ylab_flag=True, x_flag=True, figsize=(6.4,4.8), ax=None):
    
    N = np.size(Inputs_nPE,2)
    M = np.size(Inputs_nPE,1)
    cmap = mpl.cm.get_cmap(Perturbations)
    Label_all = ['nPE\n(soma)', 'pPE\n(soma)', 'nPE\n(dend)', 'pPE\n(dend)', r'PV$_S$', r'PV$_P$', 'SOM', 'VIP']
    
    if len(range_target)==0:
        idx_targets=np.arange(N)
    else:
        idx_targets=range_target
        
    if len(range_pert)==0:
        range_pert=np.arange(M)
    else:
        range_pert=range_pert
    
    # FR vs. target for all perturbations
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    else:
        ax = ax

    ax.locator_params(axis='y', nbins=3) 
    
    k=-1
    for i in idx_targets:
        k += 1
        for j in range_pert:
            BL = 0.5 * (Inputs_nPE[0,-1-j,i] + Inputs_pPE[0,-1-j,i])
            FB = 0.5 * (Inputs_nPE[1,-1-j,i] + Inputs_pPE[1,-1-j,i])
            ax.plot([k+1-0.2,k+1+0.2], [BL, FB], marker='|', color=cmap(j/M), ms=ms, lw=lw)
    
    if x_flag:
        x_idx = np.arange(1,len(idx_targets)+1)
        ax.set_xticks(x_idx) 
        ax.set_xticklabels(Label_all[x_idx[0]-1+idx_targets[0]:x_idx[-1]+idx_targets[0]], fontsize=fs, rotation=30)
        ax.set_xlabel('Perturbation target', fontsize=fs)
        ax.set_xlim([x_idx[0]-0.5, x_idx[-1]+0.5])
    else:
        ax.set_xticklabels([])
        
    if ylab_flag:
        ax.set_ylabel('Input to PC (1/s)', fontsize=fs)
    
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=1.0)
    sns.despine(ax=ax)
 
    if ax==None:
        plt.savefig(filename_save + '.png', bbox_inches='tight', transparent=True, dpi=300)


def Plot_PE_MFN(stim_low_all, RE_MM, RE_PB, filename_plot, figsize=(6.4,4.8), ylab_flag=1, fs=7, ms=1, lw=1, ax1=None, ax2=None):
    
    # Inset
    x = np.linspace(0,2,100)
    y = np.ones_like(x)
    y[x<0.5] = 0
    y[x>1.5] = 0
    
    # Plot 1
    if ax1==None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    else:
        ax = ax1
            
    ax.plot(stim_low_all/stim_low_all[-1], RE_MM[0,:], color=Col_Rate_nE, lw=lw, marker='.', ms=ms) #'s'
    ax.plot(stim_low_all/stim_low_all[-1], RE_MM[1,:], color=Col_Rate_pE, lw=lw, marker='.', ms=ms) #'o'
    #ax.set_title('Motor input = ' + str(stim_low_all[-1]) + ' (1/s)', fontsize=fs)
    #ax.set_xlabel('S / P', fontsize=fs)
    ax.set_ylabel('Rate (1/s)', fontsize=fs)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=2.0)
    sns.despine(ax=ax)
    
    # ax_in = inset_axes(ax,width="30%",height="30%",loc=1)
    # for i in range(5):
    #     ax_in.plot(x,i*y/4, color=Col_Input_V, lw=lw)
    # ax_in.plot(x,y, color=Col_Input_M, lw=lw)
    # ax_in.axis('off')
    
    # Plot 2
    if ax1==None:
        plt.savefig(filename_plot +  '_MM.png', bbox_inches='tight',transparent=True,dpi=300)
    
    if ax2==None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    else:
        ax = ax2
        
    ax.plot(stim_low_all/stim_low_all[-1], RE_PB[0,:],color=Col_Rate_nE, lw=lw, marker='.', ms=ms) #'s'
    ax.plot(stim_low_all/stim_low_all[-1], RE_PB[1,:],color=Col_Rate_pE, lw=lw, marker='.', ms=ms) #'o'
    #ax.set_title('Visual input = ' + str(stim_low_all[-1]) + ' (1/s)', fontsize=fs)
    #ax.set_xlabel('P / S', fontsize=fs)
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.tick_params(axis='both', which='both', size=2.0)
    #if ylab_flag==1:
    ax.set_ylabel('Rate (1/s)', fontsize=fs)
    sns.despine(ax=ax)
    
    # ax_in = inset_axes(ax,width="30%",height="30%",loc=1)
    # for i in range(5):
    #     ax_in.plot(x,i*y/4, color=Col_Input_M, lw=lw)
    # ax_in.plot(x,y, color=Col_Input_V, lw=lw)
    # ax_in.axis('off')
    
    if ax2==None:
        plt.savefig(filename_plot +  '_PB.png', bbox_inches='tight',transparent=True,dpi=300)


def Plot_Test_MFN(ind_plot, folder, fln, filename_plot, figsize=(6.4,4.8), 
                  fs=7, ms=1, lw=1, lab_flag=0, dy=0, bar_y=3, dir_path=None, flag_Inp=False, G=None):
    
    ### Load data 
    if dir_path is None:
        PathData = 'Results/Data/' + folder
    else:
        PathData = dir_path + 'Results/Data/' + folder

    arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + str(fln) + '.dat',delimiter=' ')
    t, R_plot = arr[:,0], arr[:,1:]   
    
    ### Plotting
    N_plot = len(ind_plot)
    
    if G is None:
        plt.figure(figsize=figsize,tight_layout=True)
        GG = gridspec.GridSpec(N_plot,1)
        GG.update(hspace=0.0)
    else:
        GG = G 
    
    Label_all = ['nPE', 'pPE', r'nPE$_\mathrm{dend}$', r'pPE$_\mathrm{dend}$', 
                 r'PV$_\mathrm{S}$', r'PV$_\mathrm{P}$','SOM', 'VIP']
    Colors =[Col_Rate_nE, Col_Rate_pE, Col_Rate_nE, Col_Rate_pE,
             Col_Rate_PVv, Col_Rate_PVm, Col_Rate_SOM, Col_Rate_VIP]
    
    r_BL = np.mean(R_plot[(t>500.0) & (t<1000.0),:],0)
    print(r_BL)
    y_max = 1.1 * np.max(R_plot - r_BL[None,:])
    y_min = 1.1 * np.min(R_plot - r_BL[None,:]) 
    #offset = dy
    
    for k in range(N_plot): 
        
        j = ind_plot[k]
        ax = plt.subplot(GG[k,0])

        ax.plot(t, R_plot[:,j]-r_BL[j], lw=lw, color=Colors[j]) # '#5E5E5E'
        if lab_flag==1:
            ax.text(-850,4, Label_all[j], color=Colors[j], fontsize=fs)
        
        if k==0:
            ax.plot([1500,1500],[y_max-bar_y,y_max],'k-', lw=lw)
            ax.plot([1500,2000],[y_max-bar_y,y_max-bar_y],'k-', lw=lw)
        
        ax.axvspan(1000,2000,color='#F4F3EB') 
        ax.axvspan(3000,4000,color='#F4F3EB')
        ax.axvspan(5000,6000,color='#F4F3EB')

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim([500.0,6500.0])
        ax.set_ylim([y_min,y_max])
        #ax.set_ylim([-1.0,1.05]) 
        ax.axis('off')
    
    if flag_Inp:
        ax = plt.subplot(GG[-1,0])
        x = 1000 * np.array([0,1,1,2,2,3,3,4,4,5,5,6,6,7])
        M = 5 * np.array([0,0,1,1,0,0,1,1,0,0,0,0,0,0])
        V = 5 * np.array([0,0,1,1,0,0,0,0,0,0,1,1,0,0])
        ax.plot(x, V, color=Col_Input_V, lw=lw)
        ax.plot(x, M, color=Col_Input_M, lw=lw)
        if lab_flag==1:
            ax.text(0,-0.2,'Sensory input (S)', fontsize=fs, color=Col_Input_V, transform=ax.transAxes)
            ax.text(0,-0.6,'Prediction (P)', fontsize=fs, color=Col_Input_M, transform=ax.transAxes)
        ax.axvspan(1000,2000,color='#F4F3EB') 
        ax.axvspan(3000,4000,color='#F4F3EB')
        ax.axvspan(5000,6000,color='#F4F3EB')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim([500.0,6500.0])
        ax.set_ylim([y_min,y_max])
        #ax.set_ylim([-0.3,1.5]) 
        ax.axis('off')
    else:
        ax.text(1150,-11.6+dy,'FP', fontsize=fs)
        ax.text(3150,-11.6+dy,'OP', fontsize=fs)
        ax.text(5150,-11.6+dy,'UP', fontsize=fs)
           
    ### Save
    if G is None:
        plt.savefig(filename_plot +  '.png', bbox_inches='tight',transparent=True,dpi=300)
    #plt.close()


def Plot_Pathways(VS, VV, filename_data, filename_plot, figsize=(6.4,4.8), fs=7, ms=1, lw=1, 
                  dy=0, anno=False, flag_single_pathways=False, ax1=None, ax2=None):
    
    #plt.locator_params(axis='y', nbins=3)
    #plt.locator_params(axis='x', nbins=10)
    
    Pthwys_nE_E, Pthwys_nE_I, Pthwys_pE_E, Pthwys_pE_I, Pthwys_nD_E, Pthwys_nD_I, Pthwys_pD_E, Pthwys_pD_I = PathwaysCont(VS, VV, filename_data)
    N = np.size(Pthwys_nE_E,0)
    
    # Plotting results
    bar_width = 0.45
    C = [Col_Rate_nE, Col_Rate_pE, 
         Col_Rate_nD, Col_Rate_pD, 
         Col_Rate_PVv, Col_Rate_PVm,
         Col_Rate_SOM, Col_Rate_VIP]
    
    if ax1==None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    else:
        ax = ax1
        
    for i in range(N):
        if flag_single_pathways:
            ax.bar(0,Pthwys_nE_E[i],bottom=np.sum(Pthwys_nE_E[:i],0), width=bar_width,fc=C[i],ec=C[i],lw=0)
            ax.bar(0.5,Pthwys_nE_I[i],bottom=np.sum(Pthwys_nE_I[:i],0), width=bar_width,fc=C[i],ec=C[i],lw=0)
            ax.bar(1.5,Pthwys_nD_E[i],bottom=np.sum(Pthwys_nD_E[:i],0), width=bar_width,fc=C[i],ec=C[i],lw=0)
            ax.bar(2,Pthwys_nD_I[i],bottom=np.sum(Pthwys_nD_I[:i],0), width=bar_width,fc=C[i],ec=C[i],lw=0)
        else:
            ax.bar(0,Pthwys_nE_E[i],bottom=np.sum(Pthwys_nE_E[:i],0), width=bar_width,fc=Col_Exc,ec=Col_Exc,lw=0)
            ax.bar(0.5,Pthwys_nE_I[i],bottom=np.sum(Pthwys_nE_I[:i],0), width=bar_width,fc=Col_Inh,ec=Col_Inh,lw=0)
            ax.bar(1.5,Pthwys_nD_E[i],bottom=np.sum(Pthwys_nD_E[:i],0), width=bar_width,fc=Col_Exc,ec=Col_Exc,lw=0)
            ax.bar(2,Pthwys_nD_I[i],bottom=np.sum(Pthwys_nD_I[:i],0), width=bar_width,fc=Col_Inh,ec=Col_Inh,lw=0)
        
    if anno:
        ax.text(-0.2,1,'Exc.',color='w',rotation=90, fontsize=fs-1, weight="bold")#, transform=ax.transData)
        ax.text(0.3,1,'Inh.',color='w',rotation=90, fontsize=fs-1, weight="bold")
    
    ax.locator_params(axis='y', nbins=4)
    ax.spines['bottom'].set_position('zero')
    ax.tick_params(axis='x', which='both', size=0.0)
    ax.tick_params(axis='y', which='both', size=2.0)
    ax.set_ylabel('Strength, nPE',fontsize=fs)
    ax.set_xticks([])
    #ax.set_xticks([0.25,1.75, 3.25])
    #ax.set_xticklabels(['FB','MM', 'PB'],fontsize=fs)
    ax.locator_params(axis='y', nbins=3)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    sns.despine(ax=ax)
    
    if ax1==None:
        plt.savefig(filename_plot + '_nPE.png', bbox_inches='tight',transparent=True,dpi=300)
        
    if ax2==None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    else:
        ax = ax2
        
    for i in range(N):
        if flag_single_pathways:
            ax.bar(0, Pthwys_pE_E[i],bottom=np.sum(Pthwys_pE_E[:i],0), width=bar_width,fc=C[i],ec=C[i],lw=0)
            ax.bar(0.5, Pthwys_pE_I[i],bottom=np.sum(Pthwys_pE_I[:i],0), width=bar_width,fc=C[i],ec=C[i],lw=0)
            ax.bar(1.5, Pthwys_pD_E[i],bottom=np.sum(Pthwys_pD_E[:i],0), width=bar_width,fc=C[i],ec=C[i],lw=0)
            ax.bar(2, Pthwys_pD_I[i],bottom=np.sum(Pthwys_pD_I[:i],0), width=bar_width,fc=C[i],ec=C[i],lw=0)
        else:
            ax.bar(0, Pthwys_pE_E[i],bottom=np.sum(Pthwys_pE_E[:i],0), width=bar_width,fc=Col_Exc,ec=Col_Exc,lw=0)
            ax.bar(0.5, Pthwys_pE_I[i],bottom=np.sum(Pthwys_pE_I[:i],0), width=bar_width,fc=Col_Inh,ec=Col_Inh,lw=0)
            ax.bar(1.5, Pthwys_pD_E[i],bottom=np.sum(Pthwys_pD_E[:i],0), width=bar_width,fc=Col_Exc,ec=Col_Exc,lw=0)
            ax.bar(2, Pthwys_pD_I[i],bottom=np.sum(Pthwys_pD_I[:i],0), width=bar_width,fc=Col_Inh,ec=Col_Inh,lw=0)
        
    ax.locator_params(axis='y', nbins=4)
    ax.spines['bottom'].set_position('zero')
    ax.tick_params(axis='x', which='both', size=0.0)
    ax.tick_params(axis='y', which='both', size=2.0)
    ax.set_ylabel('Strength, pPE',fontsize=fs)
    ax.set_xticks([0.25,1.75])
    ax.set_xticklabels(['soma','dend'],fontsize=fs)
    ax.locator_params(axis='y', nbins=2)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    sns.despine(ax=ax)
    
    if ax2==None:
        plt.savefig(filename_plot + '_pPE.png', bbox_inches='tight',transparent=True,dpi=300)
        
        
def Plot_Inputs_Phases(folder, fln_data, filename_plot, figsize=(6.4,4.8), fs=7, ms=1, lw=1, dy=0, flag_comp=0, ylab=True, dir_path = None, anno=False, ax1=None, ax2=None):
    
    [Weights, stim_duration, stim_high, E_nE_phases, I_nE_phases, E_pE_phases,
    I_pE_phases, E_nD_phases, I_nD_phases, E_pD_phases, I_pD_phases] = LoadInput2PC_MFN(folder, fln_data, dir_path)
    
    # Plotting results
    bar_width = 0.45
    C = [Col_Exc, Col_Inh]
    
    if ax1==None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    else:
        ax = ax1
    
    if flag_comp==0:
        ax.bar([0, 1.5, 3],E_nE_phases, width=bar_width,fc=C[0],ec=C[0],lw=0)
        ax.bar([0.5, 2, 3.5],-I_nE_phases, width=bar_width,fc=C[1],ec=C[1],lw=0)
    elif flag_comp==1:
        ax.bar([0, 1.5, 3],E_nD_phases, width=bar_width,fc=C[0],ec=C[0],lw=0)
        ax.bar([0.5, 2, 3.5],-I_nD_phases, width=bar_width,fc=C[1],ec=C[1],lw=0)
        
    if anno:
        ax.text(-0.19,0.15,'Exc.',color='w',rotation=90, fontsize=fs-1, weight="bold")#, transform=ax.transData)
        ax.text(0.32,0.15,'Inh.',color='w',rotation=90, fontsize=fs-1, weight="bold")
    
    ax.locator_params(axis='y', nbins=2)
    ax.spines['bottom'].set_position('zero')
    ax.tick_params(axis='x', which='both', size=0.0)
    ax.tick_params(axis='y', which='both', size=1.0)
    if ylab:
        ax.set_ylabel('Inputs (1/s) \nto nPE',fontsize=fs)
    ax.set_xticks([])
    ax.set_ylim([0,2.9])
    #ax.set_xticks([0.25,1.75, 3.25])
    #ax.set_xticklabels(['FB','MM', 'PB'],fontsize=fs)
    ax.locator_params(axis='y', nbins=3)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    sns.despine(ax=ax)
    
    if ax1==None:
        plt.savefig(filename_plot + '_nPE.png', bbox_inches='tight',transparent=True,dpi=300)
        
    if ax2==None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    else:
        ax = ax2
        
    if flag_comp==0:
        ax.bar([0, 1.5, 3],E_pE_phases, width=bar_width,fc=C[0],ec=C[0],lw=0)
        ax.bar([0.5, 2, 3.5],-I_pE_phases, width=bar_width,fc=C[1],ec=C[1],lw=0)
    elif flag_comp==1:
        ax.bar([0, 1.5, 3],E_pD_phases, width=bar_width,fc=C[0],ec=C[0],lw=0)
        ax.bar([0.5, 2, 3.5],-I_pD_phases, width=bar_width,fc=C[1],ec=C[1],lw=0)
    
    ax.locator_params(axis='y', nbins=2)
    ax.spines['bottom'].set_position('zero')
    ax.tick_params(axis='x', which='both', size=0.0)
    ax.tick_params(axis='y', which='both', size=1.0)
    if ylab:
        ax.set_ylabel('Inputs (1/s)\nto pPE',fontsize=fs)
    ax.set_xticks([0.25,1.75, 3.25])
    ax.set_ylim([0,2.9])
    ax.set_xticklabels(['FP','OP', 'UP'],fontsize=fs)
    ax.locator_params(axis='y', nbins=2)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    sns.despine(ax=ax)
    
    if ax2==None:
        plt.savefig(filename_plot + '_pPE.png', bbox_inches='tight',transparent=True,dpi=300)
        
   
# def Plot_Num_PE(Num_nPE, Num_pPE, N_seeds, xlab, xticklab=[], ylab_flag = True, rot = 0, flag_leg = True, 
#                 y_lim_max = 50, fln=None, fs=7, ms=1, lw=1, axs=None):
    
#     if axs==None:
#         fig, ax = plt.subplots(figsize=(5,5), tight_layout=True)
#     else:
#         ax = axs
    
#     x = np.arange(np.size(Num_nPE,0))
#     width = 0.35  # the width of the bars

#     err_num_nPE = np.std(Num_nPE,1)/np.sqrt(N_seeds)
#     err_num_pPE = np.std(Num_pPE,1)/np.sqrt(N_seeds)

#     ax.bar(x - width/2, np.mean(Num_nPE,1), width, yerr=err_num_nPE, label='nPE', color='#955F89')#, capsize=2)
#     ax.bar(x + width/2, np.mean(Num_pPE,1), width, yerr=err_num_pPE, label='pPE', color='#CB9173')

#     ax.set_ylim([0,y_lim_max])
        
#     ax.set_xticks(x)
#     if len(xticklab)>0:
#         ax.set_xticklabels(xticklab, rotation=rot)
#     else:
#         ax.set_xticklabels(x)
#     ax.set_xlabel(xlab, fontsize=fs)
    
#     if flag_leg:
#         ax.legend(loc=2, ncol=2, handlelength=1, fontsize=fs, frameon=False)
    
#     if ylab_flag:
#         ax.set_ylabel('# cells (%)', fontsize=fs)
    
#     ax.locator_params(axis='y', nbins=3)
#     ax.tick_params(axis='both', which='both', labelsize=fs)
#     ax.tick_params(axis='both', which='both', size=1.0)

#     sns.despine(ax=ax)


# def Plot_Num_PE_types_weight(folder, fln_data, xlab, ylab=None, fln=None, fs=7, ms=1, lw=1, axs=None):
    
#     if axs==None:
#         fig, ax = plt.subplots(figsize=(5,5), tight_layout=True)
#     else:
#         ax = axs
    
#     _, filename = Create_PathFilePickle(folder, fln_data)
#     with open(filename,'rb') as f:
#         w, num_pp_arr, num_pn_arr, num_np_arr, num_nn_arr = pickle.load(f)
    
#     ax.plot(w, num_pp_arr, color='#9E3039')
#     ax.plot(w, num_pn_arr, color='#CDCAE2')
#     if any(num_nn_arr)>0:
#         ax.plot(w, num_nn_arr, color='#508FCE')
#     ax.plot(w, num_np_arr, color='#F3D4A5')
    
#     ax.set_xlabel(xlab, fontsize=fs)
#     if ylab!=None:
#         ax.set_ylabel(ylab, fontsize=fs)
        
#     ax.set_ylim([0,1])
#     ax.set_xlim([0.2,0.9])
#     ax.locator_params(axis='y', nbins=2)
#     ax.locator_params(axis='x', nbins=2)
    
#     ax.tick_params(axis='both', which='both', labelsize=fs)
#     ax.tick_params(axis='both', which='both', size=1.0)
    
#     sns.despine(ax=ax)
      

# def Plot_Fraction4Example(folder, fln_data, lab=False, fln=None, fs=7, ms=1, lw=1, axs=None):
    
#     if axs==None:
#         fig, ax = plt.subplots(figsize=(5,5), tight_layout=True)
#     else:
#         ax = axs
        
#     _, filename = Create_PathFilePickle(folder, fln_data + '_Example')
#     with open(filename,'rb') as f:
#         alpha_arr, wPS_arr, wPV_arr, MM_sign, PB_sign, num_pp, num_nn, num_pn, num_np = pickle.load(f)
    
#     if lab:
#         if num_nn>0:
#             ax.pie([num_pp,num_pn,num_nn, num_np], labels=['++','+-','--','-+'], 
#                    colors=['#9E3039','#CDCAE2','#508FCE','#F3D4A5'], textprops={'fontsize': fs})
#         else:
#             ax.pie([num_pp,num_pn,num_np], labels=['++','+-','-+'], 
#                    colors=['#9E3039','#CDCAE2','#F3D4A5'], textprops={'fontsize': fs})   
#     else:
#         if num_nn>0:
#             ax.pie([num_pp,num_pn,num_nn,num_np], colors=['#9E3039','#CDCAE2','#508FCE','#F3D4A5']) 
#         else:
#             ax.pie([num_pp,num_pn,num_np], colors=['#9E3039','#CDCAE2','#F3D4A5']) 


# def Plot_PhasePlane_PE_types(folder, fln_data, ylab=False, fln=None, fs=7, ms=1, lw=1, axs=None):

#     if axs==None:
#         fig, ax = plt.subplots(figsize=(5,5), tight_layout=True)
#     else:
#         ax = axs
        
#     _, filename = Create_PathFilePickle(folder, fln_data + '_Example')
#     with open(filename,'rb') as f:
#         alpha_arr, wPS_arr, wPV_arr, MM_sign, PB_sign, num_pp, num_nn, num_pn, num_np = pickle.load(f)
    
#     N = np.size(alpha_arr, 0)
#     wPS_flat = np.repeat(wPS_arr,N)
#     wPV_flat = np.tile(wPV_arr,N)
#     MM_sign_flat = MM_sign.flatten()
#     PB_sign_flat = PB_sign.flatten()    
    
#     ax.scatter(wPS_flat[(MM_sign_flat==1) & (PB_sign_flat==1)], wPV_flat[(MM_sign_flat==1) & (PB_sign_flat==1)], color='#9E3039',s=0.2)
#     ax.scatter(wPS_flat[(MM_sign_flat==-1) & (PB_sign_flat==-1)], wPV_flat[(MM_sign_flat==-1) & (PB_sign_flat==-1)], color='#508FCE',s=0.2)
#     ax.scatter(wPS_flat[(MM_sign_flat==1) & (PB_sign_flat<=0)], wPV_flat[(MM_sign_flat==1) & (PB_sign_flat<=0)], color='#CDCAE2',s=0.2)
#     ax.scatter(wPS_flat[(MM_sign_flat<=0) & (PB_sign_flat==1)], wPV_flat[(MM_sign_flat<=0) & (PB_sign_flat==1)], color='#F3D4A5',s=0.2)
    
#     ax.set_xlabel(r'$w_\mathrm{PS}$', fontsize=fs, labelpad=0)
#     if ylab:
#         ax.set_ylabel(r'$w_\mathrm{PV}$', fontsize=fs)
    
#     ax.set_xlim([0,2]), ax.set_ylim([0,2])
#     ax.locator_params(axis='y', nbins=2)
#     ax.locator_params(axis='x', nbins=2)
#     ax.tick_params(axis='both', which='both', labelsize=fs)
#     ax.tick_params(axis='both', which='both', size=1.0)
    

# def Plot_PopRate_PE_All(NeuPar, SimPar, t, RE, folder, fln=None, anno_flag=False,  fs=7, ms=1, lw=1, G=None):
    
#     # Pre-processing
#     stim_duration = SimPar.stim_duration
    
#     PC_phases  = np.zeros((NeuPar.NCells[0],4))
#     PC_phases[:,0] = np.mean(RE[(t>0.5*stim_duration) & (t<stim_duration),:],0)
#     PC_phases[:,1] = np.mean(RE[(t>1.5*stim_duration) & (t<2*stim_duration),:],0)
#     PC_phases[:,2] = np.mean(RE[(t>3.5*stim_duration) & (t<4*stim_duration),:],0)
#     PC_phases[:,3] = np.mean(RE[(t>5.5*stim_duration) & (t<6*stim_duration),:],0)
    
#     bool_pn = 1*((PC_phases[:,2] > 0.) & (PC_phases[:,3] <= 0.))
#     bool_np = 1*((PC_phases[:,2] <= 0.) & (PC_phases[:,3] > 0.))
#     bool_pp = 1*((PC_phases[:,2] > 0.) & (PC_phases[:,3] > 0.))
#     bool_nn = 1*((PC_phases[:,2] <= 0.) & (PC_phases[:,3] <= 0.))
    
#     y_max_pn = 1.1*np.max(np.mean(RE[:,bool_pn==1],1))
#     y_max_np = 1.1*np.max(np.mean(RE[:,bool_np==1],1))
#     y_max_pp = 1.1*np.max(np.mean(RE[:,bool_pp==1],1))
#     y_max_nn = 1.1*np.max(np.mean(RE[:,bool_nn==1],1))
#     y_max = np.nanmax([y_max_pn, y_max_np, y_max_pp, y_max_nn, 2])
    
#     y_min_pn = 1.1*np.min(np.mean(RE[:,bool_pn==1],1))
#     y_min_np = 1.1*np.min(np.mean(RE[:,bool_np==1],1))
#     y_min_pp = 1.1*np.min(np.mean(RE[:,bool_pp==1],1))
#     y_min_nn = 1.1*np.min(np.mean(RE[:,bool_nn==1],1))
#     y_min = np.nanmin([y_min_pn, y_min_np, y_min_pp, y_min_nn, -0.5])
    
#     # Plotting
#     if G is None:
#         plt.figure(figsize=(4,5),tight_layout=True)
#         GG = gridspec.GridSpec(4,1)
#     else:
#         GG = G 
    
#     for i in range(4):
        
#         if i==0:
#             bool_set = bool_pn
#         elif i==1:
#             bool_set = bool_np
#         elif i==2:
#             bool_set = bool_pp
#         elif i==3:
#             bool_set = bool_nn  
        
#         ax = plt.subplot(GG[i,0])
#         R_set = np.mean(RE[:,bool_set==1],1)
#         ax.plot(t, R_set, lw=lw, color=Col_Rate_E)
        
#         ax.text(500,1,str(sum(bool_set)), fontsize=fs)
    
#         ax.axvspan(1000,2000,color='#F4F3EB') 
#         ax.axvspan(3000,4000,color='#F4F3EB')
#         ax.axvspan(5000,6000,color='#F4F3EB')

#         ax.set_ylim([y_min,y_max])
#         ax.set_xlim([500.0,6500.0])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         if ((i==3) and anno_flag):
#             ax.text(1150,y_min-1,'FP', fontsize=fs)
#             ax.text(3150,y_min-1,'OP', fontsize=fs)
#             ax.text(5150,y_min-1,'UP', fontsize=fs)
            
#         if i==0:
#             ax.plot([5500,5500],[y_max-1,y_max],'k-', lw=lw)
#             ax.plot([5500,6000],[y_max-1,y_max-1],'k-', lw=lw)

#         ax.axis('off')


#     if G is None:
#         PathFig = 'Results/Figures/' + folder
#         plt.savefig(PathFig + '/Plot_PopRate_PE_All' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300) 
    


# def Plot_WeightPhasePlane(VE, VP, MP, VS, VV, wSV, wVS, num, folder, fln=None,
#                           Examples=None, ylab=True, fs=7, ms=1, lw=1, axs=None):

#     if axs==None:
#         fig, ax = plt.subplots(figsize=(5,5), tight_layout=True, projection='3d')
#     else:
#         ax = axs
     
#     ax.locator_params(axis='z', nbins=2)
#     ax.locator_params(axis='y', nbins=2)
#     ax.locator_params(axis='x', nbins=2)
    
#     # plot phase plane
#     if ((VS==1) & (VV==0)):
#         alpha_all, wPS_all, wPV_all = DeterminePhasePlane_10(VE, VP, MP, wSV, wVS, num)
        
#     elif((VS==1) & (VV==1)):
#         alpha_all, wPS_all, wPV_all = DeterminePhasePlane_11(VE, VP, MP, wSV, wVS, num)
        
#     elif((VS==0) & (VV==0)):
#         alpha_all, wPS_all, wPV_all = DeterminePhasePlane_00(VE, VP, MP, wSV, wVS, num)
        
#     elif((VS==0) & (VV==1)):
#         alpha_all, wPS_all, wPV_all = DeterminePhasePlane_01(VE, VP, MP, wSV, wVS, num)
        
        
#     alpha_nPE = alpha_all["alpha_nPE"]
#     alpha_pPE = alpha_all["alpha_pPE"]
#     alpha_dMM = alpha_all["alpha_dMM"]
#     alpha_hMM = alpha_all["alpha_hMM"]
    
#     wPS_nPE = wPS_all["wPS_nPE"]
#     wPS_pPE = wPS_all["wPS_pPE"]
#     wPS_dMM = wPS_all["wPS_dMM"]
#     wPS_hMM = wPS_all["wPS_hMM"]
    
#     wPV_nPE = wPV_all["wPV_nPE"]
#     wPV_pPE = wPV_all["wPV_pPE"]
#     wPV_dMM = wPV_all["wPV_dMM"]
#     wPV_hMM = wPV_all["wPV_hMM"]
    
#     ax.scatter(alpha_dMM, wPS_dMM, wPV_dMM, edgecolor=Col_dMM, alpha=0.5, marker='.', depthshade=True, facecolor='none', s=ms, zorder=0)
#     ax.scatter(alpha_hMM, wPS_hMM, wPV_hMM, edgecolor=Col_hMM, alpha=0.5, marker='.', depthshade=True, facecolor='none', s=ms, zorder=0)
#     ax.scatter(alpha_nPE, wPS_nPE, wPV_nPE, facecolor=Col_Rate_nE, marker='.', depthshade=False, s=ms, zorder=20)
#     ax.scatter(alpha_pPE, wPS_pPE, wPV_pPE, facecolor=Col_Rate_pE, marker='.', depthshade=False, s=ms, zorder=20)

#     # examples
#     Color = [Col_Exc, Col_Inh, 'k']
#     marker = ['s','o','v']
#     if Examples is not None:
#         n_examples = np.size(Examples,2)
#         for i in range(n_examples):
#             ax.scatter(Examples[:,0,i], Examples[:,1,i], Examples[:,2,i], marker=marker[i], s=ms+3, color='k')
#             # ax.plot([Examples[0,0,i],Examples[0,0,i]], [Examples[0,1,i],Examples[0,1,i]], [0,Examples[0,2,i]],color='k',ls='--',lw=1)
#             # ax.scatter(0, Examples[:,1,i], Examples[:,2,i], marker='o', s=(ms+3)**2, color=Color[i])
#             # ax.scatter(Examples[:,0,i], 0, Examples[:,2,i], marker='o', s=(ms+3)**2, color=Color[i])
#             # ax.scatter(Examples[:,0,i], Examples[:,1,i], 0, marker='o', s=(ms+3)**2, color=Color[i])
            
#     ax.view_init(elev=19, azim=42) #(elev=30, azim=35)
#     ax.tick_params(axis='both', which='both', labelsize=fs)
#     ax.tick_params(axis='both', which='both', size=1.0, pad=-3)
#     ax.set_xlabel(r'(1+w$_\mathrm{PP}$)/w$_\mathrm{EP}$', fontsize=fs, labelpad=-6)
#     #ax.set_xlim([0,1])
#     ax.set_ylabel(r'w$_\mathrm{PS}$',fontsize=fs, labelpad=-8)
#     if ylab:
#         ax.zaxis.set_rotate_label(False)
#         ax.set_zlabel(r'w$_\mathrm{PV}$',fontsize=fs, labelpad=-8, rotation=90)

#     if axs is None:
#         PathFig = 'Results/Figures/' + folder
#         plt.savefig(PathFig + '/Plot_WeightPhasePlane_' + str(VE) + str(VP) + str(MP) + str(VS) + str(VV)+ '_' + fln + '.png', 
#                     bbox_inches='tight', transparent=True, dpi=300)  
    
#     return alpha_all, wPS_all, wPV_all 


# def Plot_ParametersPhasePlane(VE, VP, MP, VS, VV, wSV, wVS, num, 
#                               folder, fln=None, ylab=True, fs=7, ms=1, lw=1, G=None):

#     if G is None:
#         plt.figure(figsize=(5,15),tight_layout=True)#, sharex=True)
#         GG = gridspec.GridSpec(4,1)
#     else:
#         GG = G 
    
#     # plot phase plane
#     if ((VS==1) & (VV==0)):
#         alpha_all, wPS_all, wPV_all = DeterminePhasePlane_10(VE, VP, MP, wSV, wVS, num)
        
#     elif((VS==1) & (VV==1)):
#         alpha_all, wPS_all, wPV_all = DeterminePhasePlane_11(VE, VP, MP, wSV, wVS, num)
        
#     elif((VS==0) & (VV==0)):
#         alpha_all, wPS_all, wPV_all = DeterminePhasePlane_00(VE, VP, MP, wSV, wVS, num)
        
#     elif((VS==0) & (VV==1)):
#         alpha_all, wPS_all, wPV_all = DeterminePhasePlane_01(VE, VP, MP, wSV, wVS, num)
        
        
#     alpha_nPE = alpha_all["alpha_nPE"]
#     alpha_pPE = alpha_all["alpha_pPE"]
#     alpha_dMM = alpha_all["alpha_dMM"]
#     alpha_hMM = alpha_all["alpha_hMM"]
    
#     wPS_nPE = wPS_all["wPS_nPE"]
#     wPS_pPE = wPS_all["wPS_pPE"]
#     wPS_dMM = wPS_all["wPS_dMM"]
#     wPS_hMM = wPS_all["wPS_hMM"]
    
#     wPV_nPE = wPV_all["wPV_nPE"]
#     wPV_pPE = wPV_all["wPV_pPE"]
#     wPV_dMM = wPV_all["wPV_dMM"]
#     wPV_hMM = wPV_all["wPV_hMM"]
    
#     ax1 = plt.subplot(GG[0,0], projection='3d')
#     ax1.scatter(alpha_nPE, wPS_nPE, wPV_nPE, facecolor=Col_Rate_nE, marker='.', depthshade=False, s=ms)
#     ax1.scatter(alpha_pPE, wPS_pPE, wPV_pPE, facecolor=Col_Rate_pE, marker='.', depthshade=False, s=ms)
#     ax1.scatter(alpha_dMM, wPS_dMM, wPV_dMM, edgecolor=Col_dMM, marker='.', depthshade=False, facecolor='none', s=ms)
#     ax1.scatter(alpha_hMM, wPS_hMM, wPV_hMM, edgecolor=Col_hMM, marker='.', depthshade=False, facecolor='none', s=ms)
        
#     ax1.view_init(elev=30, azim=35)
#     ax1.tick_params(axis='both', which='both', labelsize=fs)
#     ax1.tick_params(axis='both', which='both', size=1.0, pad=-3)
#     ax1.set_xlabel(r'(1+w$_\mathrm{PP}$)/w$_\mathrm{EP}$', fontsize=fs, labelpad=-6)
#     ax1.set_ylabel(r'w$_\mathrm{PS}$',fontsize=fs, labelpad=-8)
#     if ylab:
#         ax1.set_zlabel(r'w$_\mathrm{PV}$',fontsize=fs, labelpad=-8)
    
#     ax2 = plt.subplot(GG[1,0])
#     ax2.scatter(alpha_nPE, wPS_nPE, facecolor=Col_Rate_nE, marker='.', s=ms)
#     ax2.scatter(alpha_pPE, wPS_pPE, facecolor=Col_Rate_pE, marker='.', s=ms)
#     ax2.scatter(alpha_dMM, wPS_dMM, edgecolor=Col_dMM, marker='.', facecolor='none', zorder=-10, s=ms)
#     ax2.scatter(alpha_hMM, wPS_hMM, edgecolor=Col_hMM, marker='.', facecolor='none', zorder=-10, s=ms)
    
#     ax2.tick_params(axis='both', which='both', labelsize=fs)
#     ax2.tick_params(axis='both', which='both', size=1.0)
#     ax2.set_xlabel(r'(1+w$_\mathrm{PP}$)/w$_\mathrm{EP}$', fontsize=fs)
#     if ylab:
#         ax2.set_ylabel(r'w$_\mathrm{PS}$',fontsize=fs)
#     ax2.tick_params(axis='both', which='both', size=2.0, labelsize = fs)
#     sns.despine(ax=ax2)
    
#     ax3 = plt.subplot(GG[2,0])
#     ax3.scatter(alpha_nPE, wPV_nPE, facecolor=Col_Rate_nE, marker='.', s=ms)
#     ax3.scatter(alpha_pPE, wPV_pPE, facecolor=Col_Rate_pE, marker='.', s=ms)
#     ax3.scatter(alpha_dMM, wPV_dMM, edgecolor=Col_dMM, marker='.', facecolor='none', zorder=-10, s=ms)
#     ax3.scatter(alpha_hMM, wPV_hMM, edgecolor=Col_hMM, marker='.', facecolor='none', zorder=-10, s=ms)
    
#     ax3.tick_params(axis='both', which='both', labelsize=fs)
#     ax3.tick_params(axis='both', which='both', size=1.0)
#     ax3.set_xlabel(r'(1+w$_\mathrm{PP}$)/w$_\mathrm{EP}$', fontsize=fs)
#     if ylab:
#         ax3.set_ylabel(r'w$_\mathrm{PV}$',fontsize=fs)
#     ax3.tick_params(axis='both', which='both', size=2.0, labelsize = fs)
#     sns.despine(ax=ax3)
    
#     ax4 = plt.subplot(GG[3,0])
#     ax4.scatter(wPS_nPE, wPV_nPE, facecolor=Col_Rate_nE, marker='.', s=ms)
#     ax4.scatter(wPS_pPE, wPV_pPE, facecolor=Col_Rate_pE, marker='.', s=ms)
#     ax4.scatter(wPS_dMM, wPV_dMM, edgecolor=Col_dMM, marker='.', facecolor='none', zorder=-10, s=ms)
#     ax4.scatter(wPS_hMM, wPV_hMM, edgecolor=Col_hMM, marker='.', facecolor='none', zorder=-10, s=ms)
    
#     ax4.tick_params(axis='both', which='both', labelsize=fs)
#     ax4.tick_params(axis='both', which='both', size=1.0)
#     ax4.set_xlabel(r'w$_\mathrm{PS}$', fontsize=fs)
#     if ylab:
#         ax4.set_ylabel(r'w$_\mathrm{PV}$',fontsize=fs)
#     ax4.tick_params(axis='both', which='both', size=2.0, labelsize = fs)
#     sns.despine(ax=ax4)
    
#     if G is None:
#         PathFig = 'Results/Figures/' + folder
#         plt.savefig(PathFig + '/Plot_ParametersPhasePlane_' + str(VE) + str(VP) + str(MP) + str(VS) + str(VV)+ '_' + fln + '.png', 
#                     bbox_inches='tight', transparent=True, dpi=300)  
    
#     return alpha_all, wPS_all, wPV_all 


# def Plot_weight_over_time(folder, fln, fs=7, ms=1, lw=1, ylab_flag=True, G=None):
    
#     if G is None:
#         plt.figure(figsize=(10,10),tight_layout=True)#, sharex=True)
#         GG = gridspec.GridSpec(4,1)
#     else:
#         GG = G 
    
#     PathData = 'Results/Data/' + folder
#     fln1 = PathData + '/Data_PlasticNetwork_' + fln + '.hdf'
#     f = tables.open_file(fln1, 'r')
#     t = f.root.weights[:-1,0]      
#     w = f.root.weights[:-1,1:]
#     N = np.size(w,1)
    
#     Ylab = [r'$\frac{\Delta\, w_\mathrm{EP}}{\Delta\, t}$', 
#             r'$\frac{\Delta\, w_\mathrm{DS}}{\Delta\, t}$', 
#             r'$\frac{\Delta\, w_\mathrm{PS}}{\Delta\, t}$', 
#             r'$\frac{\Delta\, w_\mathrm{PV}}{\Delta\, t}$']
    
#     for k in range(N):
        
#         ax = plt.subplot(GG[k,0])
#         weight = w[:,k]
#         ax.plot(t[:-1],np.diff(weight), color=Col_Inh, lw=lw)
        
#         ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#         ax.yaxis.get_offset_text().set_fontsize(fs)
#         ax.tick_params(axis='both', which='both', labelsize=fs)
#         ax.tick_params(axis='both', which='both', size=1.0)
#         sns.despine(ax=ax)
        
#         if k<N-1:
#             ax.set_xticklabels([])
#         else:
#             ax.set_xlabel('Time (sec)', fontsize=fs)
            
#         if ylab_flag:
#             ax.set_ylabel(Ylab[k], fontsize=fs+2)#, labelpad=8)
        
#     if G is None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + 'Plot_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300)  
        

# def Plot_InputIllustration(folder, FB_flag=1, MM_flag=1, PB_flag=1, xlab=False, ylab=False, leg=False, 
#                            figsize=(4,4), fs=7, ms=1, lw=1, axs=None):
    
#     if axs==None:
#         fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
#     else:
#         ax = axs
        
#     if PB_flag==1:
#         ax.fill_between([0,1], [0,1], color=Col_Input_V)
#     if MM_flag==1:
#         ax.fill_between([0,1], [1,1], [0,1], color=Col_Input_M)
#     if FB_flag==1:
#         ax.plot([0,1],[0,1], color=Col_Rate_E)
#     if xlab:
#         ax.set_xlabel('Sensory Input', fontsize=fs)
#     if ylab:
#         ax.set_ylabel('Prediction', fontsize=fs)
#     if leg:
#         ax.text(1.2,0.7,'Fully predicted', fontsize=fs, color=Col_Rate_E)
#         ax.text(1.2,0.5,'Overpredicted', fontsize=fs, color=Col_Input_M)
#         ax.text(1.2,0.3,'Underpredicted', fontsize=fs, color=Col_Input_V)
        
#     ax.set_xlim([0,1])
#     ax.set_ylim([0,1])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     #sns.despine(ax=ax)
    
#     if axs==None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + 'InputIllustration_' + str(FB_flag) + str(MM_flag) + str(PB_flag) + '.png', 
#                     bbox_inches='tight', transparent=True, dpi=300)
   
        
# def Plot_Numbers_DependenceWeights(flag_type, Num_nPE, Num_pPE, folder, fln, figsize=(4,4), fs=7, ms=1, 
#                                    lw=1, xlab_flag=True, ylab_flag=True, axs=None): # title=None, 
    
#     if axs==None:
#         fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
#     else:
#         ax = axs
        
#     ax.locator_params(axis='y', nbins=3)
#     ax.locator_params(axis='x', nbins=2)
    
#     ax.plot([1,0], Num_nPE[flag_type,:],'.-',color=Col_Rate_nE, lw=lw, markersize=ms, label='nPE')
#     ax.plot([1,0], Num_pPE[flag_type,:],'.-',color=Col_Rate_pE, lw=lw, markersize=ms, label='pPE')
    
#     if xlab_flag:
#         ax.set_xlabel(r'Scaling factor (w$_\mathrm{EP}$, w$_\mathrm{PS}$)',fontsize=fs)
#     if ylab_flag:
#         ax.set_ylabel('# cells (%)',fontsize=fs)
    
#     ax.set_xticks([0,1])
#     ax.set_xticklabels(['(1,1)','(2,8)'], fontsize=fs)
#     ax.set_xlim([-0.3,1.3])
    
#     ax.tick_params(axis='both', which='both', labelsize=fs)
#     ax.tick_params(axis='both', which='both', size=1.0)

#     # if title is not None:
#     #     ax.text(0.4,0.9, title, fontsize=fs, transform=ax.transAxes)
    
#     sns.despine(ax=ax)
    
#     if axs==None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + fln + '_Numbers.png', bbox_inches='tight', transparent=True, dpi=300)

# def Plot_SwitchAttractor_old(StimPar, SimPar, t, Pred, RE, bool_nPE, bool_pPE, dist1, dist2, 
#                              folder, fln, fs=7, ms=1, lw=1, G=None):
    
#     ### Pre-processing
#     stimuli = StimPar.stimuli
#     N_stimuli = len(stimuli)
#     distribution_flag = StimPar.distribution_flag
#     stim_duration = SimPar.stim_duration
    
#     mean_dist1 = np.mean(dist1)
#     mean_dist2 = np.mean(dist2)

#     Pred_initial = np.zeros((2, N_stimuli))
#     Pred_final = np.zeros((2, N_stimuli))
    
#     for i in range(N_stimuli):
#         Pred_initial[:,i] = np.mean(Pred[:, (t>i*stim_duration) & (t<(0.01+i)*stim_duration)], 1)
#         Pred_final[:,i] = np.mean(Pred[:, (t>(0.5+i)*stim_duration) & (t<(1+i)*stim_duration)], 1) 
    
#     ### Plot
#     if G is None:
#         plt.figure(figsize=(10,10),tight_layout=True)#, sharex=True)
#         GG = gridspec.GridSpec(2,1)
#     else:
#         GG = G 
    
#     for k in range(2):
        
#         if k==0:
#             ax = plt.subplot(GG[0,0])
#             Rate = Pred_initial
#         else:
#             ax = plt.subplot(GG[1,0])
#             Rate = Pred_final
    
#         for i in range(N_stimuli-1):
            
#             if distribution_flag[i]==0:
#                 col = cmap_weak(0.5)
#             elif distribution_flag[i]==1:
#                 col = cmap_strong(0.5)
#             # ax.plot([i,i+1],[stimuli[i],stimuli[i]],color=col, lw=lw)
#             # ax.plot([i+1,i+1],[stimuli[i],stimuli[i+1]],color='k',ls='--', alpha=0.5, lw=lw)
            
#             # if (np.diff(distribution_flag)[i]!=0):
#             #     ax.plot(i+1, 9, 'kv', ms=ms)
#             # if stimuli[i]==5:
#             #     ax.plot(i+0.5, 9, 'kx', ms=ms+1)
            
#         #ax.plot([N_stimuli-1,N_stimuli],[stimuli[N_stimuli-1],stimuli[N_stimuli-1]],color=col, lw=lw)
            
#         ax.plot(np.arange(0.5, N_stimuli, 1), Rate[0,:],'-o', color=cmap_weak(0.5), markersize=ms, lw=lw, label='Test')
#         ax.plot(np.arange(0.5, N_stimuli, 1), Rate[1,:],'-o', color=cmap_strong(0.5), markersize=ms, lw=lw, label='Testa')
#         #ax.axhline(5, color='k',ls=':', zorder=0, lw=lw)#, alpha=0.5)
#         ax.axhline(mean_dist1,color=cmap_weak(0.5),ls=':', zorder=0, lw=lw, label='Mean of dist. 1')#, alpha=0.5)
#         ax.axhline(mean_dist2,color=cmap_strong(0.5),ls=':', zorder=0, lw=lw, label='Mean of dist. 2')#, alpha=0.5)
        
#         # winner = 0 * (Rate[0,:]>Rate[1,:]) + 1 * (Rate[0,:]<Rate[1,:])
        
#         # for i in range(len(winner)):
#         #     if winner[i]==0:
#         #         col = '#F9EFE9' #cmap_weak(0.5)
#         #     elif winner[i]==1:
#         #         col = '#EAF0EF' #cmap_strong(0.5)
#         #     ax.axvspan(i,i+1,color=col, zorder=0) #, alpha=0.1)
        
#         for i in range(len(distribution_flag)):
#             if distribution_flag[i]==0:
#                 col = '#F9EFE9' #cmap_weak(0.5)
#             elif distribution_flag[i]==1:
#                 col = '#EAF0EF' #cmap_strong(0.5)
#             ax.axvspan(i,i+1,color=col, zorder=0) #, alpha=0.1)
            
#         ax.locator_params(axis='y', nbins=3)
#         ax.locator_params(axis='x', nbins=3)
#         ax.tick_params(axis='both', which='both', labelsize=fs)
#         ax.tick_params(axis='both', which='both', size=1.0)
        
#         if k!=0:
#             ax.set_xlabel('# Trials', fontsize=fs)
#         else:
#             ax.set_xticklabels([])
#         ax.set_ylabel('Prediction (1/s)', fontsize=fs) # 'Activity (1/s)'
        
#         ax.set_xlim([0,N_stimuli])
#         ax.set_ylim([-0.3,9])
        
#         sns.despine(ax=ax)
    
#     if G is None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + 'Plot_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300)  
    
# def Plot_CuedTransitions(StimPar, Out_per_stimulus, folder, fln, fs=7, ms=1, lw=1, axs=None):
    
#     if axs==None:
#         fig, ax = plt.subplots(1,1,tight_layout=True)
#     else:
#         ax = axs
    
#     Color = ['#DB4B2C', '#447DBC']
    
#     cue = StimPar.distribution_flag
#     stimuli = StimPar.stimuli
#     N_stimuli = len(stimuli)
#     Out = np.max(Out_per_stimulus,0)
    
#     for i in range(N_stimuli-1):
#         ax.plot([i,i+1],[stimuli[i],stimuli[i]],color=Color[cue[i]],lw=3)
#         ax.plot([i+1,i+1],[stimuli[i],stimuli[i+1]],color='k',ls='--', alpha=0.5)
        
#     ax.plot([N_stimuli-1,N_stimuli],[stimuli[N_stimuli-1],stimuli[N_stimuli-1]],color=Color[cue[i]-1],lw=3)
#     ax.plot(np.arange(0.5, N_stimuli, 1), Out, 'ok')
#     ax.axhspan(5,11,color='#447DBC', alpha=0.1)
#     ax.axhspan(1,5,color='#DB4B2C', alpha=0.1)
#     ax.axhline(3,color='#DB4B2C',ls=':', zorder=0, alpha=0.5)
#     ax.axhline(7,color='#447DBC',ls=':', zorder=0, alpha=0.5)
#     ax.set_ylim([0,9.1])
#     ax.set_xlim([0,30])
    
#     ax.set_xlabel('# Trials')
#     ax.set_ylabel('R input & output')
#     sns.despine(ax=ax)
    
#     if axs==None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + 'Plot_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300)
    

# def Plot_Inputs_MM_PB(Inp_MM, Inp_PB, bool_nPE, bool_pPE, folder, fln, figsize=(4,4), fs=7, ms=1, lw=1, 
#                       leg_x=0.5, leg_y=0.7, ylab=True, axs=None):
    
#     if axs==None:
#         fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
#     else:
#         ax = axs
        
#     ax.locator_params(axis='y', nbins=3)
#     ax.locator_params(axis='x', nbins=2)
    
#     ax.scatter(Inp_PB[bool_nPE==1], Inp_MM[bool_nPE==1], marker = 'o', color=Col_Rate_nE, zorder=20, s=ms)
#     ax.scatter(Inp_PB[bool_pPE==1], Inp_MM[bool_pPE==1], marker = 'o', color=Col_Rate_pE, zorder=20, s=ms)
#     ax.scatter(Inp_PB[(bool_pPE==0) & (bool_nPE==0)], Inp_MM[(bool_pPE==0) & (bool_nPE==0)], color='#B8B8B8', marker='o', s=ms, zorder=10)
    
#     x_range = ax.get_xlim()
#     x = np.linspace(x_range[0], x_range[1],10)
    
#     if sum(bool_nPE)>7:
#         m_nPE, n_nPE = np.polyfit(Inp_PB[bool_nPE==1], Inp_MM[bool_nPE==1],1)
#         ax.plot(x, m_nPE * x + n_nPE, '--', color=Col_Rate_nE, label='R = ' + str(np.round(m_nPE,2)), zorder=1, lw=lw)
#     if sum(bool_pPE)>7:
#         m_pPE, n_pPE = np.polyfit(Inp_PB[bool_pPE==1], Inp_MM[bool_pPE==1],1)
#         ax.plot(x, m_pPE * x + n_pPE, '--', color=Col_Rate_pE, label='R = ' + str(np.round(m_pPE,2)), zorder=1, lw=lw)
    
#     m_all, n_all = np.polyfit(Inp_PB, Inp_MM,1)
#     ax.plot(x, m_all * x + n_all, '--', color='#B8B8B8', label='R = ' + str(np.round(m_all,2)), zorder=1, lw=lw)
    
#     ax.axhline(0, color='k', ls=':', alpha=0.2, zorder=0)
#     ax.axvline(0, color='k', ls=':', alpha=0.2, zorder=0)
    
#     ax.set_xlabel('Input UP (1/s)', fontsize=fs)
#     if ylab:
#         ax.set_ylabel('Input OP (1/s)', fontsize=fs)
    
#     ax.tick_params(axis='both', which='both', labelsize=fs)
#     ax.tick_params(axis='both', which='both', size=1.0)
#     ax.legend(loc=[leg_x,leg_y], fontsize=fs-2, frameon=False)
    
#     ax.set_xlim(x_range)
#     sns.despine(ax=ax)
    
#     if axs==None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + 'Plot_Inputs_MMPB_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=300)
    

# def Plot_Experience_NumPE(SD_all, Num_nPE, Num_pPE, folder, fln, mean=None, figsize=(4,4),
#                           ylab=False, fs=7, ms=1, lw=1, axs=None):
    
#     if axs==None:
#         fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
#     else:
#         ax = axs
        
#     ax.locator_params(axis='y', nbins=4)
#     ax.locator_params(axis='x', nbins=2)
    
#     if mean==None:
#         ax.plot(SD_all, np.mean(Num_nPE,1),'.-',color=Col_Rate_nE, lw=lw, markersize=ms, label='nPE')
#         ax.plot(SD_all, np.mean(Num_pPE,1),'.-',color=Col_Rate_pE, lw=lw, markersize=ms, label='pPE')
#         ax.set_xlabel('Input SD (1/s)',fontsize=fs)
#     else:
#         ax.plot(SD_all/mean, np.mean(Num_nPE,1),'.-',color=Col_Rate_nE, lw=lw, markersize=ms, label='nPE')
#         ax.plot(SD_all/mean, np.mean(Num_pPE,1),'.-',color=Col_Rate_pE, lw=lw, markersize=ms, label='pPE')
#         ax.set_xlabel('Noise, CV',fontsize=fs)
    
#     if ylab:
#         ax.set_ylabel('# Cells (%)',fontsize=fs)
#     ax.set_ylim(bottom=0)
#     ax.tick_params(axis='both', which='both', labelsize=fs)
#     ax.tick_params(axis='both', which='both', size=1.0)
#     ax.legend(loc=0, fontsize=fs, handlelength=1, frameon=False)
    
#     sns.despine(ax=ax)
    
#     if axs==None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + fln + '_Numbers.png', bbox_inches='tight', transparent=True, dpi=300)
        
        
# def Plot_Experience_Reliability_Num(p_out, Num_nPE, Num_pPE, folder, fln, fs=7, ms=1, lw=1, ylab_flag=True, axs=None): 

#     ### Plot responses for all phases
#     N = np.size(Num_nPE,1)
          
#     ### Plot number of nPE and pPE neurons
#     if axs==None:
#         fig, ax = plt.subplots(tight_layout=True)
#     else:
#         ax = axs
    
#     ax.locator_params(axis='y', nbins=3)
#     ax.locator_params(axis='x', nbins=2)
    
#     mean, SEM = np.mean(Num_nPE,1), np.std(Num_nPE,1)/np.sqrt(N)
#     ax.plot((1-p_out), mean,'.-',color=Col_Rate_nE, lw=lw, markersize=ms)
#     ax.plot((1-p_out), mean-SEM,'_', color=Col_Rate_nE, markersize=ms, lw=lw)
#     ax.plot((1-p_out), mean+SEM,'_', color=Col_Rate_nE, markersize=ms, lw=lw)
#     ax.plot(np.tile((1-p_out),(2,1)), np.stack((mean-SEM, mean+SEM), axis=0),'-', color = Col_Rate_nE, lw=lw)
#     #ax.fill_between(1-p_out, mean-SEM, mean+SEM, facecolor=Col_Rate_nE, color=Col_Rate_nE, alpha=0.4, lw=lw)
    
#     mean, SEM = np.mean(Num_pPE,1), np.std(Num_pPE,1)/np.sqrt(N)
#     ax.plot((1-p_out), mean,'.-',color=Col_Rate_pE, lw=lw, markersize=ms)
#     ax.plot((1-p_out), mean-SEM,'_', color=Col_Rate_pE, markersize=ms, lw=lw)
#     ax.plot((1-p_out), mean+SEM,'_', color=Col_Rate_pE, markersize=ms, lw=lw)
#     ax.plot(np.tile((1-p_out),(2,1)), np.stack((mean-SEM, mean+SEM), axis=0),'-', color = Col_Rate_pE, lw=lw)
#     #ax.fill_between(1-p_out, mean-SEM, mean+SEM, facecolor=Col_Rate_pE, color=Col_Rate_pE, alpha=0.4, lw=lw)
    
#     ax.set_xlabel('Reliability', fontsize=fs)
#     if ylab_flag:
#         ax.set_ylabel('# Cells (%)', fontsize=fs)
#     ax.tick_params(axis='both', which='both', size=1.0, pad=2.0)
#     ax.tick_params(axis='both', which='both', labelsize=fs)
#     sns.despine(ax=ax)
    
#     if axs==None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + fln + '_Num.png', bbox_inches='tight', transparent=True, dpi=300)
            
    

# def Plot_Experience_Reliability(p_out, Mean_Phases_nPE, Mean_Phases_pPE, SEM_Phases_nPE, SEM_Phases_pPE, Num_nPE, Num_pPE,
#                                 folder, fln, flag_leg=1, figsize=(4,4), fs=7, ms=1, lw=1, binary_plot=[1,1,1,1], leg_lab=[0,0,0,0],
#                                 ylab_flag = [1,1,1,1], xlab_flag = [1,1,1,1], y_max = None,  G=None, axs=None): #ax_FB=None, ax_MM=None, ax_PB=None, ax_num=None):

#     ### Plot responses for all phases
#     PhaseName = ['Fully predicted', 'Overpredicted', 'Underpredicted']
#     N = np.size(Num_nPE,1)
    
#     if G is None:
#         plt.figure(figsize=figsize,tight_layout=True)
#         GG = gridspec.GridSpec(1, 3)#, sharey=True)
#     else:
#         GG = G 
    
#     #count = -1
#     for i in range(3):
    
#         if binary_plot[i]==1:
            
#             #count += 1
#             ax = plt.subplot(GG[0,i])
             
#             ax.locator_params(axis='y', nbins=3)
#             ax.locator_params(axis='x', nbins=3)
    
#             # mean, SEM = Mean_Phases_nPE[:,i+1], SEM_Phases_nPE[:,i+1]
#             mean, SEM = np.mean(Mean_Phases_nPE[:,i+1,:],1), np.std(Mean_Phases_nPE[:,i+1,:],1)/np.sqrt(N)
#             ax.plot((1-p_out)*100, mean,'.-',color=Col_Rate_nE,label='nPE', lw=lw, markersize=ms)
#             ax.plot((1-p_out)*100, mean-SEM,'_', color=Col_Rate_nE, markersize=ms, lw=lw)
#             ax.plot((1-p_out)*100, mean+SEM,'_', color=Col_Rate_nE, markersize=ms, lw=lw)
#             ax.plot(np.tile((1-p_out)*100,(2,1)), np.stack((mean-SEM, mean+SEM), axis=0),'-', color = Col_Rate_nE, lw=lw)
#             # ax.fill_between(1-p_out, mean-SEM, mean+SEM, facecolor=Col_Rate_nE, color=Col_Rate_nE, alpha=0.4, lw=lw)
            
#             #mean, SEM = Mean_Phases_pPE[:,i+1], SEM_Phases_pPE[:,i+1]
#             mean, SEM = np.mean(Mean_Phases_pPE[:,i+1,:],1), np.std(SEM_Phases_pPE[:,i+1,:],1)/np.sqrt(N)
#             ax.plot((1-p_out)*100, mean,'.-',color=Col_Rate_pE,label='pPE', lw=lw, markersize=ms)
#             ax.plot((1-p_out)*100, mean-SEM,'_', color=Col_Rate_pE, markersize=ms, lw=lw)
#             ax.plot((1-p_out)*100, mean+SEM,'_', color=Col_Rate_pE, markersize=ms, lw=lw)
#             ax.plot(np.tile((1-p_out)*100,(2,1)), np.stack((mean-SEM, mean+SEM), axis=0),'-', color = Col_Rate_pE, lw=lw)
#             # ax.fill_between(1-p_out, mean-SEM, mean+SEM, facecolor=Col_Rate_pE, color=Col_Rate_pE, alpha=0.4, lw=lw)
                
#             ax.set_ylim([-0.2, y_max])
#             ax.tick_params(axis='both', which='both', size=1.0, pad=2.0)
#             if xlab_flag[i]==1:
#                 ax.set_xlabel('# FP (%)', fontsize=fs)
#             else:
#                 ax.set_xticklabels([])
#                 ax.set_title(PhaseName[i], fontsize=fs, pad=0)
#             if ylab_flag[i]==1:
#                 ax.set_ylabel('Rate (1/s)', fontsize=fs)
#             ax.tick_params(axis='both', which='both', labelsize=fs)
#             sns.despine(ax=ax)
                       
#             if leg_lab[i]==1:
#                 ax.legend(loc=0, fontsize=fs, handlelength=1, frameon=False)
                
#             if binary_plot[i-1]==1:
#                 ax.set_yticklabels([])
                
#     if G==None:
#         PathFig = 'Results/Figures/' + folder + '/'
#         plt.savefig(PathFig + fln + '_Rates.png', bbox_inches='tight', transparent=True, dpi=300)
            
#     ### Plot number of nPE and pPE neurons
#     if binary_plot[-1]==1:
        
#         if axs==None:
#             fig, ax = plt.subplots(tight_layout=True)
#         else:
#             ax = axs
        
#         ax.locator_params(axis='y', nbins=3)
#         ax.locator_params(axis='x', nbins=2)
        
#         mean, SEM = np.mean(Num_nPE,1), np.std(Num_nPE,1)/np.sqrt(N)
#         ax.plot((1-p_out)*100, mean,'.-',color=Col_Rate_nE, lw=lw, markersize=ms)
#         ax.plot((1-p_out)*100, mean-SEM,'_', color=Col_Rate_nE, markersize=ms, lw=lw)
#         ax.plot((1-p_out)*100, mean+SEM,'_', color=Col_Rate_nE, markersize=ms, lw=lw)
#         ax.plot(np.tile((1-p_out)*100,(2,1)), np.stack((mean-SEM, mean+SEM), axis=0),'-', color = Col_Rate_nE, lw=lw)
#         #ax.fill_between(1-p_out, mean-SEM, mean+SEM, facecolor=Col_Rate_nE, color=Col_Rate_nE, alpha=0.4, lw=lw)
        
#         mean, SEM = np.mean(Num_pPE,1), np.std(Num_pPE,1)/np.sqrt(N)
#         ax.plot((1-p_out)*100, mean,'.-',color=Col_Rate_pE, lw=lw, markersize=ms)
#         ax.plot((1-p_out)*100, mean-SEM,'_', color=Col_Rate_pE, markersize=ms, lw=lw)
#         ax.plot((1-p_out)*100, mean+SEM,'_', color=Col_Rate_pE, markersize=ms, lw=lw)
#         ax.plot(np.tile((1-p_out)*100,(2,1)), np.stack((mean-SEM, mean+SEM), axis=0),'-', color = Col_Rate_pE, lw=lw)
#         #ax.fill_between(1-p_out, mean-SEM, mean+SEM, facecolor=Col_Rate_pE, color=Col_Rate_pE, alpha=0.4, lw=lw)
        
#         if xlab_flag[-1]==1:
#             ax.set_xlabel('# FP (%)', fontsize=fs)
#         else:
#             ax.set_xticklabels([])
#         if ylab_flag[-1]==1:
#             ax.set_ylabel('# cells (%)', fontsize=fs)
#         ax.tick_params(axis='both', which='both', size=1.0, pad=2.0)
#         ax.tick_params(axis='both', which='both', labelsize=fs)
#         sns.despine(ax=ax)
        
#         if axs==None:
#             PathFig = 'Results/Figures/' + folder + '/'
#             plt.savefig(PathFig + fln + '_Num.png', bbox_inches='tight', transparent=True, dpi=300)


# def Plot_WeightConvergence(NetPar, LearnPar, folder, fln, fs=7, ms=1, lw=1,
#                            figsize=(6.4,4.8), ax=None, dpi = 500):
    
#     ### Define color scheme
#     Col = ['#76D6FF', '#7A81FF', Col_Rate_E, '#0096FF','#011993']
    
#     ### Load data
#     PathData = 'Results/Data/' + folder
#     fln1 = PathData + '/Data_PlasticNetwork_' + fln + '.hdf'
#     f = tables.open_file(fln1, 'r')
#     t = f.root.weights[:-1,0]      
#     w = f.root.weights[:-1,1:]
    
#     Lab = NetPar.weight_name[LearnPar.plastic_flag!=0]
    
#     ### Plot
#     if ax==None:
#         fig, ax = plt.subplots(tight_layout=True)
#     else:
#         ax = ax
         
#     for k in range(np.size(w,1)):
#         weight = w[:,k]/w[0,k]
#         ax.plot(t,weight,c=Col[k],label=Lab[k]) #/(1000*60*60)

#     ax.set_xlabel('Time (ms)',fontsize=fs)
#     ax.set_ylabel(r'w$_\mathrm{sim}$ (normalised)',fontsize=fs)
#     ax.tick_params(axis='both', which='both', size=2.0)
#     ax.legend(loc=0)
#     sns.despine(ax=ax)
    
#     ### Save
#     PathFig = 'Results/Figures/' + folder
#     if not os.path.exists(PathFig):
#         os.mkdir(PathFig)
    
#     plt.savefig(PathFig + '/Convergence_' + fln + '.png', bbox_inches='tight', transparent=True, dpi=dpi)
#     plt.close()
      

# def Plot_Manipulation_Space_Summary_old(filename_save, Inputs_nPE, Inputs_pPE, stim_extra_all, show=0, 
#                                     fs=7, ms=1, lw=1, figsize=(6.4,4.8), ax=None):
    
#     N = np.size(Inputs_nPE,2)
    
#     #Label = ['nEs', 'pEs', 'nEd', 'pEd', 'PVv', 'PVm', 'SOM', 'VIP']
#     Label = ['1', '2', '3', '4', '5', '6', '7', '8']
    
#     if ax==None:
#         fig, ax = plt.subplots(tight_layout=True)
#     else:
#         ax = ax
    
#     ax.plot([0,0],[0,9],'k', lw=lw)
#     ax.plot([0,9],[0,0],'k', lw=lw)
    
#     theta = np.linspace(0, 2*np.pi, 100)
    
#     for i in range(N):
        
#         nPE_V, nPE_M = Inputs_nPE[3,:,i]-Inputs_nPE[0,:,i], Inputs_nPE[2,:,i]-Inputs_nPE[0,:,i]
#         pPE_V, pPE_M = Inputs_pPE[3,:,i]-Inputs_pPE[0,:,i], Inputs_pPE[2,:,i]-Inputs_pPE[0,:,i]
    
#         angle_nPE = np.arctan2(nPE_M,nPE_V)
#         angle_pPE = np.arctan2(pPE_M,pPE_V)
        
#         x_nPE = (i+1)*np.cos(angle_nPE)
#         y_nPE = (i+1)*np.sin(angle_nPE)
#         x_pPE = (i+1)*np.cos(angle_pPE)
#         y_pPE = (i+1)*np.sin(angle_pPE)
#         x1 = (i+1)*np.cos(theta)
#         x2 = (i+1)*np.sin(theta)
        
#         if show==1:
#             x_nPE[stim_extra_all<0] *= 100
#             y_nPE[stim_extra_all<0] *= 100
#             x_pPE[stim_extra_all<0] *= 100
#             y_pPE[stim_extra_all<0] *= 100
#         elif show==-1:
#             x_nPE[stim_extra_all>0] *= 100
#             y_nPE[stim_extra_all>0] *= 100
#             x_pPE[stim_extra_all>0] *= 100
#             y_pPE[stim_extra_all>0] *= 100
    
#         ax.plot(x1, x2, 'k', lw=lw)
#         if (i%2)==1:
#             ax.text((i+2)*np.cos(-235),(i+2)*np.sin(-235),Label[i],fontsize=fs-1)
#         ax.scatter(x_nPE,y_nPE, marker='s', s=ms**2, c=stim_extra_all, cmap=Perturbations, zorder=50) # coolwarm
#         ax.scatter(x_pPE,y_pPE, marker='o', s=ms**2, c=stim_extra_all, cmap=Perturbations, zorder=50)
        
#         ax.axhspan(ymin=0,ymax=9, xmin=0, xmax=0.5, facecolor='#FFFBEB')#, alpha=0.5)
#         ax.axvspan(xmin=0,xmax=9, ymin=0,ymax=0.5, facecolor='#FFFBEB')#, alpha=0.5)
        
#         ax.axis('off')
        
#     ax.set_xlim([-9,9])
#     ax.set_ylim([-9,9])
#     ax.set_aspect('equal', adjustable='box')
    
#     ax.annotate('0°', xy=(9,0.2), fontsize=fs)
#     ax.annotate('90°', xy=(0.2,9), fontsize=fs)
    
#     sns.despine(ax=ax)
    
#     if ax==None:
#         if show==0:
#             plt.savefig(filename_save + '_All_' + Label[i] + '.png', bbox_inches='tight', transparent=True, dpi=300)
#         elif show==1:
#             plt.savefig(filename_save + '_Positive_' + Label[i] + '.png', bbox_inches='tight', transparent=True, dpi=300)
#         elif show==-1:
#             plt.savefig(filename_save + '_Negative_' + Label[i] + '.png', bbox_inches='tight', transparent=True, dpi=300)


# def Plot_Manipulation_Space(filename_save, Inputs_nPE, Inputs_pPE, stim_extra_all, ind=np.arange(8), 
#                             fs=7, ms=1, lw=1, cbar_flag=1, figsize=(6.4,4.8), ax=None):
    
#     Label = ['nEs', 'pEs', 'nEd', 'pEd', 'PVv', 'PVm', 'SOM', 'VIP']
    
#     for i in ind:
        
#         if ax==None:
#             fig, ax = plt.subplots(tight_layout=True)
#         else:
#             ax = ax
        
#         ax.spines['left'].set_position('zero')
#         ax.spines['bottom'].set_position('zero')
        
#         nPE_V, nPE_M = Inputs_nPE[3,:,i]-Inputs_nPE[0,:,i], Inputs_nPE[2,:,i]-Inputs_nPE[0,:,i]
#         pPE_V, pPE_M = Inputs_pPE[3,:,i]-Inputs_pPE[0,:,i], Inputs_pPE[2,:,i]-Inputs_pPE[0,:,i]
        
#         ax.scatter(nPE_V, nPE_M, marker='s', s=ms**2, c=stim_extra_all, cmap=Perturbations, zorder=50)
#         ax.scatter(pPE_V, pPE_M, marker='o', s=ms**2, c=stim_extra_all, cmap=Perturbations, zorder=50)
        
#         ax.plot(np.nan,np.nan, 's', color='k',label='nPE (w/o pert.)', ms=ms)
#         ax.plot(np.nan,np.nan, 'o', color='k',label='pPE (w/o pert.)', ms=ms)
#         ax.tick_params(axis='both', which='both', labelsize=fs)
#         ax.tick_params(axis='both', which='both', size=2.0)
        
#         x_max = 1.1*np.max([abs(nPE_V),abs(pPE_V)])
#         y_max = 1.1*np.max([abs(nPE_M),abs(pPE_M)])
#         ax.set_xlim([-x_max,x_max])
#         ax.set_ylim([-y_max,y_max])
        
#         yticks_all = ax.yaxis.get_major_ticks()
#         len_yticks = len(yticks_all)
#         yticks_all[int(len_yticks/2)].label1.set_visible(False)
        
#         xticks_all = ax.xaxis.get_major_ticks()
#         len_xticks = len(xticks_all)
#         xticks_all[int(len_xticks/2)].label1.set_visible(False)
        
#         #ax.legend(loc=3, frameon=False, fontsize=fs)
        
#         ax.axhspan(ymin=0,ymax=y_max, xmin=0, xmax=0.5, facecolor='#FFFBEB')
#         ax.axvspan(xmin=0,xmax=x_max, ymin=0,ymax=0.5, facecolor='#FFFBEB',)

#         sns.despine(ax=ax)
        
#         if cbar_flag==1:
#             ax_in = inset_axes(ax, width="30%", height="3%",loc=1) # loc=3
#             norm = mpl.colors.Normalize(vmin=stim_extra_all[0], vmax=stim_extra_all[-1])
#             cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=Perturbations),
#                               cax=ax_in, orientation='horizontal')
#             ax_in.tick_params(size=2.0,pad=2.0)
#             ax_in.tick_params(axis='both', which='both', labelsize=fs)
#             cb.ax.set_title('Perturbation (1/s)', fontsize=fs)
#             cb.outline.set_visible(False)
        
#         if ax==None:
#             plt.savefig(filename_save + '_' + Label[i] + '.png', bbox_inches='tight', transparent=True, dpi=300)
             