# -*- coding: utf-8 -*-
"""
Plot radiance before and after PCA filter
"""

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc')
import utils as utl
from datetime import datetime
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np

import matplotlib
import warnings

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12
# plt.close('all')

#%% Inputs
source_ch1=os.path.join(cd,'data/rhod.assist.z01.00/ch1/assistChA.20240514.000103.cdf')
source_ch1nf=os.path.join(cd,'data/rhod.assist.z01.00/ch1nf/assistChA.20240514.000103.cdf')

#graphics
wnum_min=600
wnum_max=1000
N_plot=5

#%% Initialization
Data=xr.open_dataset(source_ch1)
Data_nf=xr.open_dataset(source_ch1nf)

Data=Data.where(Data.wnum>wnum_min,drop=True).where(Data.wnum<wnum_max,drop=True).sortby('time')
Data_nf=Data_nf.where(Data.wnum>wnum_min,drop=True).where(Data.wnum<wnum_max,drop=True).sortby('time')


bt=Data['base_time'].values[0]/10**3
time=[datetime.utcfromtimestamp(t+bt) for t in Data['time'].values]

#%% Main
rad=Data.mean_rad.where(Data.sceneMirrorAngle==0)
rad_nf=Data_nf.mean_rad.where(Data_nf.sceneMirrorAngle==0)

rad_diff=rad_nf-rad

#%% Plots
i_plot=np.int64(np.linspace(0,len(Data.wnum)-1,N_plot))
plt.figure(figsize=(18,10))
ctr=1
for i in i_plot:
    plt.subplot(N_plot,1,ctr)
    
    plt.plot(time,rad[:,i],'.k',label='Raw',markersize=1)
    plt.plot(time,rad_nf[:,i],'.g',label='Filtered',markersize=1)
    plt.plot(time,rad_diff[:,i],'.r',label='Difference',markersize=1)
    plt.text(time[10],100,s=r'$\tilde{\nu}=$'+str(np.round(Data.wnum.values[i]))+' cm$^{-1}$')
    plt.grid()
    plt.ylim([-10,150])
    plt.ylabel('Radiance [r.u.]')
    if i!=i_plot[-1]:
        plt.gca().set_xticklabels([])
    ctr+=1
plt.legend()
plt.tight_layout()    
    
    

