# -*- coding: utf-8 -*-
"""
Plot data quality of ASSIST
"""
import os
cd=os.getcwd()
import sys
from doe_dap_dl import DAP
import xarray as xr
import numpy as np
import yaml
import glob
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import warnings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
channel='wfip3/barg.assist.z01.a0'
source_config=os.path.join(cd,'configs/config.yaml')
sdate='20240101000000'
edate='20250101000000'
date_break='2024-08-26T00:00:00'

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils']) 
import utils as utl

#DAP
a2e = DAP('a2e.energy.gov',confirm_downloads=False)
a2e.setup_cert_auth(username=config['password'], password=config['username'])

#%% Main

#downloads
_filter = {'Dataset': channel,
            'date_time': {
                'between': [sdate,edate],
            },
            'file_type': 'nc',
        }

a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel), replace=False)

#load data
files=glob.glob(os.path.join(cd,'data',channel,'*nc'))
Data=xr.open_mfdataset(files)
Data=Data.where(Data['hatch_status']==1)

#%% Plots
plt.close('all')
plt.figure(figsize=(18,6))
plt.plot(Data.time,Data.sw_responsivity,'.k',markersize=1)
plt.plot([np.datetime64(date_break),np.datetime64(date_break)],[0,1.2*10**5],'--r')
date_fmt = mdates.DateFormatter('%m/%d')
plt.gca().xaxis.set_major_formatter(date_fmt)
plt.grid()
plt.ylabel(r'Short-wave responsivity [r.u. counts$^{-1}$]')
plt.xlabel('Time (UTC)')
plt.ylim([0.95*10**5,1.1*10**5])
plt.xticks(np.arange(Data.time.values[0],Data.time.values[-1],np.timedelta64(3600*24*7, 's'),dtype='datetime64'))

plt.figure(figsize=(10,6))
plt.plot(Data.time.where(Data.time>=np.datetime64(date_break)),Data.sw_responsivity.where(Data.time>=np.datetime64(date_break)),'.k',markersize=1)
plt.gca().xaxis.set_major_formatter(date_fmt)
plt.grid()
plt.ylabel(r'Short-wave responsivity [r.u. counts$^{-1}$]')
plt.xlabel('Time (UTC)')
plt.ylim([1*10**5,1.1*10**5])


plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
plt.hist(Data.sw_responsivity.where(Data.time<np.datetime64(date_break)),bins=np.arange(0.95*10**5,10.1**5,100),color='g')
plt.ylabel('Occurrence')
plt.title('Before '+str(date_break.replace('T',' ')))
plt.xlim([0.95*10**5,1.1*10**5])
plt.grid()
plt.subplot(2,1,2)
plt.hist(Data.sw_responsivity.where(Data.time>=np.datetime64(date_break)),bins=np.arange(0.95*10**5,10.2**5,100),color='r')
plt.xlim([0.95*10**5,1.1*10**5])
plt.xlabel(r'Short-wave responsivity [r.u. counts$^{-1}$]')
plt.ylabel('Occurrence')
plt.title('After '+str(date_break.replace('T',' ')))
plt.grid()