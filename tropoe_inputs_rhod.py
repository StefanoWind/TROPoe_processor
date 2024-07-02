# -*- coding: utf-8 -*-
import os
cd=os.path.dirname(__file__)
import sys
import shutil
import pandas as pd
from datetime import datetime
from datetime import timedelta
import logging
import subprocess
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import glob

import matplotlib
import warnings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
channel_assist='wfip3/rhod.assist.z01.00'
channel_lidar= 'wfip3/rhod.lidar.z02.a0'
channel_met=   'wfip3/rhod.met.z01.00'

site='rhod'

server=False

if server:
    path_python='/home/sletizia/anaconda3/bin/python3.11'
    path_dap_py='/home/sletizia/codes/dap-py'
    path_utils='/home/sletizia/codes/utils'
else:
    path_python='C:/ProgramData/Anaconda3/python'
    path_dap_py='C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions/dap-py'
    path_utils='C:/Users/SlETIZIA/OneDrive - NREL/Desktop/PostDoc/utils'
    
path_cbh=os.path.join(cd,'cbh')
path_nsf=os.path.join(cd,'irs_nf')

download=False

met_headers={'temperature':11,'pressure':8,'humidity':12}

date='2024-05-20'#date to process

#download
username = 'sletizia'
password = 'pass_DAP1506@'

#processing
N_days_pca=8#number of days for pca filter

#%% Initialization

if date=='':
    date=sys.argv[1]

#imports
sys.path.append(path_utils) 
sys.path.append(path_dap_py) 
sys.path.append(path_cbh) 
sys.path.append(path_nsf) 
import utils as utl
from doe_dap_dl import DAP
import cbh_halo as cbh

if download:
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    a2e.setup_cert_auth(username=username, password=password)
    
if os.path.exists('data/processed-rhod.txt'):
    with open('data/processed-rhod.txt') as fid:
        processed_cbh=fid.readlines()
    processed_cbh=[p.strip() for p in processed_cbh]
else: 
    processed_cbh=[]

#%% Main
if download:
    #download assists summary data
    time_range_dap = [datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y%m%d%H%M%S'),
                      datetime.strftime(datetime.strptime(date, '%Y-%m-%d')+timedelta(days=1),'%Y%m%d%H%M%S')]
    
    time_range_dap_pca = [datetime.strftime(datetime.strptime(date, '%Y-%m-%d')-timedelta(days=N_days_pca),'%Y%m%d%H%M%S'),
                      datetime.strftime(datetime.strptime(date, '%Y-%m-%d')+timedelta(days=1),'%Y%m%d%H%M%S')]
    
    _filter = {
                'Dataset': channel_assist,
                'date_time': {
                    'between': time_range_dap_pca
                },
                'file_type': 'cdf',
                'ext1':['assistsummary','assistcha'],
            }
    
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel_assist), replace=False)

#rename and copy assist summary files
files=glob.glob(os.path.join(cd,'data',channel_assist,'*cdf'))

for f in files:
    old_name=os.path.basename(f).split('.')
    new_name=old_name[-2]+'.'+old_name[-4]+'.'+old_name[-3]+'.cdf'
    
    if not os.path.exists(os.path.join(cd,'data',channel_assist,'ch1')):
        os.mkdir(os.path.join(cd,'data',channel_assist,'ch1'))
        os.mkdir(os.path.join(cd,'data',channel_assist,'sum'))
    
    if 'assistcha' in f:
        shutil.copy(f,os.path.join(cd,'data',channel_assist,'ch1',new_name))
    elif 'assistsummary' in f:
        shutil.copy(f,os.path.join(cd,'data',channel_assist,'sum',new_name))
  
#run PCA filter
print('PCA filter')
sdate=datetime.strftime(datetime.strptime(date,'%Y-%m-%d')-timedelta(days=N_days_pca),'%Y%m%d')
edate=datetime.strftime(datetime.strptime(date,'%Y-%m-%d'),'%Y%m%d')
chassistdir=os.path.join(cd,'data',channel_assist,'ch1')
sumassistdir=os.path.join(cd,'data',channel_assist,'sum')
nfchassistdir=os.path.join(cd,'data',channel_assist,'nfc')

command=path_python+f' run_irs_nf.py --create {sdate} {edate} {chassistdir} {sumassistdir} {nfchassistdir} "assist"'

result = subprocess.run(command, shell=True, text=True,capture_output=True)
print(result.stdout)
print(result.stderr)

command=path_python+f' run_irs_nf.py --apply {sdate} {edate} {chassistdir} {sumassistdir} {nfchassistdir} "assist"'
result = subprocess.run(command, shell=True, text=True,capture_output=True)
print(result.stdout)
print(result.stderr)

#download lidar data (for cbh)
if download:    
    _filter = {
                'Dataset': channel_lidar,
                'date_time': {
                    'between': time_range_dap
                },
                'file_type': 'nc',
            }
    
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel_lidar), replace=False)

#estimate cbh
print('Estimating CBH height')
if not os.path.exists(os.path.join(cd,'data',channel_lidar.replace('a0','cbh'))):
    os.mkdir(os.path.join(cd,'data',channel_lidar.replace('a0','cbh')))
     
files=glob.glob(os.path.join(cd,'data',channel_lidar,'*nc'))
tnum_cbh_all=[]
cbh_all=[]
for f in files:
    time_cbh,cbh_lidar=cbh.compute_cbh(f,utl,plot=False)
    tnum_cbh_all=np.append(tnum_cbh_all,(time_cbh-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1, 's'))
    cbh_all=np.append(cbh_all,cbh_lidar)
    # with open('data/processed_cbh.txt', 'a') as fid:
    #     fid.write(os.path.basename(f)+'\n')

basetime_cbh=utl.floor(tnum_cbh_all[0],24*3600)
time_offset_cbh=tnum_cbh_all-basetime_cbh

Output_cbh=xr.Dataset()
Output_cbh['first_cbh']=xr.DataArray(data=np.int32(np.nan_to_num(cbh_all,nan=-9999)),
                                     coords={'time':np.int32(time_offset_cbh)})

# Output_cbh['time_offset']=xr.DataArray(data=time_offset_cbh,
#                                      coords={'time':np.arange(len(time_offset_cbh))})

Output_cbh['base_time']=np.int64(basetime_cbh)
Output_cbh.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
Output_cbh.to_netcdf(os.path.join(cd,'data',channel_lidar.replace('a0','cbh'),channel_lidar.replace('a0','ceil').replace('wfip3/','')+'.'+utl.datestr(basetime_cbh,'%Y%m%d.%H%M%S')+'.nc'))

#download met data
if download:
    
    _filter = {
                'Dataset': channel_met,
                'date_time': {
                    'between': time_range_dap
                },
                'file_type': 'csv',
            }
    
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel_met), replace=False)

           
print('Generating met data files')
if not os.path.exists(os.path.join(cd,'data',channel_met.replace('00','sel'))):
    os.mkdir(os.path.join(cd,'data',channel_met.replace('00','sel')))
    
files=glob.glob(os.path.join(cd,'data',channel_met,'*csv'))
tnum_met_all=[]
temp_all=[]
press_all=[]
rh_all=[]
for f in files:
    Data=pd.read_csv(f)
    Data.iloc[Data.iloc[:,5].values>=60,5]=59.99
    for i in range(len(Data.index)):
        tstr_met='{i:04d}'.format(i=Data.iloc[i,0])+'-'+\
             '{i:02d}'.format(i=Data.iloc[i,1])+'-'+\
             '{i:02d}'.format(i=Data.iloc[i,2])+' '+\
             '{i:02d}'.format(i=Data.iloc[i,3])+':'+\
             '{i:02d}'.format(i=Data.iloc[i,4])+':'+\
             '{i:05.2f}'.format(i=Data.iloc[i,5])
             
        tnum_met_all=np.append(tnum_met_all,utl.datenum(tstr_met,'%Y-%m-%d %H:%M:%S.%f'))
    
    temp_all=np.append(temp_all,Data.iloc[:,met_headers['temperature']].values)  
    press_all=np.append(press_all,Data.iloc[:,met_headers['pressure']].values) 
    rh_all=np.append(rh_all,Data.iloc[:,met_headers['humidity']].values) 

basetime_met=utl.floor(tnum_met_all[0],24*3600)
time_offset_met=tnum_met_all-basetime_met

Output_met=xr.Dataset()
Output_met['temp_mean']=     xr.DataArray(data=np.nan_to_num(temp_all,nan=-9999),
                                     coords={'time':np.arange(len(time_offset_met))})
Output_met['atmos_pressure']=xr.DataArray(data=np.nan_to_num(press_all/10,nan=-9999),
                                     coords={'time':np.arange(len(time_offset_met))})
Output_met['rh_mean']=       xr.DataArray(data=np.nan_to_num(rh_all,nan=-9999),
                                     coords={'time':np.arange(len(time_offset_met))})

Output_met['time_offset']=xr.DataArray(data=time_offset_met,
                                     coords={'time':np.arange(len(time_offset_met))})
Output_met['base_time']=np.float64(basetime_met)
Output_met.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
Output_met.to_netcdf(os.path.join(cd,'data',channel_met.replace('00','sel'),channel_met.replace('00','sel').replace('wfip3/','')+'.'+utl.datestr(basetime_met,'%Y%m%d.%H%M%S')+'.nc'))

#%% Plots
plt.figure(figsize=(18,8))

#plot radiance at 675 cm^-1
Data_ch1=xr.open_dataset(glob.glob(os.path.join(cd,'data',channel_assist,'ch1','*cdf'))[-1]).sortby('time')
tnum_ch1=Data_ch1.time.values+Data_ch1.base_time.values/10**3
time_ch1=np.array([datetime.utcfromtimestamp(np.float64(t)) for t in tnum_ch1])
sky_ch1=Data_ch1['sceneMirrorAngle'].values==0

Data_ch1_nsf=xr.open_dataset(glob.glob(os.path.join(cd,'data',channel_assist,'nfc','*cdf'))[-1]).sortby('time')
tnum_ch1_nsf=Data_ch1_nsf.time.values+Data_ch1_nsf.base_time.values/10**3
time_ch1_nsf=np.array([datetime.utcfromtimestamp(np.float64(t)) for t in tnum_ch1_nsf])
sky_ch1_nsf=Data_ch1_nsf['sceneMirrorAngle'].values==0

plt.subplot(5,1,1)
plt.plot(time_ch1[sky_ch1],Data_ch1['mean_rad'].interp(wnum=675).values[sky_ch1],'r',label='Raw')
plt.plot(time_ch1_nsf[sky_ch1_nsf],Data_ch1_nsf['mean_rad'].interp(wnum=675).values[sky_ch1_nsf],'g',label='Filtered')
plt.ylabel('Mean radiance\n'+r'at 675 cm$^{-1}$ [r.u.]')
plt.grid()
plt.legend()
date_fmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(date_fmt)
plt.title('Input data on '+date)

#plot cbh
tnum_cbh=Output_cbh.time.values+Output_cbh.base_time.values
time_cbh=np.array([datetime.utcfromtimestamp(t) for t in tnum_cbh])
real=Output_cbh.first_cbh>0
plt.subplot(5,1,2)
plt.plot(time_cbh[real],Output_cbh.first_cbh.values[real],'.b')
plt.ylabel('CBH [m]')
plt.grid()
plt.gca().xaxis.set_major_formatter(date_fmt)

#plot temperature
tnum_met=Output_met.time_offset.values+Output_met.base_time.values
time_met=np.array([datetime.utcfromtimestamp(t) for t in tnum_met])
real=Output_met.temp_mean>0
plt.subplot(5,1,3)
plt.plot(time_met[real],Output_met.temp_mean.values[real],'.r')
plt.ylabel(r'$T$ [C]')
plt.grid()
plt.gca().xaxis.set_major_formatter(date_fmt)

#plot pressure
tnum_met=Output_met.time_offset.values+Output_met.base_time.values
time_met=np.array([datetime.utcfromtimestamp(t) for t in tnum_met])
real=Output_met.atmos_pressure>0
plt.subplot(5,1,4)
plt.plot(time_met[real],Output_met.atmos_pressure.values[real],'.k')
plt.ylabel(r'$P$ [kPa]')
plt.grid()
plt.gca().xaxis.set_major_formatter(date_fmt)

#plot humidity
tnum_met=Output_met.time_offset.values+Output_met.base_time.values
time_met=np.array([datetime.utcfromtimestamp(t) for t in tnum_met])
real=Output_met.rh_mean>0
plt.subplot(5,1,5)
plt.plot(time_met[real],Output_met.rh_mean.values[real],'.c')
plt.ylabel('RH [%]')
plt.grid()
plt.gca().xaxis.set_major_formatter(date_fmt)
plt.tight_layout()
plt.xlabel('Time (UTC)')

utl.mkdir(os.path.join(cd,'figures/rhod'))
plt.savefig(os.path.join(cd,'figures/rhod',utl.datestr(utl.datenum(date,'%Y-%m-%d'),'%Y%m%d.%H%M%S')+'.rhod.tropoeinput.png'))