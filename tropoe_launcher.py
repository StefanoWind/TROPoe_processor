'''
Run TROPoe retrieval and store results
'''

import os
cd=os.path.dirname(__file__)

import sys
sys.path.append('/home/utletizia/codes')
    
import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import datetime as dt
import re
import xarray as xr
# import shutil
import subprocess
import glob as glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates

plt.close('all')

#%% Inputs
source_ch1=os.path.join(cd,'data/wfip3/{site}.assist.z01.00/nfc')
source_sum=os.path.join(cd,'data/wfip3/{site}.assist.z01.00/sum')
path_utils='C:/Users/SlETIZIA/OneDrive - NREL/Desktop/PostDoc/utils'

time_range=['2024-05-20 00:00','2024-05-21 00:00']
site='rhod'
site_prior='upton_NY'
time_pattern = r'\d{8}\.\d{6}'
max_time_diff=dt.timedelta(seconds=60)#maximum time difference between chA and summary file

#graphics
max_z=1800#[m]
min_dthetav_dz=10**-3#[K/m]
min_lwp=10#[g/m^2] minimum liquid water path
colorbar_fs = 14#colorbar fontsize
label_fs = 14#labels fontsize
tick_fs = 14#tick labels fontsize

#%% Initialization

sys.path.append(path_utils) 
import utils as utl

if site=='':
    site=int(sys.argv[1])
print('Site '+str(site)+' selected')

if os.path.exists(os.path.join(cd,'data/processed-assist-{site}.txt'.format(site=site))):
    with open(os.path.join(cd,'data/processed-assist-{site}.txt'.format(site=site))) as fid:
        processed=fid.readlines()
    processed=[p.strip() for p in processed]
else: 
    processed=[]

files_ch1=np.array(glob.glob(os.path.join(source_ch1.format(site=site),'assistcha*cdf')))
t_file_ch1=[]
for f in files_ch1:
    match = re.search(time_pattern, f)
    t=datetime.strptime(match.group(0),'%Y%m%d.%H%M%S')
    t_file_ch1=np.append(t_file_ch1,t)

sel_t=(t_file_ch1>datetime.strptime(time_range[0],'%Y-%m-%d %H:%M'))*(t_file_ch1<datetime.strptime(time_range[1],'%Y-%m-%d %H:%M'))
files_ch1_sel=[os.path.basename(f) for f in files_ch1[sel_t]]
t_file_ch1_sel=t_file_ch1[sel_t]

files_sum=np.array(glob.glob(os.path.join(source_sum.format(site=site),'assistsummary*cdf')))
t_file_sum=[]
for f in files_sum:
    match = re.search(time_pattern, f)
    t=datetime.strptime(match.group(0),'%Y%m%d.%H%M%S')
    t_file_sum=np.append(t_file_sum,t)

sel_t=(t_file_sum>datetime.strptime(time_range[0],'%Y-%m-%d %H:%M'))*(t_file_sum<datetime.strptime(time_range[1],'%Y-%m-%d %H:%M'))
files_sum_sel=[os.path.basename(f) for f in files_sum[sel_t]]
t_file_sum_sel=t_file_sum[sel_t]

#find matching files
t_match=utl.match_arrays(t_file_ch1_sel,t_file_sum_sel,max_time_diff)

#change permission
command='chmod 777 data/tropoe'
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

# utl.mkdir('data/ch1/{side}'.format(site=site))
# utl.mkdir('data/sum/{side}'.format(site=site))

#%% Main
for i_ch1,i_sum in zip(t_match[:,0],t_match[:,1]):
    
    # #clean old temp files
    # for fr in glob.glob('data/ch1/{side}/*cdf'.format(site=site)):
    #     os.remove(fr)
    # for fr in glob.glob('data/sum/{side}/*cdf'.format(site=site)):
    #     os.remove(fr)

    f_ch1=files_ch1_sel[i_ch1]
    f_sum=files_sum_sel[i_sum]
    
    if sum([f_ch1 == f for f in processed])==0:

        print(f_ch1+' and '+f_sum+' will be processed')
        
        #chA file import
        # shutil.copy(os.path.join(source_ch1.format(site=site),f_ch1),os.path.join('data/ch1/{side}'.format(site=site),f_ch1))
        Data=xr.open_dataset(os.path.join(source_ch1.format(site=site),f_ch1))
        time=np.sort(np.float64(Data['time'].values)+Data['base_time'].values/10**3)

        # #summary file import
        # shutil.copy(os.path.join(source_ch1.format(site=site),f_sum),os.path.join('data/sum/{side}'.format(site=site),f_sum))

        #TROPoe setup
        date = utl.datestr(time[0],'%Y%m%d')
        month= utl.datestr(time[0],'%m')

        #set permissions
        command='chmod 755 '+os.path.join(source_ch1.format(site=site),f_ch1)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

        command='chmod 755 '+os.path.join(source_sum.format(site=site),f_sum)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

        #run TROPoe
        command ='./run_tropoe_ops.sh {date} tropoe/vip_{site}_wfip3.txt prior/Xa_Sa_datafile.{site_prior}.55_levels.month_{month}.cdf 0 24'.format(date=date,site_prior=site_prior,month=month,site=site)+ \
        ' 1 '+os.path.join(cd,'data') + os.path.join(cd,'data')+ 'davidturner53/tropoe:latest'
        print('The following will be executed \n'+command+'\n')

        result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
        print(result.stdout)

        #save output to log file
        with open(os.path.join(cd,'data/logs/'+os.path.basename(f_ch1[:-4])+'_'+str(site)+'_log.txt'), 'w') as fid:
            fid.write(result.stdout)

        #add to processed list
        with open(os.path.join(cd,'data/processed-assist-{site}.txt'.format(site=site)), 'a') as fid:
            fid.write(f_ch1+'\n')

        #plot and archive results
        file_tropoe=glob.glob(os.path.join('data/tropoe/{site}'.format(site=site),'{site}.assist.z01.c0.'))
        
        Data=xr.open_dataset(file_tropoe)

        time=np.array(Data['time'])
        height0=np.array(Data['height'][:])*1000
        sel_z=height0<max_z
        height=height0[sel_z]

        T=np.array(Data['temperature'][:,sel_z])#[C]
        r=np.array(Data['waterVapor'][:,sel_z])#[g/Kg]
        cbh=np.array(Data['cbh'][:])*1000#[m]
        lwp=np.array(Data['lwp'][:])
        cbh_sel=cbh.copy()
        cbh_sel[lwp<min_lwp]=np.nan

        #%% Plots
        plt.close('all')
        fig=plt.figure(figsize=(18,10))
        ax=plt.subplot(2,1,1)
        CS=plt.contourf(time,height,T.T,np.round(np.arange(np.nanpercentile(T, 5),np.nanpercentile(T, 95),1)),cmap='hot',extend='both')
        plt.ylim([0,max_z])
        plt.plot(time,cbh_sel,'.m',label='Cloud base height',markersize=10)
        if np.sum(~np.isnan(cbh_sel))>0:
            plt.legend()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '2%', pad=0.65)
        cb = fig.colorbar(CS, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=colorbar_fs)
        cb.set_label(r'Temperature [$^\circ$C]', fontsize=colorbar_fs)
        ax.set_xlabel('Time (UTC)', fontsize=label_fs)
        ax.set_ylabel(r'z [m $AGL$]', fontsize=label_fs)
        ax.set_xlim(time[0]-np.timedelta64(120,'s'),time[-1]+np.timedelta64(120,'s'))
        ax.set_ylim(0, np.max(height)+10)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%M'))
        ax.set_title('TROPoe retrieval at ' + Data.attrs['Site'] + ' on '+utl.datestr(utl.dt64_to_num(time[0]),'%Y%m%d')+'\n File: '+os.path.basename(file_tropoe), fontsize=label_fs, x=0.45)

        ax=plt.subplot(2,1,2)
        CS=plt.contourf(time,height,r.T,np.round(np.arange(np.nanpercentile(r, 5),np.nanpercentile(r, 95),0.25),2),cmap='GnBu',extend='both')
        plt.ylim([0,max_z])
        plt.plot(time,cbh_sel,'.m',label='Cloud base height',markersize=10)
        if np.sum(~np.isnan(cbh_sel))>0:
            plt.legend()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '2%', pad=0.65)
        cb = fig.colorbar(CS, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=colorbar_fs)
        cb.set_label(r'Mixing ratio [g Kg$^{-1}$]', fontsize=colorbar_fs)
        ax.set_xlabel('Time (UTC)', fontsize=label_fs)
        ax.set_ylabel(r'Z [m $AGL$]', fontsize=label_fs)
        ax.set_xlim(time[0]-np.timedelta64(120,'s'),time[-1]+np.timedelta64(120,'s'))
        ax.set_ylim(0, np.max(height)+10)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%M'))

        # shutil.copy(ft,os.path.join(folder_processed.format(site=site),os.path.basename(ft)))
        # os.remove(ft)
        # print(ft+' saved in '+folder_processed.format(site=site))

        plt.savefig(os.path.join('data/tropoe/nreltropoe_{site}'.format(site=site),file_tropoe).replace('.nc','_T_r.png'))

    else:
        print('Skipping '+f_ch1)