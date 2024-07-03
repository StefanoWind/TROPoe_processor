'''
Run TROPoe retrieval and store results
'''

import os
cd=os.getcwd()
import sys
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
import re
import xarray as xr
import logging
import subprocess
import yaml
import glob as glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates

plt.close('all')

#%% Inputs
source_config=os.path.join(cd,'configs/config_local.yaml')

if len(sys.argv)==1:
    site='rhod'
    sdate='20240520'
    edate='20240521'
else:
    site=sys.argv[1]
    date=sys.argv[2]

#%% Initialization

#inputs
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
channel_irs=config['channel_irs'][site]
source_ch1=os.path.join(cd,'data',channel_irs,'ch1')
source_sum=os.path.join(cd,'data',channel_irs,'sum')

#imports
sys.path.append(config['path_utils']) 
import utils as utl

#processed list
if os.path.exists(os.path.join(cd,'data/processed-{site}.txt'.format(site=site))):
    with open(os.path.join(cd,'data/processed-{site}.txt'.format(site=site))) as fid:
        processed=fid.readlines()
    processed=[p.strip() for p in processed]
else: 
    processed=[]

#change permissions
command='chmod -R 777 '+cd
print(command)
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

#create directories
utl.mkdir(os.path.join(cd,'data',channel_irs)[0].replace('00','c0'))
utl.mkdir(os.path.join('log',site))

#get ch1 filenames
files_ch1=np.array(glob.glob(os.path.join(source_ch1.format(site=site),'assistcha*cdf')))
t_file_ch1=[]
for f in files_ch1:
    match = re.search(r'\d{8}\.\d{6}', f)
    t=datetime.strptime(match.group(0),'%Y%m%d.%H%M%S')
    t_file_ch1=np.append(t_file_ch1,t)

sel_t=(t_file_ch1>datetime.strptime(sdate,'%Y%m%d'))*(t_file_ch1<datetime.strptime(edate,'%Y%m%d'))
files_ch1_sel=[os.path.basename(f) for f in files_ch1[sel_t]]
t_file_ch1_sel=t_file_ch1[sel_t]

#get summary filenames
files_sum=np.array(glob.glob(os.path.join(source_sum.format(site=site),'assistsummary*cdf')))
t_file_sum=[]
for f in files_sum:
    match = re.search(r'\d{8}\.\d{6}', f)
    t=datetime.strptime(match.group(0),'%Y%m%d.%H%M%S')
    t_file_sum=np.append(t_file_sum,t)

sel_t=(t_file_sum>datetime.strptime(sdate,'%Y%m%d'))*(t_file_sum<datetime.strptime(edate,'%Y%m%d'))
files_sum_sel=[os.path.basename(f) for f in files_sum[sel_t]]
t_file_sum_sel=t_file_sum[sel_t]

#find matching files
t_match=utl.match_arrays(t_file_ch1_sel,t_file_sum_sel,timedelta(seconds=config['max_time_diff']))

#%% Main
for i_ch1,i_sum in zip(t_match[:,0],t_match[:,1]):

    f_ch1=files_ch1_sel[i_ch1]
    f_sum=files_sum_sel[i_sum]
    
    if sum([f_ch1 == f for f in processed])==0:

        #load ch1 file time information
        Data=xr.open_dataset(os.path.join(source_ch1,f_ch1))
        time=np.sort(np.float64(Data['time'].values)+Data['base_time'].values/10**3)
        date = utl.datestr(time[0],'%Y%m%d')
        month= utl.datestr(time[0],'%m')

        #logger
        if os.path.exists(os.path.join('log',site,date+'_tropoe.log')):
            open(os.path.join('log',site,date+'_tropoe.log'), 'w').close()
        logging.basicConfig(
            filename=os.path.join('log',site,date+'_tropoe.log'),       # Log file name
            level=logging.INFO,      # Set the logging level
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
            datefmt='%Y-%m-%d %H:%M:%S'  # Date format
        )

        logger = logging.getLogger()
        logger.info('Running TROPoe at '+site+' on '+date)

        #create input files
        command=config['path_python']+f' tropoe_inputs.py {site} {date} '
        result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
        logger.info(result.stdout)
        logger.error(result.stderr)
        
        raise BaseException()
        
        #run TROPoe
        command ='./run_tropoe_ops.sh {date} configs/vip_{site}.txt prior/Xa_Sa_datafile.{site_prior}.55_levels.month_{month}.cdf 0 24'.format(date=date,site_prior=config['site_prior'],month=month,site=site)+ \
        ' 3 '+cd + ' '+ cd + ' davidturner53/tropoe:latest'
        logger.info('The following will be executed": \n'+command+'\n')
        result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
        logger.info(result.stdout)
        logger.error(result.stderr)
        
        #% plots
        if len(glob.glob(os.path.join(cd,'data',channel_irs[0].replace('00','c0'),'*'+date+'*cdf')))==1:
            
            #add to processed list
            with open(os.path.join(cd,'data/processed-{site}.txt'.format(site=site)), 'a') as fid:
                fid.write(f_ch1+'\n')
                
            file_tropoe=glob.glob(os.path.join(cd,'data',channel_irs[0].replace('00','c0'),'*'+date+'*cdf'))[0]
        
            Data=xr.open_dataset(file_tropoe)
    
            time=np.array(Data['time'])
            height0=np.array(Data['height'][:])*1000
            sel_z=height0<config['max_z']
            height=height0[sel_z]
    
            T=np.array(Data['temperature'][:,sel_z])#[C]
            r=np.array(Data['waterVapor'][:,sel_z])#[g/Kg]
            cbh=np.array(Data['cbh'][:])*1000#[m]
            lwp=np.array(Data['lwp'][:])
            cbh_sel=cbh.copy()
            cbh_sel[lwp<config['min_lwp']]=np.nan
    
            plt.close('all')
            fig=plt.figure(figsize=(18,10))
            ax=plt.subplot(2,1,1)
            CS=plt.contourf(time,height,T.T,np.round(np.arange(np.nanpercentile(T, 5),np.nanpercentile(T, 95),1)),cmap='hot',extend='both')
            plt.ylim([0,config['max_z']])
            plt.plot(time,cbh_sel,'.m',label='Cloud base height',markersize=10)
            if np.sum(~np.isnan(cbh_sel))>0:
                plt.legend()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size = '2%', pad=0.65)
            cb = fig.colorbar(CS, cax=cax, orientation='vertical')
           
            cb.set_label(r'Temperature [$^\circ$C]')
            ax.set_xlabel('Time (UTC)')
            ax.set_ylabel(r'$z$ [m.a.g.l.]')
            ax.set_xlim(time[0]-np.timedelta64(120,'s'),time[-1]+np.timedelta64(120,'s'))
            ax.set_ylim(0, np.max(height)+10)
            ax.grid()
            ax.tick_params(axis='both', which='major')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_title('TROPoe retrieval at ' + Data.attrs['Site'] + ' on '+utl.datestr(utl.dt64_to_num(time[0]),'%Y-%m-%d')+'\n File: '+os.path.basename(file_tropoe), x=0.45)
    
            ax=plt.subplot(2,1,2)
            CS=plt.contourf(time,height,r.T,np.round(np.arange(np.nanpercentile(r, 5),np.nanpercentile(r, 95),0.25),2),cmap='GnBu',extend='both')
            plt.ylim([0,config['max_z']])
            plt.plot(time,cbh_sel,'.m',label='Cloud base height',markersize=10)
            if np.sum(~np.isnan(cbh_sel))>0:
                plt.legend()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size = '2%', pad=0.65)
            cb = fig.colorbar(CS, cax=cax, orientation='vertical')
            cb.set_label(r'Mixing ratio [g Kg$^{-1}$]')
            ax.set_xlabel('Time (UTC)')
            ax.set_ylabel(r'$z$ [m.a.g.l.]')
            ax.set_xlim(time[0]-np.timedelta64(120,'s'),time[-1]+np.timedelta64(120,'s'))
            ax.set_ylim(0, np.max(height)+10)
            ax.grid()
            ax.tick_params(axis='both', which='major')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.savefig(file_tropoe.replace('.nc','_T_r.png'))
    
        else:
            logger.info('Skipping '+f_ch1)