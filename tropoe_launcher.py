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
import xarray as xr
import subprocess
import yaml
import glob as glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates

plt.close('all')

#%% Inputs
source_config=os.path.join(cd,'configs/config.yaml')

if len(sys.argv)==1:
    site='rhod'
    sdate='20240520'
    edate='20240520'
else:
    site=sys.argv[1]
    sdate=sys.argv[2]
    edate=sys.argv[3]
    
#%% Initialization

#inputs
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
channel_irs=config['channel_irs'][site]
channel_cbh=config['channel_cbh'][site]
channel_met=config['channel_met'][site]
site_prior=config['site_prior']
verbosity=config['verbosity']

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

#create directories
utl.mkdir(os.path.join(cd,'data',channel_irs).replace('00','c0').replace('assist','assist.tropoe'))
utl.mkdir(os.path.join('log',site))

# Loop to generate the range of datetimes
days=[]
current_date = datetime.strptime(sdate,'%Y%m%d')
while current_date <= datetime.strptime(edate,'%Y%m%d'):
    days.append(current_date)
    current_date += timedelta(days=1)
    
#%% Main
for d in days:
    date=datetime.strftime(d,'%Y%m%d')
    month=datetime.strftime(d,'%m')
    if sum([date == p for p in processed])==0:

        logger,handler=utl.create_logger(os.path.join('log',site,date+'.log'))

        logger.info('Running TROPoe at '+site+' on '+date)
        print('Running TROPoe at '+site+' on '+date)

        #create input files
        command=config['path_python']+f' tropoe_inputs.py {site} {date} '
        result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
        logger.info(result.stdout)
        logger.error(result.stderr)
        
        if len(glob.glob(os.path.join(cd,'data',channel_cbh.replace('a0','cbh'),'*'+date+'*')))==0:
            logger.error('No CBH inputs found. Skipping.')
            continue
        
        if len(glob.glob(os.path.join(cd,'data',channel_met.replace('00','sel'),'*'+date+'*')))==0:
            logger.error('No met inputs found. Skipping.')
            continue
        
        if len(glob.glob(os.path.join(cd,'data',channel_irs,'nfc','*'+date+'*cdf')))==1 and len(glob.glob(os.path.join(cd,'data',channel_irs,'sum','*'+date+'*cdf')))==1:
            f_ch1=glob.glob(os.path.join(cd,'data',channel_irs,'nfc','*'+date+'*cdf'))[0]
            f_sum=glob.glob(os.path.join(cd,'data',channel_irs,'sum','*'+date+'*cdf'))[0]
            
            #time check
            Data_ch1=xr.open_dataset(f_ch1)
            time_ch1=np.sort(Data_ch1['time'].values+Data_ch1['base_time'].values/10**3)
            del(Data_ch1)
            
            Data_sum=xr.open_dataset(f_sum)
            time_sum=np.sort((Data_sum['time'].values+Data_sum['base_time'].values)/np.timedelta64(1,'s'))
            
            if np.abs(np.nanmax(time_ch1)-np.nanmax(time_sum))>config['max_time_diff'] or np.abs(np.nanmin(time_ch1)-np.nanmin(time_sum))>config['max_time_diff']:
                logger.error('Inconsistent time on '+date+'. Skipping.')
                utl.close_logger(logger, handler)
                continue
        else:
            logger.error('Missing or multiple files found on '+date+'. Skipping.')
            utl.close_logger(logger, handler)
            continue
                    
        #run TROPoe
        command='chmod -R 777 '+cd
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        
        command =f'./run_tropoe_ops.sh {date} configs/vip_{site}.txt prior/Xa_Sa_datafile.{site_prior}.55_levels.month_{month}.cdf 0 24'+\
        f' {verbosity} ' +cd + ' ' + cd + ' davidturner53/tropoe:latest'
        logger.info('The following will be executed: \n'+command+'\n')
        result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
        logger.info(result.stdout)
        logger.error(result.stderr)
        
        #% plots
        if len(glob.glob(os.path.join(cd,'data',channel_irs.replace('00','c0').replace('assist','assist.tropoe'),'*'+date+'*nc')))==1:
            
            #add to processed list
            with open(os.path.join(cd,'data/processed-{site}.txt'.format(site=site)), 'a') as fid:
                fid.write(date+'\n')
                
            file_tropoe=glob.glob(os.path.join(cd,'data',channel_irs.replace('00','c0').replace('assist','assist.tropoe'),'*'+date+'*nc'))[0]
            
            logger.info('Succesfully created retrieval '+file_tropoe)
        
            Data=xr.open_dataset(file_tropoe)
    
            time=np.array(Data['time'])
            height0=np.array(Data['height'][:])*1000
            sel_z=height0<config['max_z']
            height=height0[sel_z]
    
            T=np.array(Data['temperature'].where(Data['gamma']<=config['max_gamma']).where(Data['rmsr']<=config['max_rmsr'])[:,sel_z])#[C]
            r= np.array(Data['waterVapor'].where(Data['gamma']<=config['max_gamma']).where(Data['rmsr']<=config['max_rmsr'])[:,sel_z])#[g/Kg]
            cbh=np.array(Data['cbh'].where(Data['gamma']<=config['max_gamma']).where(Data['rmsr']<=config['max_rmsr'])[:])*1000#[m]
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
            ax.set_title('TROPoe retrieval at ' + Data.attrs['Site'] + ' on '+utl.datestr(utl.dt64_to_num(time[0]),'%Y%m%d')+'\n File: '+os.path.basename(file_tropoe), x=0.45)
            ax.set_facecolor((0.9,0.9,0.9))
            
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
            ax.set_facecolor((0.9,0.9,0.9))
            
            plt.savefig(file_tropoe.replace('.nc','_T_r.png'))
            utl.close_logger(logger, handler)
    
        else:
            logger.info('Skipping '+f_ch1)
            utl.close_logger(logger, handler)