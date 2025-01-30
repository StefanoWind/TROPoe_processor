# -*- coding: utf-8 -*-
"""
General data preconditioning for TROPoe
"""
import os
cd=os.getcwd()
import sys
sys.path.append(os.path.join(cd,'utils')) 
import tropoe_utils as trp
from datetime import datetime
from datetime import timedelta
import logging
import subprocess
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import shutil
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import glob
import yaml
import matplotlib
import warnings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs

if len(sys.argv)==1:
    site='nwtc.z02'
    date='20220515'
    source_config=os.path.join(cd,'configs/config_basic.yaml')
    
else:
    site=sys.argv[1]
    date=sys.argv[2]
    source_config=sys.argv[3]
    tmpdir=sys.argv[4]

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#imports
sys.path.append(config['path_utils']) 
import utils as utl



logger,handler=utl.create_logger(os.path.join('log',site,date+'.log'))
logger = logging.getLogger()
logger.info('Building TROPoe inputs for '+date+' at '+site)

#%% Main
channel_irs=config['channel_irs'][site]

#define temporary directory if not provided
if len(sys.argv)==1:
    tmpdir=os.path.join(cd,'data',channel_irs,date+'-tmp')

#clear old temp files
if os.path.exists(tmpdir):
    shutil.rmtree(tmpdir)
    
os.makedirs(tmpdir,exist_ok=True)

chassistdir=os.path.join(tmpdir,'ch1')
sumassistdir=os.path.join(tmpdir,'sum')
nfchassistdir=os.path.join(tmpdir,'nfc')

#create temp vip file
with open(os.path.join(cd,'configs',f'vip_{site}.txt'), "r") as f:
    vip = f.read()
vip=vip.replace('{date}',date)
with open(os.path.join(tmpdir,f'vip_{site}.{date}.txt'), "w") as f:
    f.write(vip)

#download assist data
time_range = [datetime.strftime(datetime.strptime(date, '%Y%m%d')-timedelta(days=config['N_days_nfc']-1),'%Y%m%d%H%M%S'),
              datetime.strftime(datetime.strptime(date, '%Y%m%d')+timedelta(days=1),'%Y%m%d%H%M%S')]
if channel_irs !="":
    n_files_irs=trp.download(channel_irs,time_range,config)
    logger.info(str(n_files_irs)+' ASSIST files downloaded')

#qc settings
logger.info('Running PCA filter')
sdate=datetime.strftime(datetime.strptime(date,'%Y%m%d')-timedelta(days=config['N_days_nfc']-1),'%Y%m%d')
edate=date

#pre-conditioning
trp.copy_rename_assist(channel_irs,sdate,edate,chassistdir,sumassistdir)
if config['apply_prefilter']:
    trp.pre_filter(files=glob.glob(chassistdir+'/*cdf'),logger=logger)
    
#pca filter
command=config['path_python']+f' utils/run_irs_nf.py --create {sdate} {edate} {chassistdir} {sumassistdir} {nfchassistdir} "assist"'
result = subprocess.run(command, shell=True, text=True,capture_output=True)
logger.info(result.stdout)
logger.error(result.stderr)

command=config['path_python']+f' utils/run_irs_nf.py --apply {sdate} {edate} {chassistdir} {sumassistdir} {nfchassistdir} "assist"'
result = subprocess.run(command, shell=True, text=True,capture_output=True)
logger.info(result.stdout)
logger.error(result.stderr)

#remove atmospheric pressure that causes error
if len(glob.glob(os.path.join(nfchassistdir,'*'+date+'*cdf')))==1:
    f_ch1=glob.glob(os.path.join(nfchassistdir,'*'+date+'*cdf'))[0]
    Data_ch1 = Dataset(f_ch1,'a')
    Data_ch1.renameVariable('atmosphericPressure','deprecated_atmosphericPressure')
    Data_ch1.close()

#get cbh data
channel_cbh=config['channel_cbh'][site]
if channel_cbh !="":
    if len(glob.glob(os.path.join(cd,'data',channel_cbh.replace(channel_cbh[-2:],'cbh'),'*'+date+'*nc')))==0:
        time_range = [datetime.strftime(datetime.strptime(date, '%Y%m%d'),'%Y%m%d%H%M%S'),
                      datetime.strftime(datetime.strptime(date, '%Y%m%d')+timedelta(days=0.9999),'%Y%m%d%H%M%S')]
        n_files_cbh=trp.download(channel_cbh,time_range,config)
        logger.info(str(n_files_cbh)+' CBH files downloaded')
        
        if n_files_cbh==0:
            logger.error('No CBH data found. Aborting.')
            raise BaseException()
            
        if 'lidar' in channel_cbh:
                logger.info('Running CBH retrieval from lidar data')
                Output_cbh=trp.compute_cbh_halo(channel_cbh,date,config,logger)
            
        elif 'ceil' in channel_cbh:
            logger.info('Extracting CBH from ceilometer data')
            trp.extract_cbh_ceil(channel_cbh,date,config,logger)
            
    else:
        logger.info('CBH data already available, skipping.')
        
else:
    logger.info('No CBH channel provided.')
        
#get met data
channel_met=config['channel_met'][site]
if channel_met !="":
    if len(glob.glob(os.path.join(cd,'data',channel_met.replace(channel_met[-2:],'sel'),'*'+date+'*nc')))==0:
        time_range = [datetime.strftime(datetime.strptime(date, '%Y%m%d'),'%Y%m%d%H%M%S'),
                      datetime.strftime(datetime.strptime(date, '%Y%m%d')+timedelta(days=0.9999),'%Y%m%d%H%M%S')]
        n_files_met=trp.download(channel_met,time_range,config)
        logger.info(str(n_files_met)+' met files downloaded')
        
        if n_files_met==0:
            logger.error('No met data found. Aborting.')
            raise BaseException()
            
        logger.info('Extracting met data')
        Data_met=trp.exctract_met(channel_met,date,site,config,logger)
        
    else:
        logger.info('Met data already available, skipping.')
else:
    logger.info('No met channel provided.')    
    
#%% Plots
if os.path.exists(glob.glob(os.path.join(cd,'data',channel_irs,'*'+date+'*cdf'))[0].replace('00','c0')+'.rhod.tropoeinput.png')==0:
    plt.figure(figsize=(18,10))
    date_fmt = mdates.DateFormatter('%H:%M')
    
    #plot radiance at 675 cm^-1
    if len(glob.glob(os.path.join(nfchassistdir,'*'+date+'*cdf')))==1:
        file_ch1=glob.glob(os.path.join(nfchassistdir,'*'+date+'*cha*cdf'))[0]
        Data_ch1=xr.open_dataset(file_ch1).sortby('time')
        tnum_ch1=Data_ch1.time.values+Data_ch1.base_time.values/10**3
        time_ch1=np.array([datetime.utcfromtimestamp(np.float64(t)) for t in tnum_ch1])
        sky_ch1=Data_ch1['sceneMirrorAngle'].values==0
    
        file_ch1_ncf=glob.glob(os.path.join(cd,'data',channel_irs,'nfc','*'+date+'*cdf'))[0]
        Data_ch1_nfc=xr.open_dataset(file_ch1_ncf).sortby('time')
        Data_ch1_nfc['mean_rad']=xr.where(Data_ch1_nfc['missingDataFlag']==0,Data_ch1_nfc['mean_rad'],np.nan)#mask mean_rad with missingDataFlag

        tnum_ch1_nfc=Data_ch1_nfc.time.values+Data_ch1_nfc.base_time.values/10**3
        time_ch1_nfc=np.array([datetime.utcfromtimestamp(np.float64(t)) for t in tnum_ch1_nfc])
        sky_ch1_nfc=Data_ch1_nfc['sceneMirrorAngle'].values==0
        
        plt.subplot(5,1,1)
        plt.plot(time_ch1[sky_ch1],Data_ch1['mean_rad'].interp(wnum=675).values[sky_ch1],'r',label='Raw')
        plt.plot(time_ch1_nfc[sky_ch1_nfc],Data_ch1_nfc['mean_rad'].interp(wnum=675).values[sky_ch1_nfc],'g',label='Filtered')
        plt.ylabel('Mean radiance\n'+r'at 675 cm$^{-1}$ [r.u.]')
        plt.xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
        plt.grid()
        plt.legend()
        
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.title('TROPoe input data on '+date+' at '+site)

    #plot cbh
    if len(glob.glob(os.path.join(cd,'data',channel_cbh.replace(channel_cbh[-2:],'cbh'),'*'+date+'*nc')))==1:
        file_cbh=glob.glob(os.path.join(cd,'data',channel_cbh.replace(channel_cbh[-2:],'cbh'),'*'+date+'*nc'))[0]
        Data_cbh=xr.open_dataset(file_cbh)
        tnum_cbh=Data_cbh.time.values+Data_cbh.base_time.values
        time_cbh=np.array([datetime.utcfromtimestamp(t) for t in tnum_cbh])
        real=Data_cbh.first_cbh>0
        plt.subplot(5,1,2)
        plt.plot(time_cbh[real],Data_cbh.first_cbh.values[real],'.b')
        plt.ylabel('CBH [m]')
        plt.xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
        plt.grid()
        plt.gca().xaxis.set_major_formatter(date_fmt)
    
    if len(glob.glob(os.path.join(cd,'data',channel_met.replace(channel_met[-2:],'sel'),'*'+date+'*nc')))==1:
        file_met=glob.glob(os.path.join(cd,'data',channel_met.replace(channel_met[-2:],'sel'),'*'+date+'*nc'))[0]
        Data_met=xr.open_dataset(file_met)
        
        #plot temperature
        tnum_met=Data_met.time_offset.values+Data_met.base_time.values
        time_met=np.array([datetime.utcfromtimestamp(t) for t in tnum_met])
        real=Data_met.temp_mean>0
        plt.subplot(5,1,3)
        plt.plot(time_met[real],Data_met.temp_mean.values[real],'.r')
        plt.ylabel(r'$T$ [$^\circ$C]')
        plt.xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
        plt.grid()
        plt.gca().xaxis.set_major_formatter(date_fmt)
        
        #plot pressure
        real=Data_met.atmos_pressure>0
        plt.subplot(5,1,4)
        plt.plot(time_met[real],Data_met.atmos_pressure.values[real],'.k')
        plt.ylabel(r'$P$ [kPa]')
        plt.xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
        plt.grid()
        plt.gca().xaxis.set_major_formatter(date_fmt)
        
        #plot humidity
        real=Data_met.rh_mean>0
        plt.subplot(5,1,5)
        plt.plot(time_met[real],Data_met.rh_mean.values[real],'.c')
        plt.ylabel('RH [%]')
        plt.xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
        plt.grid()
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.tight_layout()
        plt.xlabel('Time (UTC)')
    
    dir_save=os.path.join(cd,'data',channel_irs).replace('00','c0').replace('assist','assist.tropoe')
    name_save='.'.join(os.path.basename(file_ch1).replace('.00.','.c0.').replace('.assist.','.assist.tropoe.').split('.')[:-2])+'_tropoe_inputs.png'
    utl.mkdir(dir_save)
    plt.savefig(os.path.join(dir_save,name_save))