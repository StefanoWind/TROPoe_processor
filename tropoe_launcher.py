'''
Run TROPoe retrieval and store results
'''

import os
cd=os.getcwd()
import sys
sys.path.append(os.path.join(cd,'utils')) 
import tropoe_utils as trp
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool
import xarray as xr
import subprocess
import shutil
import yaml
import glob as glob

plt.close('all')

#%% Inputs

if len(sys.argv)==1:
    site='nwtc.z02'
    sdate='20220510'
    edate='20220515'
    option='parallel'
    source_config=os.path.join(cd,'configs/config_basic.yaml')
else:
    site=sys.argv[1]
    sdate=sys.argv[2]
    edate=sys.argv[3]
    option=sys.argv[4]
    source_config=os.path.join(cd,'configs',sys.argv[5])
    
#%% Fuctions
def process_day(date,config):
    
    '''
    Run TROPoe for specific day. Downloads and reformats IRS and auxiliary data.
    '''
    
    #extract config
    channel_irs=config['channel_irs'][site]
    channel_cbh=config['channel_cbh'][site]
    channel_met=config['channel_met'][site]
    site_prior=config['site_prior'][site]
    verbosity=config['verbosity']
    image_name=config['image_name']
    image_type=config['image_type']
    
    #check list of processed files
    if os.path.exists(os.path.join(cd,'data/processed-{site}.txt'.format(site=site))):
        with open(os.path.join(cd,'data/processed-{site}.txt'.format(site=site))) as fid:
            processed=fid.readlines()
        processed=[p.strip() for p in processed]
    else: 
        processed=[]

    if sum([date == p for p in processed])==0:

        #create daily logger
        logger,handler=utl.create_logger(os.path.join('log',site,date+'.log'))
        
        #define temporary directories
        tmpdir=os.path.join(cd,'data',channel_irs,date+'-tmp')
        
        logger.info('Running TROPoe at '+site+' on '+date)
        print('Running TROPoe at '+site+' on '+date)

        #create input files
        command=config['path_python']+f' tropoe_inputs.py {site} {date} {source_config} {tmpdir}'
        result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
        logger.info(result.stdout)
        logger.error(result.stderr)
        
        #check input files
        if len(glob.glob(os.path.join(cd,'data',channel_cbh.replace('a0','cbh'),'*'+date+'*')))==0 and channel_cbh !="":
            logger.error('No CBH inputs found. Skipping.')
            return 
        
        if len(glob.glob(os.path.join(cd,'data',channel_met.replace('00','sel'),'*'+date+'*')))==0 and channel_met !="":
            logger.error('No met inputs found. Skipping.')
            return
        
        if len(glob.glob(os.path.join(tmpdir,'ch1','*'+date+'*cdf')))==1 and len(glob.glob(os.path.join(tmpdir,'sum','*'+date+'*cdf')))==1:
            f_ch1=glob.glob(os.path.join(tmpdir,'ch1','*'+date+'*cdf'))[0]
            f_sum=glob.glob(os.path.join(tmpdir,'sum','*'+date+'*cdf'))[0]
            
            #time check
            Data_ch1=xr.open_dataset(f_ch1)
            time_ch1=np.sort(Data_ch1['time'].values+Data_ch1['base_time'].values/10**3)
            del(Data_ch1)
            
            Data_sum=xr.open_dataset(f_sum)
            time_sum=np.sort((Data_sum['time'].values+Data_sum['base_time'].values)/np.timedelta64(1,'s'))
            
            if np.abs(np.nanmax(time_ch1)-np.nanmax(time_sum))>config['max_time_diff'] or np.abs(np.nanmin(time_ch1)-np.nanmin(time_sum))>config['max_time_diff']:
                logger.error('Inconsistent time on '+date+'. Skipping.')
                utl.close_logger(logger, handler)
                return
        else:
            logger.error('Missing or multiple files found on '+date+'. Skipping.')
            utl.close_logger(logger, handler)
            return
        
        #run TROPoe
        vip_file=f'data/{channel_irs}/{date}-tmp/vip_{site}.{date}.txt'
        month=date[4:6]
        prior_file=f'prior/Xa_Sa_datafile.{site_prior}.55_levels.month_{month}.cdf'
        command =f'./run_tropoe_ops.sh {date} {vip_file} {prior_file} 0 24 {verbosity} {cd} {cd} {image_name} {image_type}'
        logger.info('The following will be executed: \n'+command+'\n')
        result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
        logger.info(result.stdout)
        logger.error(result.stderr)
        
        #post-processing
        if len(glob.glob(os.path.join(cd,'data',channel_irs.replace('00','c0').replace('assist','assist.tropoe'),'*'+date+'*nc')))==1:
            
            #add to processed list
            with open(os.path.join(cd,'data/processed-{site}.txt'.format(site=site)), 'a') as fid:
                fid.write(date+'\n')
            
            #clear temp files
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
                
            file_tropoe=glob.glob(os.path.join(cd,'data',channel_irs.replace('00','c0').replace('assist','assist.tropoe'),'*'+date+'*nc'))[0]
            
            #close logger
            logger.info('Succesfully created retrieval '+file_tropoe)
            utl.close_logger(logger, handler)
            
            #plot maps
            Data=xr.open_dataset(file_tropoe)
            trp.plot_temp_wvmr(Data,config,file_tropoe)
            plt.savefig(file_tropoe.replace('.nc','_T_r.png'))
            plt.close()
            
        else:
            logger.info('Skipping '+f_ch1)
            utl.close_logger(logger, handler)
            return
    
#%% Initialization

#inputs
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils']) 
import utils as utl

#WDH setup
sys.path.append(config['path_dap_py']) 
from doe_dap_dl import DAP

a2e = DAP('a2e.energy.gov',confirm_downloads=False)
a2e.setup_cert_auth(username=config['username'], password=config['password'])

#clear up space on docker
if config['image_type']=='docker':
    command='docker image prune -f'
    result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
    print(result.stdout)
    print(result.stderr)

#change files permission
command='chmod -R 777 '+cd
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
print(result.stdout)
print(result.stderr)

#create directories
os.makedirs(os.path.join(cd,'data',config['channel_irs'][site]).replace('00','c0').replace('assist','assist.tropoe'),exist_ok=True)
os.makedirs(os.path.join('log',site),exist_ok=True)

# Loop to generate the range of datetimes
days=[]
current_date = datetime.strptime(sdate,'%Y%m%d')
while current_date <= datetime.strptime(edate,'%Y%m%d'):
    days.append(current_date)
    current_date += timedelta(days=1)
    
#download all data
if config['channel_irs'][site]!="":
    time_range = [datetime.strftime(datetime.strptime(sdate, '%Y%m%d')-timedelta(days=config['N_days_nfc']-1),'%Y%m%d%H%M%S'),
                  datetime.strftime(datetime.strptime(edate, '%Y%m%d')+timedelta(days=1),'%Y%m%d%H%M%S')]
    n_files_irs=trp.download(config['channel_irs'][site],time_range,config)
    print(str(n_files_irs)+' ASSIST files downloaded')
    
if config['channel_cbh'][site]!="":
    time_range = [datetime.strftime(datetime.strptime(sdate, '%Y%m%d'),'%Y%m%d%H%M%S'),
                  datetime.strftime(datetime.strptime(edate, '%Y%m%d')+timedelta(days=0.9999),'%Y%m%d%H%M%S')]
    n_files_cbh=trp.download(config['channel_cbh'][site],time_range,config)
    print(str(n_files_cbh)+' CBH files downloaded')
    
if config['channel_met'][site]!="":
    time_range = [datetime.strftime(datetime.strptime(sdate, '%Y%m%d'),'%Y%m%d%H%M%S'),
                  datetime.strftime(datetime.strptime(edate, '%Y%m%d')+timedelta(days=0.9999),'%Y%m%d%H%M%S')]
    n_files_met=trp.download(config['channel_met'][site],time_range,config)
    print(str(n_files_met)+' met files downloaded')

#process
if option=='serial':
    for d in days:
        date=datetime.strftime(d,'%Y%m%d')
        process_day(date,config)
elif option=='parallel':
    args = [(datetime.strftime(days[i],'%Y%m%d'), config) for i in range(len(days))]
 
    # Use multiprocessing Pool to parallelize the task
    with Pool() as pool:
        pool.starmap(process_day, args)
else:
    raise ValueError(f'Input "option" should be either serial or parallel, not {option}')
        