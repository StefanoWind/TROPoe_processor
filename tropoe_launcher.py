'''
Run TROPoe retrieval and store results
'''

import os
cd=os.path.dirname(os.path.abspath(__file__))
import sys
from utils import utils as utl
from utils import tropoe_utils as trp
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
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

#%% Inputs

if len(sys.argv)==1:
    site='sa_rt'
    sdate='20260803'
    edate='20260803'
    option='serial'
    source_config=os.path.join(cd,'configs/config_anvil.yaml')
else:
    site=sys.argv[1]
    sdate=sys.argv[2]
    edate=sys.argv[3]
    option=sys.argv[4]
    source_config=os.path.join(cd,'configs',sys.argv[5])
    
#%% Fuctions
def process_day(date,config,option):

    '''
    Run TROPoe for specific day. Downloads and reformats IRS and auxiliary data.
    '''

    #extract config
    channel_irs=config['channel_irs'][site].replace('raw','00')
    channel_cbh=config['channel_cbh'][site].split('*')[0]
    channel_met=config['channel_met'][site]
    site_prior=config['site_prior'][site]
    prior_file=config['prior_file'][site]
    verbosity=config['verbosity']
    image_name=config['image_name']
    image_type=config['image_type']

    #use monthly prior if provided
    if prior_file == "":
        month=date[4:6]
        prior_file=f'prior/Xa_Sa_datafile.{site_prior}.55_levels.month_{month}.cdf'

    #split the day into retrieval chunks; full days are kept whole when days are already run in parallel
    hours_process=24 if option=='parallel' else config.get('hours_process',24)
    if 24%hours_process!=0:
        raise ValueError(f"hours_process ({hours_process}) must evenly divide into 24 hours.")
    chunks=[(h,h+hours_process) for h in range(0,24,hours_process)]
    tags={(shour,ehour):f'{date}.{shour:02d}0000' for shour,ehour in chunks}

    processed_file=os.path.join(cd,'data/processed-{site}.txt'.format(site=site))
    def read_processed():
        if os.path.exists(processed_file):
            with open(processed_file) as fid:
                return [p.strip() for p in fid.readlines()]
        return []

    if all(tags[c] in read_processed() for c in chunks):
        return

    #try to acquire a lock for the shared, day-level input build so two concurrent
    #instances never rebuild/clear the same tmpdir while a chunk retrieval is reading it
    lockdir=os.path.join(cd,'data','locks',site)
    os.makedirs(lockdir,exist_ok=True)
    build_lockfile=os.path.join(lockdir,date+'.build.lock')
    try:
        fd=os.open(build_lockfile,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(date+' is already being processed by another instance. Skipping.')
        return

    locked=[]
    try:
        pending=[c for c in chunks if tags[c] not in read_processed()]
        if len(pending)==0:
            return

        #acquire a lock per chunk, named after its output timestamp
        for shour,ehour in pending:
            tag=tags[(shour,ehour)]
            lockfile=os.path.join(lockdir,tag+'.lock')
            try:
                fd=os.open(lockfile,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                os.close(fd)
                locked.append((shour,ehour,tag,lockfile))
            except FileExistsError:
                print(tag+' is already being processed by another instance. Skipping.')
        if len(locked)==0:
            return

        try:
            #create daily logger
            logger,handler=utl.create_logger(os.path.join(cd,'log',site,date+'.log'))

            #define temporary directories
            tmpdir=os.path.join(cd,'data',channel_irs,date+'-tmp')

            logger.info('Running TROPoe at '+site+' on '+date)
            print('Running TROPoe at '+site+' on '+date)

            #create input files, shared by every chunk of this day
            command=config['path_python']+f' {os.path.join(cd,"tropoe_inputs.py")} {site} {date} {source_config} {tmpdir}'
            result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
            logger.info(result.stdout)
            logger.error(result.stderr)

            #check input files
            if len(glob.glob(os.path.join(cd,'data',channel_cbh[:-2]+'cbh','*'+date+'*')))==0 and channel_cbh !="":
                logger.error('No cbh inputs found.')
                no_cbh=True
                if config['allow_no_cbh']==False:
                    return
            else:
                no_cbh=False

            if len(glob.glob(os.path.join(cd,'data',channel_met[:-2]+'sel','*'+date+'*')))==0 and channel_met !="":
                logger.error('No met inputs found.')
                no_met=True
                if config['allow_no_met']==False:
                    return
            else:
                no_met=False

            if len(glob.glob(os.path.join(tmpdir,'ch1*','*'+date+'*cdf')))==1 and len(glob.glob(os.path.join(tmpdir,'sum*','*'+date+'*cdf')))==1:
                f_ch1=glob.glob(os.path.join(tmpdir,'ch1*','*'+date+'*cdf'))[0]
                f_sum=glob.glob(os.path.join(tmpdir,'sum*','*'+date+'*cdf'))[0]

                #time check
                Data_ch1=xr.open_dataset(f_ch1)
                time_ch1=np.sort(Data_ch1['time'].values+Data_ch1['base_time'].values/10**3)
                del(Data_ch1)

                Data_sum=xr.open_dataset(f_sum,decode_timedelta=False)
                time_sum=np.sort(Data_sum['time'].values+Data_sum['base_time'].values/10**3)
                del(Data_sum)

                if np.abs(np.nanmax(time_ch1)-np.nanmax(time_sum))>config['max_time_diff'] or np.abs(np.nanmin(time_ch1)-np.nanmin(time_sum))>config['max_time_diff']:
                    logger.error('Inconsistent time on '+date+'. Skipping.')
                    utl.close_logger(logger, handler)
                    return
            else:
                logger.error('Missing or multiple files found on '+date+'. Skipping.')
                utl.close_logger(logger, handler)
                return

            #launch every pending chunk at once: Popen does not block, unlike subprocess.run,
            #so all containers start together instead of running one after another
            vip_file=f'/data/data/{channel_irs}/{date}-tmp/vip_{site}.{date}.txt'
            tropoe_shell=config['tropoe_shell']
            procs=[]
            for shour,ehour,tag,lockfile in locked:
                chunk_tmp=os.path.join(tmpdir,f'tmp2_{shour:02d}{ehour:02d}')
                os.makedirs(chunk_tmp,exist_ok=True)
                command=f'{os.path.join(cd,tropoe_shell)} {date} {vip_file} {prior_file} {shour} {ehour} {verbosity} {cd} {chunk_tmp} {image_name} {image_type}'
                logger.info('The following will be executed: \n'+command+'\n')
                proc=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
                procs.append((shour,ehour,tag,lockfile,chunk_tmp,proc))

            #wait for each chunk and post-process it independently as it finishes
            for shour,ehour,tag,lockfile,chunk_tmp,proc in procs:
                stdout,stderr=proc.communicate()
                logger.info(stdout)
                logger.error(stderr)

                matches=glob.glob(os.path.join(config['output_dir'][site],f'*{date}.{shour:02d}0000*nc'))
                if len(matches)==1:
                    file_tropoe=matches[0]

                    #add chunk to processed list
                    with open(processed_file, 'a') as fid:
                        fid.write(tag+'\n')

                    logger.info('Succesfully created retrieval '+file_tropoe)

                    #plot maps
                    Data=xr.open_dataset(file_tropoe)
                    trp.plot_temp_wvmr(Data,config,file_tropoe,no_cbh,no_met)
                    plt.savefig(file_tropoe.replace('.nc','_T_r.png'))
                    plt.close()
                else:
                    logger.info('Skipping chunk '+tag+': no output produced.')

                if os.path.exists(chunk_tmp):
                    shutil.rmtree(chunk_tmp)
                if os.path.exists(lockfile):
                    os.remove(lockfile)

            utl.close_logger(logger, handler)

            #clear the shared temp files only once every chunk of this day is accounted for
            if all(tags[c] in read_processed() for c in chunks) and os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
        finally:
            #release any chunk lock not already released above (e.g. early-return paths)
            for shour,ehour,tag,lockfile in locked:
                if os.path.exists(lockfile):
                    os.remove(lockfile)
    finally:
        if os.path.exists(build_lockfile):
            os.remove(build_lockfile)

#%% Initialization

#inputs
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#clear up space on docker
if config['image_type']=='docker':
    command='docker image prune -f'
    result=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True) 
    print(result.stdout)
    print(result.stderr)

#change files permission
command='chmod -R 777 '+cd
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

#create directories
os.makedirs(config['output_dir'][site],exist_ok=True)
os.makedirs(os.path.join(cd,'log',site),exist_ok=True)

# Loop to generate the range of datetimes
days=[]
current_date = datetime.strptime(sdate,'%Y%m%d')
while current_date <= datetime.strptime(edate,'%Y%m%d'):
    days.append(current_date)
    current_date += timedelta(days=1)
    
#download all data
if not "raw" in config['channel_irs'][site]:
    if config['channel_irs'][site]!="":
        time_range = [datetime.strftime(datetime.strptime(sdate, '%Y%m%d')-timedelta(days=config['N_days_nfc'][site]-1),'%Y%m%d%H%M%S'),
                      datetime.strftime(datetime.strptime(edate, '%Y%m%d')+timedelta(days=1),'%Y%m%d%H%M%S')]
        n_files_irs=trp.download(config['channel_irs'][site],time_range,'',config)
        print(str(n_files_irs)+' ASSIST files downloaded')
        
    if config['channel_cbh'][site]!="":
        time_range = [datetime.strftime(datetime.strptime(sdate, '%Y%m%d'),'%Y%m%d%H%M%S'),
                      datetime.strftime(datetime.strptime(edate, '%Y%m%d')+timedelta(days=0.9999),'%Y%m%d%H%M%S')]
        n_files_cbh=trp.download(config['channel_cbh'][site].split('*')[0],time_range,config['channel_cbh'][site].split('*')[1],config)
        print(str(n_files_cbh)+' cbh files downloaded')
        
    if config['channel_met'][site]!="":
        time_range = [datetime.strftime(datetime.strptime(sdate, '%Y%m%d'),'%Y%m%d%H%M%S'),
                      datetime.strftime(datetime.strptime(edate, '%Y%m%d')+timedelta(days=0.9999),'%Y%m%d%H%M%S')]
        n_files_met=trp.download(config['channel_met'][site],time_range,'',config)
        print(str(n_files_met)+' met files downloaded')

#process
if option=='serial':
    for d in days:
        date=datetime.strftime(d,'%Y%m%d')
        process_day(date,config,option)
elif option=='parallel':
    args = [(datetime.strftime(days[i],'%Y%m%d'), config, option) for i in range(len(days))]

    # Use multiprocessing Pool to parallelize the task
    with Pool() as pool:
        pool.starmap(process_day, args)
else:
    raise ValueError(f'Input "option" should be either "serial" or "parallel", not {option}')
        