# -*- coding: utf-8 -*-
'''
Compare CBH from lidar and ceilometer
'''

import os
cd=os.getcwd()
import sys
sys.path.append(os.path.join(cd,'utils')) 
import tropoe_utils as trp
from matplotlib import pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from datetime import timedelta
import logging
import glob
import yaml

plt.close('all')

#%% Inputs
if len(sys.argv)==1:
    channel_ceil='wfip3/barg.ceil.z01.b0'
    channel_lid='wfip3/barg.lidar.z03.a0*stare'
    sdate='20240529'
    edate='20241010'
    source_config=os.path.join(cd,'configs/config_wfip3_c1_no_ceil.yaml')
else:
    channel_ceil=sys.argv[1]
    channel_lid=sys.argv[2]
    sdate=sys.argv[3]
    edate=sys.argv[4]
    source_config=os.path.join(cd,'configs',sys.argv[5])
    
#%% Functions
def process_day(date,config):
    logger,handler=utl.create_logger(os.path.join('log',channel_ceil.split('/')[1]+'-'+channel_lid.split('/')[1].split('*')[0],date+'.log'))
    logger = logging.getLogger()
    logger.info('Building TROPoe inputs for '+date+' for CBH comparison between '+channel_ceil+' and '+channel_lid)
    
    if len(glob.glob(os.path.join(cd,'data',channel_ceil.replace(channel_ceil[-2:],'cbh'),'*'+date+'*nc')))==0:
        Output_cbh_ceil=trp.extract_cbh_ceil(channel_ceil,date,config,logger)
    if len(glob.glob(os.path.join(cd,'data',channel_lid.split('*')[0].replace(channel_lid.split('*')[0][-2:],'cbh'),'*'+date+'*nc')))==0:
        Output_cbh_lid=trp.compute_cbh_halo(channel_lid.split('*')[0],date,config,logger)
    # return Output_cbh_ceil,Output_cbh_lid
    
#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

sys.path.append(config['path_utils']) 
import utils as utl

days=[]
current_date = datetime.strptime(sdate,'%Y%m%d')
while current_date <= datetime.strptime(edate,'%Y%m%d'):
    days.append(current_date)
    current_date += timedelta(days=1)
    
os.makedirs(os.path.join(cd,'log',channel_ceil.split('/')[1]+'-'+channel_lid.split('/')[1].split('*')[0]),exist_ok=True)

#%% Main

#download
time_range = [datetime.strftime(datetime.strptime(sdate, '%Y%m%d'),'%Y%m%d%H%M%S'),
              datetime.strftime(datetime.strptime(edate, '%Y%m%d')+timedelta(days=0.9999),'%Y%m%d%H%M%S')]
n_files_cbh_ceil=trp.download(channel_ceil,time_range,'',config)
n_files_cbh_lid=trp.download(channel_lid.split('*')[0],time_range,channel_lid.split('*')[1],config)

#process    
args = [(datetime.strftime(days[i],'%Y%m%d'), config) for i in range(len(days))]

with Pool() as pool:
    pool.starmap(process_day, args)
    
    
        