# -*- coding: utf-8 -*-
"""
Auxiliary modules for TROPoe
"""

def download(channel,time_range,config):
    import sys
    import os
    cd=os.getcwd()
    sys.path.append(config['path_dap_py']) 
    from doe_dap_dl import DAP
    
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    a2e.setup_cert_auth(username=config['username'], password=config['password'])
    
    _filter={}
    
    if 'assist' in channel:
        _filter = {
                    'Dataset': channel,
                    'date_time': {
                        'between': time_range
                    },
                    'file_type': 'cdf',
                    'ext1':['assistsummary','assistcha'],
                }
        
    elif 'lidar' in channel:
        _filter = {
                    'Dataset': channel,
                    'date_time': {
                        'between': time_range
                    },
                    'file_type': 'nc',
                    'ext1':['user1'],
                }
        
            
    elif 'rhod.met' in channel:
        _filter = {
                    'Dataset': channel,
                    'date_time': {
                        'between': time_range
                    },
                    'file_type': 'csv',
                }
    
    
    files=a2e.search(_filter)
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel), replace=False)
    return len(files)

def copy_rename_assist(channel,date):
    import os
    cd=os.getcwd()
    import glob
    import shutil
    '''
    rename and copy assist summary files
    '''
    files=glob.glob(os.path.join(cd,'data',channel,'*'+date+'*cdf'))

    for f in files:
        old_name=os.path.basename(f).split('.')
        new_name=old_name[-2]+'.'+old_name[-4]+'.'+old_name[-3]+'.cdf'
        
        if not os.path.exists(os.path.join(cd,'data',channel,'ch1')):
            os.mkdir(os.path.join(cd,'data',channel,'ch1'))
            os.mkdir(os.path.join(cd,'data',channel,'sum'))
        
        if 'assistcha' in f:
            shutil.copy(f,os.path.join(cd,'data',channel,'ch1',new_name))
        elif 'assistsummary' in f:
            shutil.copy(f,os.path.join(cd,'data',channel,'sum',new_name))

        

def compute_cbh_halo(channel,date,config,logger):
    '''
    Generate daily CBH time series for TROPoe
    '''
    import os
    cd=os.getcwd()
    import glob
    import sys
    sys.path.append(config['path_utils']) 
    import cbh_halo as cbh
    import utils as utl
    import numpy as np
    import xarray as xr
    from datetime import datetime
    
    if not os.path.exists(os.path.join(cd,'data',channel.replace('a0','cbh'))):
        os.mkdir(os.path.join(cd,'data',channel.replace('a0','cbh')))
         
    files=sorted(glob.glob(os.path.join(cd,'data',channel,'*'+date+'*nc')))
    tnum_cbh_all=[]
    cbh_all=[]
    for f in files:
        try:
            time_cbh,cbh_lidar=cbh.compute_cbh(f,utl,plot=config['detailed_plots'])
            tnum_cbh_all=np.append(tnum_cbh_all,(time_cbh-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1, 's'))
            cbh_all=np.append(cbh_all,cbh_lidar)
        except:
            logger.error('CBH estimation failed on '+f)
        
    basetime_cbh=utl.floor(tnum_cbh_all[0],24*3600)
    time_offset_cbh=tnum_cbh_all-basetime_cbh
    
    if np.max(np.diff(np.concatenate([[0],time_offset_cbh,[3600*24]])))>config['max_data_gap']:
        logger.error('Unallowable data gap found in CBH data. Aborting.')
        raise BaseException()

    Output_cbh=xr.Dataset()
    Output_cbh['first_cbh']=xr.DataArray(data=np.int32(np.nan_to_num(cbh_all,nan=-9999)),
                                         coords={'time':np.int32(time_offset_cbh)})

    Output_cbh['base_time']=np.int64(basetime_cbh)
    Output_cbh.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
    Output_cbh.to_netcdf(os.path.join(cd,'data',channel.replace('a0','cbh'),channel.replace('a0','ceil').replace('wfip3/','')+'.'+utl.datestr(basetime_cbh,'%Y%m%d.%H%M%S')+'.nc'))

    return Output_cbh

def exctract_met(channel,date,site,config,logger):
    '''
    Extract met information
    '''

    import os
    cd=os.getcwd()
    import glob
    import sys
    sys.path.append(config['path_utils']) 
    import utils as utl
    import numpy as np
    import xarray as xr
    import pandas as pd
    from datetime import datetime
    
    if site=='rhod':
        met_headers=config['met_headers_rhod']
        if not os.path.exists(os.path.join(cd,'data',channel.replace('00','sel'))):
            os.mkdir(os.path.join(cd,'data',channel.replace('00','sel')))
            
        files=sorted(glob.glob(os.path.join(cd,'data',channel,'*'+date+'*csv')))
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
        
        if np.max(np.diff(np.concatenate([[0],time_offset_met,[3600*24]])))>config['max_data_gap']:
            logger.error('Unallowable data gap found in met data. Aborting.')
            raise BaseException()

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
        Output_met.to_netcdf(os.path.join(cd,'data',channel.replace('00','sel'),channel.replace('00','sel').replace('wfip3/','')+'.'+utl.datestr(basetime_met,'%Y%m%d.%H%M%S')+'.nc'))

        return Output_met