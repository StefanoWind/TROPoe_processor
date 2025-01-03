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
                    'ext1':['assistsummary','assistcha','assistno10cha','assistno11cha','assistno12cha'],
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
    
    elif 'ceil' in channel:
        _filter = {
                    'Dataset': channel,
                    'date_time': {
                        'between': time_range
                    },
                    'file_type': 'nc',
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

def copy_rename_assist(channel,sdate, edate):
    import os
    cd=os.getcwd()
    import glob
    import shutil
    from datetime import datetime,timedelta
    '''
    Rename and copy assist summary files
    '''
    dates = []
    cdate = datetime.strptime(sdate,'%Y%m%d')
    while cdate <= datetime.strptime(edate,'%Y%m%d'):
        dates.append(datetime.strftime(cdate,'%Y%m%d'))
        cdate += timedelta(days=1)
    for d in dates:
        files=glob.glob(os.path.join(cd,'data',channel,'*'+d+'*cdf'))
    
        for f in files:
            old_name=os.path.basename(f).split('.')
            new_name=old_name[-2]+'.'+old_name[-4]+'.'+old_name[-3]+'.cdf'
            
            if not os.path.exists(os.path.join(cd,'data',channel,'ch1')):
                os.mkdir(os.path.join(cd,'data',channel,'ch1'))
                os.mkdir(os.path.join(cd,'data',channel,'sum'))
            
            if 'cha' in f:
                shutil.copy(f,os.path.join(cd,'data',channel,'ch1',new_name))
            elif 'summary' in f:
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
    tnum_all=[]
    cbh_all=[]
    for f in files:
        try:
            time_cbh,cbh_lidar=cbh.compute_cbh(f,utl,plot=config['detailed_plots'])
            tnum_all=np.append(tnum_all,(time_cbh-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1, 's'))
            cbh_all=np.append(cbh_all,cbh_lidar)
        except:
            logger.error('CBH estimation failed on '+f)
        
    basetime=utl.floor(tnum_all[0],24*3600)
    time_offset=tnum_all-basetime
    
    if np.max(np.diff(np.concatenate([[0],time_offset,[3600*24]])))>config['max_data_gap']:
        logger.error('Unallowable data gap found in CBH data. Aborting.')
        raise BaseException()

    Output=xr.Dataset()
    Output['first_cbh']=xr.DataArray(data=np.int32(np.nan_to_num(cbh_all,nan=-9999)),
                                         coords={'time':np.int32(time_offset)},
                                         attrs={'description':'First cloud base height','units':'m'})

    Output['base_time']=np.int64(basetime)
    Output.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
    Output.to_netcdf(os.path.join(cd,'data',channel.replace('a0','cbh'),channel.replace('a0','ceil').replace('wfip3/','')+'.'+utl.datestr(basetime,'%Y%m%d.%H%M%S')+'.nc'))

    return Output

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
    
    files=sorted(glob.glob(os.path.join(cd,'data',channel,'*'+date+'*')))
    tnum_all=[]
    temp_all=[]
    press_all=[]
    rh_all=[]
    
    if site=='rhod':
        met_headers=config['met_headers'][site]
        if not os.path.exists(os.path.join(cd,'data',channel.replace('00','sel'))):
            os.mkdir(os.path.join(cd,'data',channel.replace('00','sel')))
            
        
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
                     
                tnum_all=np.append(tnum_all,utl.datenum(tstr_met,'%Y-%m-%d %H:%M:%S.%f'))
            
            temp_all=np.append(temp_all,Data.iloc[:,met_headers['temperature']].values)  
            press_all=np.append(press_all,Data.iloc[:,met_headers['pressure']].values) 
            rh_all=np.append(rh_all,Data.iloc[:,met_headers['humidity']].values) 

    elif site=='barg':
        met_headers=config['met_headers'][site]
        if not os.path.exists(os.path.join(cd,'data',channel.replace('00','sel'))):
            os.mkdir(os.path.join(cd,'data',channel.replace('00','sel')))
        
        for f in files:
            try:
                Data=pd.read_csv(f, delimiter=r'\s+|,', engine='python',header=None)
            except:
                logger.error(f+' failed to load')
            
            len_date=np.array([len(d) if d is not None else 0 for d in Data.iloc[:,0]])
            len_time=np.array([len(t) if t is not None else 0 for t in Data.iloc[:,1]])
            
            Data=Data[(len_date==10)*(len_time==12)]
            
            for i in range(len(Data.index)):
                tstr=Data.iloc[i,0]+' '+Data.iloc[i,1]
                tnum_all=np.append(tnum_all,utl.datenum(tstr,'%Y/%m/%d %H:%M:%S.%f'))
            temp_all=np.append(temp_all,Data.iloc[:,met_headers['temperature']].values)  
            press_all=np.append(press_all,Data.iloc[:,met_headers['pressure']].values) 
            rh_all=np.append(rh_all,Data.iloc[:,met_headers['humidity']].values) 
        
    basetime=utl.floor(tnum_all[0],24*3600)
    time_offset=tnum_all-basetime
    
    if np.max(np.diff(np.concatenate([[0],time_offset,[3600*24]])))>config['max_data_gap']:
        logger.error('Unallowable data gap found in met data. Aborting.')
        raise BaseException()

    Output=xr.Dataset()
    Output['temp_mean']=     xr.DataArray(data=np.nan_to_num(temp_all,nan=-9999),
                                         coords={'time':np.arange(len(time_offset))},
                                         attrs={'description':'Surface air temperature','units':'C'})
    Output['atmos_pressure']=xr.DataArray(data=np.nan_to_num(press_all/10,nan=-9999),
                                         coords={'time':np.arange(len(time_offset))},
                                         attrs={'description':'Surface air pressure','units':'kPa'})
    Output['rh_mean']=       xr.DataArray(data=np.nan_to_num(rh_all,nan=-9999),
                                         coords={'time':np.arange(len(time_offset))},
                                         attrs={'description':'Surface relative humidity','units':'%'})
    Output['time_offset']=xr.DataArray(data=time_offset,
                                         coords={'time':np.arange(len(time_offset))},
                                         attrs={'description':'Time since midnight','units':'s'})
    Output['base_time']=np.float64(basetime)
    Output.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
    Output.to_netcdf(os.path.join(cd,'data',channel.replace('00','sel'),channel.replace('00','sel').replace('wfip3/','')+'.'+utl.datestr(basetime,'%Y%m%d.%H%M%S')+'.nc'))

    return Output

def extract_cbh_ceil(channel,date,config,logger):
    '''
    Extact first CBH from Vaisala ceilometer and format as TROPoe input
    '''
    import os
    cd=os.getcwd()
    import sys
    sys.path.append(config['path_utils']) 
    import utils as utl
    import glob
    import xarray as xr
    import numpy as np
    from datetime import datetime
    
    if not os.path.exists(os.path.join(cd,'data',channel.replace(channel[-2:],'cbh'))):
        os.mkdir(os.path.join(cd,'data',channel.replace(channel[-2:],'cbh')))
    
    #load data
    files=sorted(glob.glob(os.path.join(cd,'data',channel,'*'+date+'*')))
    Data=xr.open_mfdataset(files)
    
    cbh=np.float64(Data['cloud_data'].values[:,0])#first CBH
    
    #time info (UTC)
    tnum=np.float64(Data['time'].values)
    basetime=utl.floor(tnum[0],24*3600)
    time_offset=tnum-basetime
    
    if np.max(np.diff(np.concatenate([[0],time_offset,[3600*24]])))>config['max_data_gap']:
        logger.error('Unallowable data gap found in CBH data. Aborting.')
        raise BaseException()

    Output=xr.Dataset()
    Output['first_cbh']=xr.DataArray(data=np.int32(np.nan_to_num(cbh,nan=-9999)),
                                         coords={'time':np.int32(time_offset)},
                                         attrs={'description':'First cloud base height','units':'m'})

    Output['base_time']=np.int64(basetime)
    Output.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
    Output.to_netcdf(os.path.join(cd,'data',channel.replace('b0','cbh'),channel.replace('b0','ceil').replace('wfip3/','')+'.'+utl.datestr(basetime,'%Y%m%d.%H%M%S')+'.nc'))

def pre_filter(files,min_resp=0.7,max_ir=0.5,resp_wnum=1000,ir_wnum=985,logger=None):
    '''
    Add missingDataFlag to raw ASSIST data.
    
    Inputs:
        files: list of files to process
        min_resp: minimum responsivity at wnum=res_wnum as a fraction of maximum responsivity
        max_ir: maximum absolute value of imaginary radiance at wnum=ir_wnum
        logger: handler to logger
    '''
    import numpy as np
    import xarray as xr
    import os
    for f in files:
        Data=xr.open_dataset(f,mode='r')
        
        #data extraction
        resp=Data.mean_resp.interp(wnum=resp_wnum)
        ir=Data.mean_imag_rad.interp(wnum=ir_wnum)

        #QC
        f_all=((resp<min_resp*np.nanpercentile(resp, 99)) + (np.abs(ir)>max_ir))+0
        f_abb=f_all.where(np.round(Data.sceneMirrorAngle)==60)
        f_abb_fill=((f_abb.bfill(dim='time')+f_abb.ffill(dim='time'))>0)+0

        f_hbb=f_all.where(np.round(Data.sceneMirrorAngle)==300)
        f_hbb_fill=((f_hbb.bfill(dim='time')+f_hbb.ffill(dim='time'))>0)+0

        Data['missingDataFlag']=((f_abb_fill+f_hbb_fill+f_all)>0)+0
        
        logger.info(f'{np.sum(Data.missingDataFlag.values==1)} bad spectra identified in pre-filter.')
        
        Data.to_netcdf(f.replace('.cdf','_temp.cdf'))
        Data.close()
        os.replace(f.replace('.cdf','_temp.cdf'),f)