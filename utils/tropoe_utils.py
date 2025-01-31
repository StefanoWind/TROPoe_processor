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
        
    elif 'barg.ecflux' in channel:
        _filter = {
                    'Dataset': channel,
                    'date_time': {
                        'between': time_range
                    },
                    'file_type': 'nc',
                }
   
    
    files=a2e.search(_filter)
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel), replace=False)
    
    if files==[]:
        return 0
    else:
        return len(files)

def copy_rename_assist(channel,sdate, edate,chassistdir,sumassistdir):
    import os
    cd=os.getcwd()
    import glob
    import shutil
    from datetime import datetime,timedelta
    '''
    Rename and copy assist summary files
    '''
    
    #create folders
    os.makedirs(chassistdir,exist_ok=True)
    os.makedirs(sumassistdir,exist_ok=True)
    
    #define date rage
    dates = []
    cdate = datetime.strptime(sdate,'%Y%m%d')
    while cdate <= datetime.strptime(edate,'%Y%m%d'):
        dates.append(datetime.strftime(cdate,'%Y%m%d'))
        cdate += timedelta(days=1)
        
    #copy  and rename files
    for d in dates:
        files=glob.glob(os.path.join(cd,'data',channel,'*'+d+'*cdf'))
    
        for f in files:
            old_name=os.path.basename(f).split('.')
            new_name=old_name[-2]+'.'+old_name[-4]+'.'+old_name[-3]+'.cdf'
            
            if 'cha' in f:
                shutil.copy(f,os.path.join(chassistdir,new_name))
            elif 'summary' in f:
                shutil.copy(f,os.path.join(sumassistdir,new_name))

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

    os.makedirs(os.path.join(cd,'data',channel[:-2]+'cbh'),exist_ok=True)
         
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
    name_save=channel.split('/')[1][:-1]+'ceil.'+utl.datestr(basetime,'%Y%m%d.%H%M%S')+'.nc'
    Output.to_netcdf(os.path.join(cd,'data',channel[:-2]+'cbh',name_save))

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
    
    os.makedirs(os.path.join(cd,'data',channel[:-2]+'sel'),exist_ok=True)
    
    if site=='rhod':
        met_headers=config['met_headers'][site]
        
        for f in files:
            try:
                Data=pd.read_csv(f)
            except:
                logger.error(f+' failed to load')
                
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
        
        #run this if using the raw met data
        if 'barg' in config['met_headers']:
            met_headers=config['met_headers'][site]
            for f in files:
                try:
                    Data=pd.read_csv(f, delimiter=r'\s+|,', engine='python',header=None)
                except:
                    logger.error(f+' failed to load')
                
                    #remove invalid rows
                    len_date=np.array([len(d) if d is not None else 0 for d in Data.iloc[:,0]])
                    len_time=np.array([len(t) if t is not None else 0 for t in Data.iloc[:,1]])
                    
                    Data=Data[(len_date==10)*(len_time==12)]
                    
                    for i in range(len(Data.index)):
                        tstr=Data.iloc[i,0]+' '+Data.iloc[i,1]
                        tnum_all=np.append(tnum_all,utl.datenum(tstr,'%Y/%m/%d %H:%M:%S.%f'))
                    temp_all=np.append(temp_all,Data.iloc[:,met_headers['temperature']].values)  
                    press_all=np.append(press_all,Data.iloc[:,met_headers['pressure']].values) 
                    rh_all=np.append(rh_all,Data.iloc[:,met_headers['humidity']].values) 
            
        #rund this if using the a0 met data
        else:
            for f in files:
                try:
                    Data=xr.open_dataset(f, decode_times=False)
                except:
                    logger.error(f+' failed to load')
                
                    tnum_all=np.append(tnum_all,(Data.time.values-719529)*60*60*24)#Matlab time ot UNIX timestamp
                    temp_all=np.append(temp_all,Data['air_temp_c'].sel(air_temp_sensors=0).values)
                    press_all=np.append(press_all,Data['air_press_mb'].sel(num_air_press=0).values)
                    rh_all=np.append(rh_all,Data['rh'].sel(rhT_sensors=0).values) 
        
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
    name_save=channel.split('/')[1][:-2]+'sel'+'.'+utl.datestr(basetime,'%Y%m%d.%H%M%S')+'.nc'
    Output.to_netcdf(os.path.join(cd,'data',channel[:-2]+'sel',name_save))

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
    
    os.makedirs(os.path.join(cd,'data',channel[:-2]+'cbh'),exist_ok=True)
    
    #load data
    files=sorted(glob.glob(os.path.join(cd,'data',channel,'*'+date+'*')))
    Data=xr.open_mfdataset(files,combine="nested",concat_dim="time")
    
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
    name_save=channel.split('/')[1][:-1]+'ceil.'+utl.datestr(basetime,'%Y%m%d.%H%M%S')+'.nc'
    Output.to_netcdf(os.path.join(cd,'data',channel[:-2]+'cbh',name_save))

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
   
def brightness_temp(rad,wnum):
    import numpy as np
    k=1.380649*10**-23#[J/Kg] Boltzman's constant
    h=6.62607015*10**-34#[J s] Plank's constant
    c=299792458.0#[m/s] speed of light

    Tb=100*h*c*wnum/k/np.log(2*10**11*c**2*h*wnum**3/rad+1)-273.15
    
    return Tb
def overrride_hatch_flag(files,thresh=1,replace_flag=[-1],logger=None):
    '''
    Override hatch flag and replace with proxy using difference of Tb at two wnums
    
    Inputs:
        files_sum: [list] summary files to process
        thresh: [K] maximum difference in brightness temperature between 990 and 680 cm^-1
        replace_flag: [list] flags to be replaces
    '''
    import numpy as np
    import xarray as xr
    import os

    for f in files:
        with xr.open_dataset(f) as Data0:
            Data = Data0.sortby('time').load()  
    
        #data extraction
        hatch=Data.hatchOpen
       
        if 'summary' in f:
            T1=Data.mean_Tb_985_990.where(np.abs(Data.sceneMirrorAngle)<0.1)
            T2=Data.mean_Tb_675_680.where(np.abs(Data.sceneMirrorAngle)<0.1)
        elif 'cha' in f:
            Tb=brightness_temp(Data.mean_rad,Data.wnum)
            T1=Tb.where(np.abs(Data.sceneMirrorAngle)<0.1).sel(wnum=slice(985,990)).mean(dim='wnum')
            T2=Tb.where(np.abs(Data.sceneMirrorAngle)<0.1).sel(wnum=slice(675,680)).mean(dim='wnum')
            
        #new hatch flag
        replace_open=np.abs(T1-T2)>thresh
        replace_closed=np.abs(T1-T2)<=thresh
        
        #select flags to be replaced
        sel=np.array([False]*len(hatch))
        for h in replace_flag:
            sel+=hatch==h
        
        #ovverride hatch flag
        new_hatch=hatch.where(np.abs(Data.sceneMirrorAngle)<0.1)
        new_hatch[sel*replace_open]=1
        new_hatch[sel*replace_closed]=0
        
        #add undefined flag when moving
        new_hatch=new_hatch.interpolate_na(dim='time')
        new_hatch[np.isnan(new_hatch)+(new_hatch>0)*(new_hatch<1)+(new_hatch>-1)*(new_hatch<0)]=-1
        
        #save new hatch flag in summary file
        Data['hatchOpen']=new_hatch
        
        logger.info(f'{np.sum((sel*replace_open).values)} hatch flag overridden with status "open".')
        logger.info(f'{np.sum((sel*replace_closed).values)} hatch flag overridden with status "closed".')
        
        Data.to_netcdf(f.replace('.cdf','_temp.cdf'))
        Data.close()
        os.replace(f.replace('.cdf','_temp.cdf'),f)
        
def plot_temp_wvmr(Data,config,filename=''):
    '''
    Plot time-height maps of QC temperature and water vapor mixing ratio
    '''
    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.dates as mdates
    from datetime import datetime
    from datetime import timedelta
    import os
    
    Data=Data.resample(time=str(np.median(np.diff(Data['time']))/np.timedelta64(1,'m'))+'min').nearest(tolerance='1min')

    
    time=np.array(Data['time'])
    date=str(Data.base_time.values)[:10].replace('-','')
    height0=np.array(Data['height'][:])*1000
    sel_z=height0<config['max_z']
    height=height0[sel_z]

    T=np.array(Data['temperature'].where(Data['gamma']<=config['max_gamma']).where(Data['rmsa']<=config['max_rmsa'])[:,sel_z])#[C]
    r= np.array(Data['waterVapor'].where(Data['gamma']<=config['max_gamma']).where(Data['rmsa']<=config['max_rmsa'])[:,sel_z])#[g/Kg]
    cbh=np.array(Data['cbh'].where(Data['gamma']<=config['max_gamma']).where(Data['rmsa']<=config['max_rmsa'])[:])*1000#[m]
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
    ax.set_xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
    ax.set_ylim(0, np.max(height)+10)
    ax.grid()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title('TROPoe retrieval at ' + Data.attrs['Site'] + ' on '+date+'\n File: '+os.path.basename(filename), x=0.45)
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
    ax.set_xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
    ax.set_ylim(0, np.max(height)+10)
    ax.grid()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_facecolor((0.9,0.9,0.9))