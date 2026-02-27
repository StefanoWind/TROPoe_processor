'''
Estimates first cloud base height from single file of lidar data. Based on Newsom et al. 2019 [https://www.arm.gov/publications/tech_reports/doe-sc-arm-tr-149.pdf]
'''

import os
cd=os.path.dirname(__file__)

import xarray as xr
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
plt.close('all')

#%% Function
def moving_average(data,window):
    import pandas as pd
    
    df = pd.DataFrame(data.T)
    return df.rolling(window=window,center=True,min_periods=1).mean().values.T

def compute_cbh(file,utl,averages=60,signal='beta',plot=True):
    '''
    Estimate CBH from lidar data
    '''
    
    #fixed inputs
    rmax=12000#maximum range
    rmin=100#minimum range
    window=5#window size for smoothing in radial direction
    min_dz=50#[m] minimum cloud thickness
    max_dz=500#[m] maximum cloud thickness
    ang_tol=0.25#[deg] angular tolerance
    min_ele=45#[deg] minimum elevation
    min_Nt=100#minimum number of beams

    if plot:
        figname_save=file.replace('.nc','_cbh.png')
       
    #load data
    Data=xr.open_dataset(file).rename({'wind_speed':'radial_wind_speed'})
    
    if 'overlapped' in Data.attrs['Scan type']:
        range_name='distance_overlapped'
    else:
        range_name='distance'
    Data=Data.where(Data[range_name]<rmax,drop=True).where(Data[range_name]>rmin,drop=True).where(Data['elevation']+ang_tol>=min_ele,drop=True)
 
    azi=utl.round(Data['azimuth'].values,ang_tol)%360
    ele=utl.round(Data['elevation'].values,ang_tol)
   
    #count beams
    azi_ele=np.array([azi[:,0],ele[:,0]])
    Nb=np.shape(np.unique(azi_ele,axis=1))[1]
    reps=int(len(Data.time)/Nb)
    
    #drop incomplete scans
    Data=Data.isel(time=slice(0,reps*Nb))
    rws0=Data['radial_wind_speed'].values
    z0=(Data[range_name]*utl.sind(Data['elevation'])).values.T
    
    #time info
    time=Data['time'].values  
    tnum=utl.dt64_to_num(time)
    Nt=len(time)
    if Nt<min_Nt:
        print('Min number of beams violated')
        return
        
    #select measurand
    if signal=='snr':
        #snr
        f0=moving_average((Data['intensity']-1).values,window)
        tol=10**2#[m] prominence of peak snr gradient
        label='SNR [dB]'
        
    elif signal=='range-corrected snr':
        #range-corrected snr
        f0=moving_average(((Data['intensity']-1)*Data[range_name]**2).values,window)
        tol=10**5#[m] prominence of peak range-corrected snr gradient
        label=r'Range-corrected SNR [dB m$^2$]'
        
    elif signal=='beta':
        #attenuated backscatter
        f0=moving_average(Data['beta'].values,window)
        tol=1*10**-6#[m^-2 sr^-1] prominence of peak of beta gradient
        label=r'$\beta$ [m$^{-1}$ sr$^{-1}$]'
        
    else:
        raise ValueError(signal+' is not an allowed variable for CBH estimation')

    #ground conditions
    z=utl.hstack(np.zeros((Nt,1)),z0)
    f=utl.hstack(np.zeros((Nt,1)),f0)
    rws=utl.hstack(np.zeros((Nt,1)),rws0)
    Nr=len(z[0,:])

    #reshape into rep x beam x range
    tnum_3d=tnum.reshape(reps,Nb)
    f_3d=f.reshape(reps,Nb,Nr)
    
    #time average
    duration=np.nanmedian(np.diff(tnum_3d,axis=0))
    bin_reps=np.arange(0,reps+1,int(averages/duration))
    tnum_avg0=[]
    f_avg=[]
    for rep1,rep2 in zip(bin_reps[:-1],bin_reps[1:]):
        sel=np.arange(rep1,rep2)
        tnum_avg0=np.append(tnum_avg0,np.nanmean(tnum_3d[sel,:],axis=0))
        f_avg=   utl.vstack(f_avg,    np.nanmean(f_3d[sel,:,:],axis=0))
        
    #gradients
    df_dz=np.gradient(f_avg,axis=1)
    
    #CBH estimation [Newsom et al. 2016]
    cbh=np.zeros(len(f_avg[:,0]))+np.nan
    for i in range(len(f_avg[:,0])):
        
        jmin=np.argmin(df_dz[i,:])
        jmax=np.argmax(df_dz[i,:])
        dz=z[i,jmin]-z[i,jmax]
        if dz>=min_dz and dz<=max_dz and df_dz[i,jmin]<-tol and df_dz[i,jmax]>tol:
            j=np.argmax(f[i,jmax:jmin])
            cbh[i]=z[i,jmax+j]
            
    #scan average
    bin_reps=np.arange(0,len(cbh)+1,Nb)
    tnum_avg=[]
    cbh_avg=[]
    for rep1,rep2 in zip(bin_reps[:-1],bin_reps[1:]):
        sel=np.arange(rep1,rep2)
        tnum_avg=np.append(tnum_avg,np.nanmean(tnum_avg0[sel]))
        cbh_avg= np.append(cbh_avg, np.nanmean(cbh[sel]))
    
    time_avg=tnum_avg.astype('datetime64[s]')
    
    #plots
    if plot:
        date_fmt = mdates.DateFormatter('%H:%M')
        time_plot=[datetime.utcfromtimestamp(t) for t in tnum]
        time_avg_plot=[datetime.utcfromtimestamp(t) for t in tnum_avg]
        z_plot=np.arange(0,np.nanmax(z),np.nanmedian(np.diff(z,axis=1)))
        
        rws_int=np.zeros((len(time),len(z_plot)))
        f_int=np.zeros((len(time),len(z_plot)))
        for i in range(len(time)):
            rws_int[i,:]=np.interp(z_plot,z[i,:],rws[i,:])
            f_int[i,:]=np.interp(z_plot,z[i,:],f[i,:])
        
        plt.figure(figsize=(18,8))
        plt.subplot(2,1,1)
        pc=plt.pcolor(time_plot,z_plot,rws_int.T,cmap='seismic')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.ylabel(r'$z$ [m]')
        plt.scatter(time_avg_plot,cbh_avg,s=50,color='c',edgecolor='k')
        plt.title(os.path.basename(file))
        plt.colorbar(pc,label=r'$u_{LOS}$ [m s$^{-1}$]')
        
        plt.subplot(2,1,2)
        pc=plt.pcolor(time_plot,z_plot,f_int.T,cmap='hot')
        plt.scatter(time_avg_plot,cbh_avg,s=50,color='c',edgecolor='k')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.ylabel(r'$z$ [m]')
        plt.colorbar(pc,label=label)
        
        plt.savefig(figname_save,dpi=150)
        plt.close()
        
    return time_avg,cbh_avg
