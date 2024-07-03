'''
Estimates first cloud base height from single file of lidar data. Based on Newsom et al. 20??
'''

import os
cd=os.path.dirname(__file__)
import sys

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

def compute_cbh(file,utl,averages=3,signal='range-corrected snr',plot=True):
    rmax=12000#maximum range
    rmin=100#minimum range
    window=5#window size for smoothing
    min_dz=50#[m] minimum cloud thickness
    max_dz=500#[m] maximum cloud thickness
    ang_tol=0.25#[deg] angular tolerance
    min_ele=45#[deg] minimum elevation
    min_Nt=100#minimum number of beams

    if plot:
        figname_save=file.replace('.nc','_cbh.png')
       
    #load data
    Data=xr.open_dataset(file)
    
    if 'overlapped' in Data.attrs['Scan type']:
        range_name='distance_overlapped'
    else:
        range_name='distance'
    Data=Data.where(Data[range_name]<rmax,drop=True).where(Data[range_name]>rmin,drop=True).where(Data['elevation']+ang_tol>=min_ele,drop=True)
    
    time=Data['time'].values
    Nt=len(time)
    if Nt<min_Nt:
        print('Min number of beams violated')
        return
    
    tnum=utl.dt64_to_num(time)
    azi=utl.round(Data['azimuth'].values,ang_tol)%360
    ele=utl.round(Data['elevation'].values,ang_tol)
    z0=(Data[range_name]*utl.sind(Data['elevation'])).values.T
    rws0=Data['radial_wind_speed'].values

    #count beams
    azi_ele=np.array([azi[:,0],ele[:,0]])
    Nb=np.shape(np.unique(azi_ele,axis=1))[1]
    reps=int(len(time)/Nb)
    if reps*Nb<Nt:
        print('Incomplete scan')
        return
        
    if signal=='snr':
        #snr
        f0=moving_average((Data['intensity']-1).values,window)
        tol=10**2#[m] prominence of peak snr gradient
    elif signal=='range-corrected snr':
        #range-corrected snr
        f0=moving_average(((Data['intensity']-1)*Data[range_name]**2).values,window)
        tol=10**5#[m] prominence of peak range-corrected snr gradient
    elif signal=='beta':
        #attenuated backscatter
        f0=moving_average(Data['beta'].values,window)
        tol=10**-5#[m^-2 sr^-1] prominence of peak of beta gradient
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
    
    #averages
    bin_reps=np.arange(0,reps+1,averages)
    tnum_avg=[]
    f_avg=[]
    for rep1,rep2 in zip(bin_reps[:-1],bin_reps[1:]):
        sel=np.arange(rep1,rep2)
        tnum_avg=np.append(tnum_avg,np.nanmean(tnum_3d[sel,:],axis=0))
        f_avg=utl.vstack(f_avg,np.nanmean(f_3d[sel,:,:],axis=0))
        
    time_avg=tnum_avg.astype('datetime64[s]')
        
    #gradients
    df_dz=np.gradient(f_avg,axis=1)
    
    #CBH estimation [Newsom et al. 2016]
    cbh=np.zeros(len(tnum_avg))+np.nan
    for i in range(len(tnum_avg)):
        
        jmin=np.argmin(df_dz[i,:])
        jmax=np.argmax(df_dz[i,:])
        dz=z[i,jmin]-z[i,jmax]
        if dz>=min_dz and dz<=max_dz and df_dz[i,jmin]<-tol and df_dz[i,jmax]>tol:
            j=np.argmax(f[i,jmax:jmin])
            cbh[i]=z[i,jmax+j]
    
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
        plt.subplot(3,1,1)
        plt.pcolor(time_plot,z_plot,rws_int.T,cmap='seismic')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.ylabel(r'$z$ [m]')
        plt.plot(time_avg_plot,cbh,'.c')
        plt.title(os.path.basename(file))
        plt.colorbar(label=r'$u_{LOS}$ [m s$^{-1}$]')
        
        plt.subplot(3,1,2)
        plt.pcolor(time_plot,z_plot,f_int.T,cmap='hot')
        plt.plot(time_avg_plot,cbh,'.c')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.ylabel(r'$z$ [m]')
        plt.colorbar(label=signal)
        
        plt.savefig(figname_save)
        plt.close()
        
    return time_avg,cbh
