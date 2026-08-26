# -*- coding: utf-8 -*-
"""
Check progress in TROPoe files
"""
import os
cd=os.getcwd()
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import sys
import xarray as xr
import glob
import numpy as np

#%% Inputs
if len(sys.argv)==2:
    folder=sys.argv[1]
else:
    folder=input('Folder: ')

#%% Functions
def progress_bar(progress, total, length=40):
    percent = progress / total
    bar = "|" * int(percent * length) + "-" * (length - int(percent * length))
    sys.stdout.write(f"\r[{bar}] {percent * 100:.1f}%")
    sys.stdout.flush()
    
#%% Initialization
files=sorted(glob.glob(os.path.join(folder,'*nc')))
progress=[]
eps=np.timedelta64(1,'s')

#%% Main
for f in files:
    with xr.open_dataset(f) as Data:
        print(os.path.basename(f))
        progress=np.append(progress,len(Data.time)*np.nanmedian(np.diff(Data.time))/(np.timedelta64(1,'D')+eps))
        if ~np.isnan(progress[-1]):
            progress_bar(progress[-1],1)
        print()
print(f'{np.sum(progress>0.99)} files completed. {np.nanmean(progress)*100:.1f}% mean progress.')
    
input('Press any key')