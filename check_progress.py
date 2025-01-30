# -*- coding: utf-8 -*-
"""
Check progress in TROPoe files
"""
import os
cd=os.getcwd()
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
files=glob.glob(os.path.join(folder,'*nc'))


#%% Main
for f in files:
    Data=xr.open_dataset(f)
    
    print(os.path.basename(f))
    progress_bar(len(Data.time),np.timedelta64(1,'D')/np.nanmedian(np.diff(Data.time)))
    print()
    
input('Press any key')