"""
Fix/overwrite specific global attributes on all .nc files in a folder.

Usage: python fix_attributes.py <folder>
"""
import os
import sys
import glob
import netCDF4 as nc

#%% Inputs
#global attributes to set (edit as needed)
ATTRS_TO_SET = {
    'Site': 'Cape Cod, MS',
}

if len (sys.argv)==1:
    folder='data/wfip3/caco.assist.tropoe.z01.c1'
else:
    folder = sys.argv[1]

#%% Initialization
files = sorted(glob.glob(os.path.join(folder, '*.nc')))

#%% Main
for f in files:
    with nc.Dataset(f, 'r+') as ds:
        for attr_name, attr_value in ATTRS_TO_SET.items():
            ds.setncattr(attr_name, attr_value)
    print(f'Updated {os.path.basename(f)}')
