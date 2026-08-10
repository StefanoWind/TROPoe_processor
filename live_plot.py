'''
Render TROPoe retrieval figures for every day found in a site's output folder,
merging same-day chunk files together before plotting
'''
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
cd=os.path.dirname(os.path.abspath(__file__))
import sys
from utils import tropoe_utils as trp
import xarray as xr
import glob
import re
import yaml
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300

#%% Inputs
if len(sys.argv)==1:
    site='sa_rt'
    source_config=os.path.join(cd,'configs/config_anvil.yaml')
else:
    site=sys.argv[1]
    source_config=sys.argv[2]

with open(source_config,'r') as fid:
    config=yaml.safe_load(fid)

#%% Main

#group every output file in the folder by the date embedded in its name (rootname.YYYYMMDD.HHMMSS.nc),
#since a chunked day is split across multiple files
files=sorted(glob.glob(os.path.join(config['output_dir'][site],'*.nc')))

dates={}
for f in files:
    match=re.search(r'\.(\d{8})\.\d{6}\.nc$',os.path.basename(f))
    if match:
        dates.setdefault(match.group(1),[]).append(f)

if len(dates)==0:
    print('No output files found for '+site)
else:
    for date,day_files in sorted(dates.items()):

        #read-only, non-locking, closed immediately: safe to run alongside a TROPoe container still appending to today's chunk
        datasets=[xr.open_dataset(f) for f in day_files]
        Data=xr.concat(datasets,dim='time',data_vars='minimal',coords='minimal',compat='override').sortby('time')
        Data=Data.drop_duplicates('time',keep='first')

        save_name=f'{".".join(day_files[0].split(".")[:-2])}.000000_merged.nc'
        
        Data.to_netcdf(save_name)
        trp.plot_temp_wvmr(Data,config,day_files[-1].split())
        plt.savefig(save_name.replace('.nc','_T_r.png'))
        plt.close()

        for ds in datasets:
            ds.close()

        print('Updated '+day_files[0].replace('.nc','_T_r.png')+' from '+', '.join(os.path.basename(f) for f in day_files))
