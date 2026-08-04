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
    odir=os.path.join(cd,'dashboard')
else:
    site=sys.argv[1]
    source_config=sys.argv[2]
    odir=sys.argv[3]

with open(source_config,'r') as fid:
    config=yaml.safe_load(fid)

os.makedirs(odir,exist_ok=True)

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
        Data=xr.concat(datasets,dim='time').sortby('time')

        trp.plot_temp_wvmr(Data,config,day_files[-1])
        plt.savefig(os.path.join(odir,site+'_'+date+'.png'))
        plt.close()

        for ds in datasets:
            ds.close()

        print('Updated '+os.path.join(odir,site+'_'+date+'.png')+' from '+', '.join(os.path.basename(f) for f in day_files))
