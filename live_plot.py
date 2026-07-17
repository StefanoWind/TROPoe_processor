'''
Render the latest TROPoe retrieval figure for a site without conflicting with an in-progress run
'''
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
cd=os.path.dirname(os.path.abspath(__file__))
import sys
from utils import tropoe_utils as trp
import xarray as xr
import glob
import yaml
from datetime import datetime, timezone
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300

#%% Inputs
if len(sys.argv)==1:
    site='s40_rt'
    source_config=os.path.join(cd,'configs/config_corsair.yaml')
    odir=os.path.join(cd,'dashboard')
else:
    site=sys.argv[1]
    source_config=sys.argv[2]
    odir=sys.argv[3]

with open(source_config,'r') as fid:
    config=yaml.safe_load(fid)

os.makedirs(odir,exist_ok=True)

#%% Main

#today's output file grows throughout the day, so it is the only one worth rendering live
date=datetime.now(timezone.utc).strftime('%Y%m%d')
files=sorted(glob.glob(os.path.join(config['output_dir'][site],'*'+date+'*nc')))

if len(files)==0:
    print('No output file found for '+site+' on '+date)
else:
    file_tropoe=files[-1]

    #read-only, non-locking, closed immediately: safe to run alongside the TROPoe container appending to the same file
    with xr.open_dataset(file_tropoe) as Data:
        trp.plot_temp_wvmr(Data,config,file_tropoe)
        plt.savefig(os.path.join(odir,site+'_latest.png'))
        plt.close()

    print('Updated '+os.path.join(odir,site+'_latest.png')+' from '+file_tropoe)
