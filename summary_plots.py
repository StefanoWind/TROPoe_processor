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
import yaml
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300

#%% Inputs
if len(sys.argv)==1:
    site='mvco'
    source_config=os.path.join(cd,'configs/config_wfip3_c1.yaml')
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

    Data=xr.open_dataset(f)

    fig=trp.plot_temp_wvmr(Data,config)
    fig.savefig(f.replace('.nc','_T_r.png'))
    plt.close()
    print(f'{os.path.basename(f)} done.')
    