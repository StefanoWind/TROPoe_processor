# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import glob
import numpy as np

import irs_nf, utils


#%% Inputs
start_date='20240902'
end_date='20240909'
python_path='C:/ProgramData/Anaconda3/python.exe'
cd='C:/Users/SLETIZIA/codes/TROPoe_processor/'
idir=os.path.join(cd,'data/wfip3/barg.assist.z01.00/ch1')
sdir=os.path.join(cd,'data/wfip3/barg.assist.z01.00/sum')
odir=os.path.join(cd,'data/wfip3/barg.assist.z01.00/nfc')
tdir=None
irs_type='assist'

sky_view_angle = 0

sfields = {
    'dmv2cdf': [['wnum1', 'SkyNENch1'], ['wnum2', 'SkyNENch2']],
    'assist': [['wnumsum1', 'SkyNENCh1'], ['wnumsum2', 'SkyNENCh2']],
    'arm': [['wnumsum5', 'SkyNENCh1'], ['wnumsum6', 'SkyNENCh2']]
}

#%% Initialization
start = datetime.strptime(start_date, "%Y%m%d")
end = datetime.strptime(end_date, "%Y%m%d")

ch1_sfield = sfields[irs_type][0]
if irs_type == "assist":
    sky_view_angle = 0
else:
    sky_view_angle = None
    
if not os.path.exists(odir):
    os.makedirs(odir)
    
# Make a temp directory if needed
if tdir is None:
    tdir = os.path.join(odir, 'tmp')

if not os.path.exists(tdir):
    os.makedirs(tdir)
    
# Get the files needed for channel 1
os.chdir(idir)
files, err = utils.findfile(idir, "*(ch1|cha|ChA)*.(nc|cdf)")
rdates = np.array([datetime.strptime(d.split('.')[-3], '%Y%m%d') for d in files])
foo = np.where((rdates >= start) & (rdates <= end))

rfilenames = np.array(files)[foo]
    
#%% Inputs

if len(rfilenames) >= 1:
    pcs_filename = os.path.join(odir, 'irs_nf_ch1.pkl')
    
    
    irs_nf.create_irs_noise_filter(rfilenames, sdir, ch1_sfield,tdir,
                                       pcs_filename=pcs_filename, sky_view_angle=sky_view_angle,use_median_noise=False)

    irs_nf.apply_irs_noise_filter(rfilenames, sdir, ch1_sfield, tdir, odir, 
                                  pcs_filename=pcs_filename,sky_view_angle=sky_view_angle,use_median_noise=False)