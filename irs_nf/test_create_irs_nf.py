# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import subprocess
import os
import logging
import logging.config
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np

from irs_nf import irs_nf, utils


cd = os.path.dirname(__file__)

#%% Inputs
sdate='20240514'
edate='20240515'
python_path='C:/ProgramData/Anaconda3/python.exe'
chassistdir=os.path.join(cd,'data/rhod.assist.z01.00/ch1')
sdir=os.path.join(cd,'data/rhod.assist.z01.00/sum')
odir=os.path.join(cd,'data/rhod.assist.z01.00/ch1nf')

sky_view_angle = 0

sfields = {
    'dmv2cdf': [['wnum1', 'SkyNENch1'], ['wnum2', 'SkyNENch2']],
    'assist': [['wnumsum1', 'SkyNENCh1'], ['wnumsum2', 'SkyNENCh2']],
    'arm': [['wnumsum5', 'SkyNENCh1'], ['wnumsum6', 'SkyNENCh2']]
}

#%% Initialization
ch1_sfield = sfields['assist'][0]

pcs_filename = os.path.join(odir, 'irs_nf_ch1.pkl')

tdir = os.path.join(odir, 'tmp')
    
#%% Inputs
rfilenames=[os.path.join(chassistdir,'assistChA.20240514.000103.cdf')]

irs_nf.create_irs_noise_filter(rfilenames, sdir, ch1_sfield,tdir,
                                pcs_filename=pcs_filename, sky_view_angle=sky_view_angle,use_median_noise=False)
