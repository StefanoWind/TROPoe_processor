# -*- coding: utf-8 -*-
"""
Copy and rename files
"""

import os
cd=os.getcwd()
import sys
sys.path.append(os.path.join(cd,'utils')) 
import re
import glob
import shutil

#%% Inputs
source='C:/Users/SLETIZIA/Box/wfip3_barge_campbell_raw/*dat'
destination='C:/Users/SLETIZIA/codes/TROPoe_processor/data/wfip3/barg.met.z01.00/'

regex_in='(?P<DD>\d{8})_(?P<HH>\d{4}).met.dat'
regex_out='barg.met.z01.00.{DD}.{HH}00.dat'

#%% Initialization
files=sorted(glob.glob(source))

#%% Main
for f in files:
    match = re.search(regex_in, f)
    new_name=regex_out
    for group_name in match.groupdict():
        new_name = new_name.replace('{'+group_name+'}',match.group(group_name))
    shutil.copy(f,os.path.join(destination,new_name))
    