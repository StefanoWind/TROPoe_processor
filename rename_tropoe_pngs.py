# -*- coding: utf-8 -*-
"""
Rename TROPoe QC plot files to insert the missing ".tropoe" tag, e.g.
mvco.assist.z01.c1.20240301.000111_tropoe_inputs.png
-> mvco.assist.tropoe.z01.c1.20240301.000111_tropoe_inputs.png
"""

import os
cd = os.getcwd()
import glob
import numpy as np

#%% Inputs
source = input('Folder: ')

#%% Initialization
files = sorted(glob.glob(os.path.join(source,'*png')))
ctr = 0

#%% Main
for f in files:
    folder, name = os.path.split(f)
    if '.assist.tropoe.' in name:
        ctr += 1
        continue
    new_name = name.replace('.assist.', '.assist.tropoe.', 1)
    os.rename(f, os.path.join(folder, new_name))

    print(str(np.round(ctr / len(files) * 100, 2)) + '% done')
    ctr += 1
