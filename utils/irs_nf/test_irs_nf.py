# -*- coding: utf-8 -*-
import subprocess
import os
cd = os.path.dirname(__file__)

#%% Inputs
sdate='20240514'
edate='20240515'
python_path='C:/ProgramData/Anaconda3/python.exe'
chassistdir=os.path.join(cd,'data/rhod.assist.z01.00/ch1')
sumassistdir=os.path.join(cd,'data/rhod.assist.z01.00/sum')
nfchassistdir=os.path.join(cd,'data/rhod.assist.z01.00/ch1nf')

# chassistdir='c:/users/sletizia/onedrive - nrel/desktop/postdoc/wfip3/tropoe_processor/rhod.assist.z01.00/ch1/data'
# sumassistdir='c:/users/sletizia/onedrive - nrel/desktop/postdoc/wfip3/tropoe_processor/rhod.assist.z01.00/sum/data'
# nfchassistdir='c:/users/sletizia/onedrive - nrel/desktop/postdoc/wfip3/tropoe_processor/rhod.assist.z01.00/ch1nf/data'

#%% Main

command=python_path+f' run_irs_nf.py --create --verbose {sdate} {edate} {chassistdir} {sumassistdir} {nfchassistdir} "assist"'

print(command)
result = subprocess.run(command, shell=True, text=True,capture_output=True)
print(result.stdout)
print(result.stderr)


command=python_path+f' run_irs_nf.py --apply --verbose {sdate} {edate} {chassistdir} {sumassistdir} {nfchassistdir} "assist"'

print(command)
result = subprocess.run(command, shell=True, text=True,capture_output=True)
print(result.stdout)
print(result.stderr)