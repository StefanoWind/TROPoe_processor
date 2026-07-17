'''
Copy rendered dashboard figures into a public GitHub Pages repo checkout and push them.

The target repo_dir must already be a git clone of a public repo with GitHub Pages enabled,
with push access configured (SSH deploy key or PAT) for the user running this script.
'''
import os
import sys
import glob
import shutil
import subprocess
from datetime import datetime, timezone

#%% Inputs
if len(sys.argv)==1:
    src_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'dashboard')
    repo_dir='/path/to/public-dashboard-repo'
else:
    src_dir=sys.argv[1]
    repo_dir=sys.argv[2]

#%% Main
pngs=sorted(glob.glob(os.path.join(src_dir,'*_latest.png')))
for f in pngs:
    shutil.copy(f,repo_dir)

timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
cache_bust=str(int(datetime.now(timezone.utc).timestamp()))

html=['<html><head>',
      '<meta http-equiv="refresh" content="300">',
      '<title>TROPoe live dashboard</title>',
      '<style>body{font-family:sans-serif;margin:2em} img{max-width:100%;display:block;margin-bottom:2em}</style>',
      '</head><body>',
      '<p>Last updated: '+timestamp+'</p>']
for f in pngs:
    name=os.path.basename(f)
    html.append('<h2>'+name.replace('_latest.png','')+'</h2>')
    html.append('<img src="'+name+'?t='+cache_bust+'">')
html.append('</body></html>')

with open(os.path.join(repo_dir,'index.html'),'w') as fid:
    fid.write('\n'.join(html))

subprocess.run('git add -A && git commit -m "update dashboard '+timestamp+'" && git push',cwd=repo_dir,shell=True)
