"""Attach exact LOO for a replay artifact, verifying context reconstruction first."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(k,'1')
import hashlib
import json
from pathlib import Path
import numpy as np
from auditkit.cli import dataset
from auditkit.influence import leave_one_out
if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('explicit GPU required')
p=Path('results/reports/breast_101_0.json');r=json.loads(p.read_text())
X,y,m=dataset(r['dataset']['dataset']);ids=np.asarray(r['experiment']['context_ids'])
Xc,yc=X[ids],y[ids]
assert hashlib.sha256(Xc.tobytes()+yc.tobytes()).hexdigest()==r['experiment']['context_sha256']
target=X[r['experiment']['target_id']:r['experiment']['target_id']+1]
r['influence']=leave_one_out(Xc,yc,target,device='cuda:0',seed=r['config']['seed'])
p.write_text(json.dumps(r,indent=2,allow_nan=False))
print('LOO rows',len(r['influence']['rows']),'seconds',round(r['influence']['elapsed_s'],1))
