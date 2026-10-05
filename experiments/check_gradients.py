"""Finite-difference check on the actual standardized, frozen-parameter adapter.

Use interior numeric rows so projection/boundary non-differentiability is not mixed
into this diagnostic. This tests implementation, not the validity of synthetic data.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(k,'1')
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from auditkit.engine import AuditConfig,FeatureDomain,_TorchAdapter

if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('explicit GPU required')
d=load_breast_cancer();X=d.data[:160].astype(np.float32);y=1-d.target[:160]
c=AuditConfig(seed=101,device='cuda:0',k=1)
domain=FeatureDomain.fit(X,c)
m=_TorchAdapter(c,domain)
z=domain.standardize(X);target=domain.standardize(d.data[200:201])
# Avoid duplicating a context row: TabPFN-3.5's ECDF channel has rank
# discontinuities at exact ties, where ordinary central differences are invalid.
row=.63*z[40:41]+.37*z[45:46];combined=np.concatenate([z,row]);labels=np.concatenate([y,[0]])
m.probability(z,y,target)
p,gradient=m.value_and_grad(combined,labels,target,1,0)
records=[]
for col in (0,3,6,12,18,24):
    a,b=combined.copy(),combined.copy(); eps=.0002
    a[-1,col]+=eps;b[-1,col]-=eps
    # Adapter objective = P(class0).
    fd=(m.probability(b,labels,target)-m.probability(a,labels,target))/(2*eps)
    records.append({'column':col,'autograd':float(gradient[0,col]),'finite_difference':float(fd),
                    'abs_error':float(abs(fd-gradient[0,col]))})
out={'records':records,'epsilon_standardized':.0002,'parameter_elements_frozen':m.frozen_parameters,
     'all_gradients_finite':bool(np.isfinite(gradient).all()),
     'max_abs_error':max(r['abs_error'] for r in records)}
p=Path('results/gradient_check.json');p.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
