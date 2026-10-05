"""Reproduce original development gradient failures and check a scoped backward fix.

No default changes. Exact original context hashes must match. Forward equivalence,
finite-gradient recovery, and optimizer progress are separate endpoints.
"""
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(v,'2')
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from auditkit.cli import dataset
from auditkit.engine import AuditConfig,FeatureDomain,_TorchAdapter,audit
from experiments.numerical_probe import experimental_std_backward,scaler_module,outlier_module

if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('explicit GPU required')
root=Path('results/development')
examples=['credit-g_101_0.json','credit-g_102_0.json','taiwan_101_0.json','taiwan_102_0.json']
results=[]
t0=time.time()
original_scaler=scaler_module.torch_nanstd;original_outlier=outlier_module.torch_nanstd
for filename in examples:
    r=json.loads((root/filename).read_text())
    X,y,meta=dataset(r['dataset']['dataset'])
    ids=np.asarray(r['experiment']['context_ids']);Xc,yc=X[ids],y[ids]
    assert hashlib.sha256(Xc.tobytes()+yc.tobytes()).hexdigest()==r['experiment']['context_sha256']
    target=X[r['experiment']['target_id']:r['experiment']['target_id']+1]
    c=AuditConfig(**r['config']);domain=FeatureDomain.fit(Xc,c,r['feature_names'])
    near=next(a for a in r['attacks'] if a['method']=='near_rows')
    rows=np.asarray(near['rows'],np.float32);labels=np.asarray(near['labels'],int)
    ctx=domain.standardize(Xc);xt=domain.standardize(target)
    inp=np.concatenate([ctx,domain.standardize(rows)]);yy=np.concatenate([yc,labels])
    model=_TorchAdapter(c,domain);model.probability(ctx,yc,xt)
    objective=1-r['baseline']['predicted_class']
    p0,g0=model.value_and_grad(inp,yy,xt,len(rows),objective)
    with experimental_std_backward():
        p1,g1=model.value_and_grad(inp,yy,xt,len(rows),objective)
        newer=audit(Xc,yc,target,r['feature_names'],c)
    rec={'artifact':filename,'context_hash_matched':True,
         'original_p':p0,'patched_p':p1,'abs_forward_difference':abs(p1-p0),
         'forward_equivalent_1e7':abs(p1-p0)<=1e-7,
         'original_nonfinite_gradients':int((~np.isfinite(g0)).sum()),
         'patched_nonfinite_gradients':int((~np.isfinite(g1)).sum()),
         'patched_audit_gradient':next(a for a in newer['attacks'] if a['method']=='gradient')}
    results.append(rec)
    print(filename,'original nonfinite',rec['original_nonfinite_gradients'],
          'patched',rec['patched_nonfinite_gradients'],'forward delta',abs(p1-p0),
          'optimizer aborted',rec['patched_audit_gradient']['aborted'],flush=True)
    assert scaler_module.torch_nanstd is original_scaler and outlier_module.torch_nanstd is original_outlier
out=Path('results/numerical_recovery');out.mkdir(exist_ok=True)
(out/'results.json').write_text(json.dumps({'cases':results,'seconds':time.time()-t0,
                               'default_engine_changed':False,'bindings_restored':True},indent=2,allow_nan=False))
