"""Forward equivalence and gradient finiteness of the opt-in stable backward.

For each dataset x context seed x target x probe type, evaluate the differentiable
surrogate with and without auditkit.stability.stable_backward on IDENTICAL inputs.
Endpoints: max |dP| (forward equivalence), fraction of finite gradients (both modes),
and gradient agreement where both are finite. Serial, one visible GPU.
"""
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(v,'2')
import json
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import train_test_split
from auditkit.cli import dataset
from auditkit.engine import AuditConfig,FeatureDomain,_TorchAdapter

if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('explicit GPU required')
records=[];t0=time.time()
for name in ('breast-cancer','credit-g','taiwan'):
    X,y,meta=dataset(name)
    for seed in (201,202,203):
        ids=np.arange(len(y))
        ic,it=train_test_split(ids,test_size=.3,stratify=y,random_state=seed)
        rng=np.random.default_rng(seed)
        ic=rng.choice(ic,200,replace=False)
        Xc,yc=X[ic],y[ic]
        cfg=AuditConfig(seed=seed,device='cuda:0',k=3,groups=tuple(tuple(g) for g in meta['groups']))
        dom=FeatureDomain.fit(Xc,cfg,meta['names'])
        a_off=_TorchAdapter(cfg,dom)
        cfg_on=AuditConfig(seed=seed,device='cuda:0',k=3,groups=cfg.groups,stable_backward=True)
        a_on=_TorchAdapter(cfg_on,dom)
        ctx=dom.standardize(Xc)
        for t in range(3):
            tid=int(it[t]);xt=dom.standardize(X[tid:tid+1])
            near=np.argsort(np.linalg.norm(ctx-xt,axis=1))[:3]
            probes={'real_rows':Xc[near],
                    'perturbed':dom.project(Xc[near]+rng.normal(size=(3,Xc.shape[1]))*dom.std*.3,Xc[near]),
                    'duplicates':np.repeat(X[tid:tid+1],3,0)}
            p_clean_off=a_off.probability(ctx,yc,xt);p_clean_on=a_on.probability(ctx,yc,xt)
            objective=1-int(p_clean_off>=.5)
            for probe,rows in probes.items():
                inp=np.concatenate([ctx,dom.standardize(rows)])
                lab=np.concatenate([yc,np.full(3,objective)])
                p0,g0=a_off.value_and_grad(inp,lab,xt,3,objective)
                p1,g1=a_on.value_and_grad(inp,lab,xt,3,objective)
                f0,f1=np.isfinite(g0).all(),np.isfinite(g1).all()
                both=np.isfinite(g0)&np.isfinite(g1)
                cos=None
                if f0 and f1 and np.linalg.norm(g0)>0 and np.linalg.norm(g1)>0:
                    cos=float((g0*g1).sum()/np.linalg.norm(g0)/np.linalg.norm(g1))
                records.append({'dataset':name,'seed':seed,'target':tid,'probe':probe,
                    'clean_forward_delta':abs(p_clean_on-p_clean_off),
                    'forward_delta':abs(p1-p0),'finite_off':bool(f0),'finite_on':bool(f1),
                    'nonfinite_entries_off':int((~np.isfinite(g0)).sum()),
                    'max_grad_delta_where_both_finite':float(np.abs(g0[both]-g1[both]).max()) if both.any() else None,
                    'cosine_if_both_finite':cos})
        print(name,seed,'done',round(time.time()-t0),flush=True)
summary={}
for name in ('breast-cancer','credit-g','taiwan'):
    r=[x for x in records if x['dataset']==name]
    summary[name]={'cases':len(r),
        'max_forward_delta':max(max(x['forward_delta'],x['clean_forward_delta']) for x in r),
        'finite_without_patch':sum(x['finite_off'] for x in r),
        'finite_with_patch':sum(x['finite_on'] for x in r),
        'min_cosine_when_both_finite':min([x['cosine_if_both_finite'] for x in r if x['cosine_if_both_finite'] is not None],default=None)}
out=Path('results/stability_check.json')
out.write_text(json.dumps({'summary':summary,'records':records,'seconds':time.time()-t0},indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))
