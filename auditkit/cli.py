"""Generate a self-contained replay artifact with explicit GPU opt-in.

python -m auditkit.cli --dataset breast-cancer --device cuda:0 --out results/reports/demo.json
Data is public research data; no production/clinical decisions are made.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from .engine import AuditConfig, FeatureDomain, audit
from .search import compare_searches


def dataset(name):
    if name=='breast-cancer':
        from sklearn.datasets import load_breast_cancer
        data=load_breast_cancer()
        # Class 1 = malignant; sklearn's target 1 means benign.
        return data.data.astype(np.float32),(1-data.target).astype(int), {
            'names':data.feature_names.tolist(),'groups':[],'integer':[],
            'label':'malignancy label (research only)',
            'source':'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
            'license':'CC BY 4.0','dataset':name}
    from experiments.screen_regimes import OPENML
    if name in OPENML:
        # Numeric-only OpenML task (categorical columns dropped); loaded by ID, never redistributed.
        from experiments.screen_regimes import load
        X,y,_=load(name)
        return X,y,{'names':[f'x{i}' for i in range(X.shape[1])],'groups':[],'integer':[],
                    'source':'https://www.openml.org/d/'+str(OPENML[name]),
                    'license':'see the OpenML dataset page; data not redistributed here','dataset':name}
    from fz.data import load_meta
    X,y,m=load_meta(name,n_max=3000,seed=0)
    if name=='taiwan':
        # Explicit source schema, not integer inference from test rows. Payment
        # status is an encoded category, not a continuous monetary measurement.
        categoricals={1:[1,2],2:list(range(7)),3:list(range(4)),
                      **{i:list(range(-2,10)) for i in range(5,11)}}
        cols=[]; names=[]; groups=[]
        for i,colname in enumerate(m['names']):
            if i in categoricals:
                levels=categoricals[i]; start=len(names)
                cols.extend([(X[:,i]==v).astype(np.float32) for v in levels])
                names.extend([f'{colname}={v}' for v in levels])
                groups.append(list(range(start,len(names))))
            else:
                cols.append(X[:,i]); names.append(colname)
        X=np.stack(cols,axis=1); m['names']=names; m['groups']=[np.asarray(g) for g in groups]
    return X,y,{'names':m['names'],'groups':[g.tolist() for g in m['groups']],
               'integer':[],'source':'https://www.openml.org/search?type=data&id='+{
                 'credit-g':'31','taiwan':'42477'}[name],
               'license':'CC BY 4.0 (UCI source)','dataset':name}


def run(args):
    if args.device.startswith('cuda') and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to the approved physical GPU before CUDA inference.')
    from sklearn.model_selection import train_test_split
    X,y,meta=dataset(args.dataset)
    # Split deterministically before context fitting.
    ids=np.arange(len(y))
    ic,it=train_test_split(ids,test_size=.3,stratify=y,random_state=args.seed)
    rng=np.random.default_rng(args.seed)
    ic=rng.choice(ic,min(args.context,len(ic)),replace=False)
    targetid=int(it[args.target%len(it)])
    Xc,yc,xt=X[ic],y[ic],X[targetid:targetid+1]
    config=AuditConfig(seed=args.seed,steps=args.steps,k=args.k,device=args.device,
                       groups=tuple(tuple(g) for g in meta['groups']),
                       stable_backward=args.stable_backward)
    dom=FeatureDomain.fit(Xc,config,meta['names'])
    config=replace(config,min_target_distance=args.min_radius*dom.nn_radius,
                   max_real_distance=args.max_radius*dom.nn_radius if args.max_radius else None)
    t=time.time()
    report=audit(Xc,yc,xt,meta['names'],config)
    report['feature_names']=meta['names']; report['target']=xt[0].tolist()
    report['context_preview']=Xc[:8].tolist(); report['dataset']=meta
    report['experiment']={'seed':args.seed,'target_id':targetid,'truth':int(y[targetid]),
        'context_sha256':hashlib.sha256(Xc.tobytes()+yc.tobytes()).hexdigest(),
        'target_sha256':hashlib.sha256(xt.tobytes()).hexdigest(),
        'context_ids':ic.tolist(),'min_radius_factor':args.min_radius,
        'max_radius_factor':args.max_radius,'versions':{p:importlib.metadata.version(p)
          for p in ('tabpfn','torch','numpy','scikit-learn')}}
    if args.compare_search:
        compare_searches(report,Xc,yc,xt,config)
    if args.verify:
        from experiments.verify import verify_report
        names=args.verify.split(',')
        report['verification'],report['receiver_controls']=verify_report(
                report,Xc,yc,xt,names,args.seed,args.device)
        report['warnings']=[w for w in report['warnings'] if 'verification is out of scope' not in w]
        report['warnings'].append('Surrogate and deployed estimator have different preprocessing; only listed receiver checks are verified.')
    report['elapsed_total_s']=time.time()-t
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(out,'elapsed',round(report['elapsed_total_s'],1))
    for a in report['attacks']:
        print(a['method'],'valid',a['valid'],'flipped',a['flipped'],'p',round(a['p_after'],4),
              'queries',a['evaluations'])
    return report


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='breast-cancer',help='breast-cancer, credit-g, taiwan, or a regime-screen name (eeg-eye-state, mozilla4, phoneme, MagicTelescope)')
    p.add_argument('--context',type=int,default=200)
    p.add_argument('--target',type=int,default=0,help='index in test split, never selected using outcome')
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--k',type=int,default=3)
    p.add_argument('--device',default='cpu')
    p.add_argument('--min-radius',type=float,default=.5)
    p.add_argument('--max-radius',type=float,default=1.5)
    p.add_argument('--compare-search',action='store_true')
    p.add_argument('--stable-backward',action='store_true',
                   help='experimental: forward-preserving backward fix for TabPFN preprocessing')
    p.add_argument('--verify',default='hgb,hgb_regularized,rf,rf_regularized,logistic,tabpfn_standard,tabpfn_nofingerprint')
    p.add_argument('--out',default='results/reports/audit.json')
    return p


if __name__=='__main__':
    run(parser().parse_args())
