"""Independent verification of duplicate sensitivity; no gradients or GPU.

Original pilot reported extreme exact-copy results. Keep clean accuracy, repetition,
benign-label controls and decision margins alongside the flip count.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from fz.data import load, split


def run(dataset, seed, n_context, n_targets):
    X,y=load(dataset,n_max=3000,seed=seed)
    tr,te,yr,ye=split(X,y,seed=seed)
    rng=np.random.default_rng(seed)
    sel=rng.choice(len(yr),n_context,replace=False)
    tr,yr=tr[sel],yr[sel]
    ts=rng.choice(len(ye),min(n_targets,len(ye)),replace=False)
    outputs=[]
    for name, factory in (
        ('hgb',lambda:HistGradientBoostingClassifier(random_state=seed)),
        ('rf',lambda:RandomForestClassifier(n_estimators=100,n_jobs=1,random_state=seed)),
    ):
        original=factory().fit(tr,yr)
        probs=original.predict_proba(te)[:,1]
        records=[]
        for i in ts:
            p=float(probs[i]); c=int(p>0.5)
            xt=te[i:i+1]
            values={}
            for label_key, lab in [('opposite',1-c),('same',c)]:
                re=factory().fit(np.concatenate([tr,xt]),np.concatenate([yr,[lab]]))
                values[label_key]=float(re.predict_proba(xt)[0,1])
            records.append({'target':int(i),'truth':int(ye[i]),'p_before':p,
                            **values,'flipped':int(values['opposite']>0.5)!=c})
        repeat=factory().fit(tr,yr).predict_proba(te)[:,1]
        outputs.append({'dataset':dataset,'seed':seed,'model':name,
                        'clean_auc':roc_auc_score(ye,probs),
                        'clean_accuracy':accuracy_score(ye,probs>0.5),
                        'repeat_max_delta':float(np.abs(probs-repeat).max()),
                        'n_targets':len(ts),'flips':sum(r['flipped'] for r in records),
                        'records':records})
    return outputs


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', default='results/controls.json')
    a=parser.parse_args()
    result=[]
    for d in ('credit-g','diabetes','taiwan'):
        result+=run(d,seed=101,n_context=300,n_targets=12)
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2))
    for r in result:
        print(r['dataset'],r['model'],'auc',round(r['clean_auc'],3),
              'flips',r['flips'],'/',r['n_targets'],'repeat delta',r['repeat_max_delta'])
