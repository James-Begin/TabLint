"""Exact leave-one-out sensitivity on a specified receiver (not causal attribution).

Fingerprint-disabled estimator explicitly named; removing a row still refits
preprocessing, so effects are pipeline-level. An intact-context repeat and shuffled
context check are included. No inference of fairness/causality from row influence.
"""
import time
import numpy as np


def leave_one_out(X,y,target,device='cpu',seed=0,limit=None):
    from fz.core import make_std_clf
    X=np.asarray(X,np.float32); y=np.asarray(y,int)
    target=np.asarray(target,np.float32).reshape(1,-1)
    if not np.isfinite(X).all() or not np.isfinite(target).all() or set(y.tolist())!={0,1}:
        raise ValueError('finite numeric binary data required')
    def prob(rows,labels):
        clf=make_std_clf(n_estimators=1,seed=seed,device=device,
                        inference_config={'FINGERPRINT_FEATURE':False})
        return float(clf.fit(rows,labels).predict_proba(target)[0,1])
    t=time.time();base=prob(X,y);repeat=prob(X,y)
    order=np.random.default_rng(seed).permutation(len(y));permuted=prob(X[order],y[order])
    subset=np.arange(len(y)) if limit is None else np.arange(min(limit,len(y)))
    records=[]
    for i in subset:
        keep=np.arange(len(y))!=i
        if len(np.unique(y[keep]))<2:
            records.append({'row':int(i),'status':'would_remove_last_class_example'});continue
        p=prob(X[keep],y[keep])
        records.append({'row':int(i),'p_without':p,'delta_p_positive':p-base,
                        'training_label':int(y[i])})
    return {'model':'tabpfn_single_nofingerprint','p_before':base,'repeat_delta':abs(repeat-base),
        'permutation_delta':abs(permuted-base),'rows':records,'elapsed_s':time.time()-t,
        'exact_over_all_context_rows':limit is None or limit>=len(y),
        'caveat':'Removal refits preprocessing; these are pipeline sensitivities, not causal responsibility.'}
