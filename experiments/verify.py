"""Verification adapters with fixed receiver settings and explicit refit controls.

No receiver is silently changed to match the differentiable surrogate. Fingerprint
on/off are separate configurations. Every attack is re-evaluated on its recorded
labels. Never re-label it toward a receiver's clean prediction after creation.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import time
import numpy as np


def factory(name, seed=0, device='cpu'):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if name=='hgb':
        return HistGradientBoostingClassifier(random_state=seed)
    if name=='hgb_regularized':
        return HistGradientBoostingClassifier(random_state=seed,max_leaf_nodes=7,
                 l2_regularization=10.0,min_samples_leaf=30,max_iter=100)
    if name=='rf':
        return RandomForestClassifier(n_estimators=100,n_jobs=1,random_state=seed)
    if name=='rf_regularized':
        return RandomForestClassifier(n_estimators=100,n_jobs=1,min_samples_leaf=10,
                                       max_depth=6,random_state=seed)
    if name=='logistic':
        return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    if name.startswith('tabpfn_'):
        from fz.core import make_std_clf
        n=1 if 'single' in name else 4
        # 'standard' uses stock fingerprint policy; 'nofingerprint' explicit ablation
        kw={'inference_config':{'FINGERPRINT_FEATURE':False}} if 'nofingerprint' in name else {}
        return make_std_clf(n_estimators=n,seed=seed,device=device,**kw)
    raise ValueError(name)


def verify_report(report,X,y,target,names,seed=0,device='cpu'):
    """Same original context/candidates for all receiver models; no target exclusions."""
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=int)
    target=np.asarray(target,dtype=np.float32).reshape(1,-1)
    verification=[]; controls=[]
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(y))
    for name in names:
        t0=time.time()
        predictor=factory(name,seed,device).fit(X,y)
        before=float(predictor.predict_proba(target)[0,1])
        again=float(factory(name,seed,device).fit(X,y).predict_proba(target)[0,1])
        perm=float(factory(name,seed,device).fit(X[order],y[order]).predict_proba(target)[0,1])
        controls.append({'model':name,'p_clean':before,'repeat_delta':abs(again-before),
                         'permutation_delta':abs(perm-before),
                         'settings':predictor.get_params(deep=False).__repr__()})
        for attack in report['attacks']:
            rows=np.asarray(attack['rows'],dtype=np.float32).reshape(-1,X.shape[1])
            labels=np.asarray(attack['labels'],dtype=int)
            if len(rows)!=len(labels) or not np.isfinite(rows).all():
                verification.append({'attack_method':attack['method'],'model':name,
                                     'status':'invalid_record'})
                continue
            newx=np.concatenate([X,rows]); newy=np.concatenate([y,labels])
            p=float(factory(name,seed,device).fit(newx,newy).predict_proba(target)[0,1])
            # Original prediction repeated as benign label; only a control, not truth.
            benign=np.concatenate([y,np.full(len(rows),int(before>=0.5))])
            pb=float(factory(name,seed,device).fit(newx,benign).predict_proba(target)[0,1])
            verification.append({'attack_method':attack['method'],'model':name,
                'p_before':before,'p_after':p,'flipped':(p>=0.5)!=(before>=0.5),
                'candidate_valid':attack.get('valid',False),
                'same_direction_as_surrogate':(before>=0.5)==(report['baseline']['p_positive']>=0.5),
                'benign_label_p_after':pb,'benign_label_flip':(pb>=0.5)!=(before>=0.5),
                'controls':{'untouched_repeat':again,'max_abs_repeat_delta':abs(again-before)}})
        controls[-1]['elapsed_s']=time.time()-t0
    return verification,controls
