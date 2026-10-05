"""Query-matched derivative-free searches over the SAME feasible domain.

Called after audit(): step zero uses its near_rows candidate and labels exactly.
Both baseline methods keep that incumbent. Every probability call counts toward
budget, including invalid proposals evaluated for matched compute. Neither method
claims to be the strongest possible black-box optimizer.
"""
import time
import numpy as np
from .engine import AuditConfig, FeatureDomain, _TorchAdapter, candidate_metrics


def compare_searches(report, X, y, target, config, adapter=None):
    domain=FeatureDomain.fit(X,config,report.get('feature_names'))
    model=adapter or _TorchAdapter(config,domain)
    ctx=domain.standardize(X); xt=domain.standardize(np.asarray(target).reshape(1,-1))
    init=next(a for a in report['attacks'] if a['method']=='near_rows')
    grad=next(a for a in report['attacks'] if a['method']=='gradient')
    seeds=np.asarray(X)[init['seed_indices']]
    startrows=np.asarray(init['rows'],dtype=np.float32)
    labels=np.asarray(init['labels'],dtype=int)
    before=report['baseline']['p_positive']; objective=1-int(before>=.5)
    # Equal number of new target-model calls. Gradient forward+backward counts as
    # one query, and every hard verification counts as another; reused step0 = 0.
    budget=grad['evaluations']
    for j,method in enumerate(('random_search','coordinate_search')):
        rng=np.random.default_rng(config.seed+100+j)
        best=startrows.copy(); bp=init['p_after']; valid=init['valid']
        history=[{'step':0,'p_after':bp,'valid':valid,'incumbent':valid}]
        t=time.time()
        for step in range(1,budget+1):
            if method=='random_search':
                raw=startrows+rng.normal(size=startrows.shape)*domain.std*rng.choice([.1,.3,.8])
            else:
                raw=best.astype(float).copy()
                mutable=[i for i in range(raw.shape[1]) if i not in config.protected_indices]
                if mutable:
                    col=int(rng.choice(mutable)); row=int(rng.integers(len(raw)))
                    raw[row,col]+=rng.choice([-1,1])*rng.choice([.1,.3,.8])*domain.std[col]
            cand=domain.project(raw,seeds)
            p=float(model.probability(np.concatenate([ctx,domain.standardize(cand)]),
                      np.concatenate([y,labels]),xt))
            metrics=candidate_metrics(domain,cand,target,seeds)
            score=p if objective else 1-p; bscore=bp if objective else 1-bp
            improved=bool(metrics['valid'] and (not valid or score>bscore))
            if improved: best,bp,valid=cand,p,True
            history.append({'step':step,'p_after':p,'valid':metrics['valid'],'incumbent':improved})
        metrics=candidate_metrics(domain,best,target,seeds)
        report['attacks'].append({'method':method,'n_rows':len(best),'rows':best.tolist(),
            'labels':labels.tolist(),'p_before':before,'p_after':bp,'valid':bool(valid),
            'flipped':bool(valid and (bp>=.5)!=(before>=.5)),
            'prediction_changed':bool((bp>=.5)!=(before>=.5)),
            'constraints':metrics,'optimization_history':history,'evaluations':budget,
            'elapsed_s':time.time()-t,'masked_gradient_entries':0,
            'initialization_evaluation_reused':True,'seed_indices':init['seed_indices']})
    report['diagnostics']['search_budget_note']='Forward+backward and hard verification each count as a query; compare wall time separately.'
    return report
