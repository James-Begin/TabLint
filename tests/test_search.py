import json
import numpy as np
from auditkit import AuditConfig,audit
from auditkit.search import compare_searches
from test_auditkit_engine import FakeAdapter


def test_search_budget_and_incumbent():
    X=np.array([[0.,10.],[2.,20.],[4.,30.],[6.,40.]],np.float32)
    y=np.array([0,0,1,1]);target=np.array([1.,15.])
    config=AuditConfig(k=1,steps=3,lr=.1,seed=12)
    model=FakeAdapter(len(y),gradient=-1)
    r=audit(X,y,target,config=config,adapter=model)
    compare_searches(r,X,y,target,config,adapter=model)
    near=next(a for a in r['attacks'] if a['method']=='near_rows')
    grad=next(a for a in r['attacks'] if a['method']=='gradient')
    for name in ('random_search','coordinate_search'):
        a=next(a for a in r['attacks'] if a['method']==name)
        assert a['evaluations']==grad['evaluations']
        assert a['optimization_history'][0]['p_after']==near['p_after']
        assert a['valid'] and a['p_after']<=near['p_after']
    json.dumps(r,allow_nan=False)
