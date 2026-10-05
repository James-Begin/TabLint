"""Descriptive development counts (NOT a hypothesis-test or headline benchmark)."""
import json
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
counts=defaultdict(lambda:{'reports':0,'valid':0,'surrogate_flips':0,'gradient_aborts':0,
                         'standard_verified_flips':0,'standard_candidates_checked':0})
for path in sorted((ROOT/'results'/'development').glob('*.json')):
    r=json.loads(path.read_text())
    if not isinstance(r,dict) or r.get('schema_version')!='1.0':continue
    ds=r['dataset']['dataset']
    for a in r['attacks']:
        key=(ds,a['method']);c=counts[key];c['reports']+=1
        c['valid']+=int(a['valid']); c['surrogate_flips']+=int(a['flipped'])
        c['gradient_aborts']+=int(a.get('aborted',False))
        v=next((v for v in r.get('verification',[]) if v['model']=='tabpfn_standard'
                and v['attack_method']==a['method']),None)
        if v:
            c['standard_candidates_checked']+=1
            c['standard_verified_flips']+=int(v.get('candidate_valid',False) and v.get('flipped',False))
out=[{'dataset':ds,'method':method,**c} for (ds,method),c in sorted(counts.items())]
(ROOT/'results'/'development'/'summary_counts.json').write_text(json.dumps(out,indent=2))
for r in out:
    print(r['dataset'],r['method'],'valid',r['valid'],'/',r['reports'],
          'surrogate flips',r['surrogate_flips'],'verified standard flips',r['standard_verified_flips'],
          'gradient aborts',r['gradient_aborts'])
