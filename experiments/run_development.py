"""Bounded multi-context development benchmark, not final confirmation.

One sequential child process per selected physical GPU, never a growing producer.
Each target report includes actual attack rows and receiver checks. No exclusions
for surrogate/receiver prediction disagreement. Failures are recorded separately.

Usage: python experiments/run_development.py --gpus 0,1,2,3,4,5,6,7 --targets 4
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
STABLE=False
STEPS=12
OUTDIR='development'


def worker(item):
    gpu,jobs=item
    out=[]
    for ds,seed,target in jobs:
        p=ROOT/'results'/OUTDIR/f'{ds}_{seed}_{target}.json'
        p.parent.mkdir(parents=True,exist_ok=True)
        log=p.with_suffix('.log')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(ROOT)}
        cmd=[sys.executable,'-m','auditkit.cli','--dataset',ds,'--seed',str(seed),
             '--target',str(target),'--context','200','--steps',str(STEPS),'--k','3',
             '--compare-search','--device','cuda:0','--out',str(p)]+(
                 ['--stable-backward'] if STABLE else [])
        with log.open('w') as f:
            try:
                result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=f,timeout=240)
                record={'dataset':ds,'seed':seed,'target':target,'gpu':gpu,'exit':result.returncode,
                        'artifact':str(p.relative_to(ROOT))}
            except subprocess.TimeoutExpired:
                record={'dataset':ds,'seed':seed,'target':target,'gpu':gpu,'exit':'timeout'}
        out.append(record)
        print(record,flush=True)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpus',default='4')
    ap.add_argument('--targets',type=int,default=4)
    ap.add_argument('--datasets',default='breast-cancer,credit-g,taiwan')
    ap.add_argument('--stable-backward',action='store_true')
    ap.add_argument('--outdir',default='development')
    ap.add_argument('--seeds',default='101,102,103')
    ap.add_argument('--steps',type=int,default=12)
    a=ap.parse_args()
    STABLE=a.stable_backward; OUTDIR=a.outdir; STEPS=a.steps
    seeds=[int(s) for s in a.seeds.split(',')]
    gpus=[int(g) for g in a.gpus.split(',')]
    if not gpus or len(set(gpus))!=len(gpus) or not set(gpus)<=set(range(8)):
        raise ValueError('unique physical GPU ids 0-7 required')
    jobs=[(ds,s,t) for ds in a.datasets.split(',') for s in seeds for t in range(a.targets)]
    code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for d in ('auditkit','experiments','fz') for p in (ROOT/d).glob('*.py')}
    dest=ROOT/'results'/OUTDIR;dest.mkdir(parents=True,exist_ok=True)
    (dest/'protocol.json').write_text(json.dumps({'phase':OUTDIR,'seeds':seeds,
        'stable_backward':STABLE,
        'targets_per_seed_dataset':a.targets,'budget':3,'optimizer_steps':STEPS,
        'constraints':'context range, explicit schema, dtarget>=0.5*Rnn, dreal<=1.5*Rnn',
        'source_sha256':code,'args':vars(a)},indent=2))
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        records=[r for result in ex.map(worker,[(g,jobs[i::len(gpus)])
                                               for i,g in enumerate(gpus)]) for r in result]
    (dest/'manifest.json').write_text(json.dumps(records,indent=2))
