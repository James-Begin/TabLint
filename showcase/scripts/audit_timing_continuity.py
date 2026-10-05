"""Reject source-time cuts that skip a change in an authored visible layer."""
import json
from pathlib import Path

DEFAULTS = dict(x=0, y=0, s=1, sx=1, sy=1, r=0, sk=0, o=1)
def pose(motion, seconds, duration):
    keys=[]
    previous=DEFAULTS.copy()
    for key in motion['keys']:
        previous={**previous, **key}; keys.append(previous.copy())
    progress=seconds/duration
    if progress>=keys[-1]['t']: return keys[-1]
    for a,b in zip(keys,keys[1:]):
        if a['t']<=progress<=b['t']:
            u=(progress-a['t'])/max(.0001,b['t']-a['t']);u=u*u*(3-2*u)
            return {key:a[key]+(b[key]-a[key])*u for key in DEFAULTS}
    return keys[0]

def audit(directory):
    directory=Path(directory)
    shots=json.loads((directory/'source-shots.json').read_text())['shots']
    timeline=json.loads((directory/'timing.json').read_text())
    cuts=[]
    for shot,scene in zip(shots,timeline['scenes']):
        for a,b in zip(scene['segments'],scene['segments'][1:]):
            before,after=a['sourceTo'],b['sourceFrom']
            if before==after: continue
            changes=[]
            for motion in shot['motions']:
                old=pose(motion,before,shot['duration']);new=pose(motion,after,shot['duration'])
                if old['o']<.001 and new['o']<.001:continue
                delta={key:round(new[key]-old[key],6) for key in DEFAULTS if abs(new[key]-old[key])>1e-5}
                if delta:changes.append({'layer':motion['id'],'delta':delta})
            cuts.append({'scene':shot['id'],'frame':scene['startFrame']+b['frame'],'from':before,'to':after,'changes':changes})
    report={'source_time_cuts':cuts,'visible_motion_jumps':sum(len(c['changes']) for c in cuts)}
    (directory/'timing-continuity.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['visible_motion_jumps']==0, report
    return report
if __name__=='__main__':
    import sys
    report=audit(sys.argv[1]);print('Checked',len(report['source_time_cuts']),'source-time cuts; no skipped visible motion.')
