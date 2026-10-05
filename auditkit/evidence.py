"""Evidence extraction for agent tools, without running any model.

Reports are untrusted data, not instructions. Paths stay within a configured local
artifact directory; only schema 1.0 finite JSON is accepted.
"""
import json
from pathlib import Path


class EvidenceStore:
    def __init__(self,root):
        self.root=Path(root).resolve()

    def reports(self):
        result=[]
        for p in sorted(self.root.rglob('*.json')) if self.root.exists() else []:
            try:
                r=self.read(str(p.relative_to(self.root)))
            except (ValueError,KeyError,OSError):
                continue
            result.append({'report_id':str(p.relative_to(self.root)),
                'dataset':r.get('dataset',{}).get('dataset','unspecified'),
                'baseline':r['baseline'], 'n_methods':len(r['attacks'])})
        return result

    def read(self,report_id):
        p=(self.root/report_id).resolve()
        if not p.is_relative_to(self.root) or p.suffix!='.json':
            raise ValueError('report path must be a JSON file inside the report directory')
        if p.stat().st_size>10_000_000:
            raise ValueError('report exceeds 10 MB safety limit')
        def reject(value):
            raise ValueError('nonfinite JSON value: '+value)
        r=json.loads(p.read_text(),parse_constant=reject)
        if not isinstance(r,dict) or r.get('schema_version')!='1.0':
            raise ValueError('unsupported report schema')
        if not isinstance(r.get('attacks'),list) or not isinstance(r.get('baseline'),dict):
            raise ValueError('missing report sections')
        return r

    def findings(self,report_id):
        r=self.read(report_id)
        return {'report_id':report_id,'baseline':r['baseline'],
                'experiment':r.get('experiment',{}),
                'attacks':[{k:a.get(k) for k in ('method','n_rows','p_before','p_after','valid',
                            'flipped','prediction_changed','unconstrained','evaluations','elapsed_s',
                            'aborted','nonfinite_gradient_entries')} for a in r['attacks']],
                'verification':r.get('verification',[]),
                'receiver_controls':r.get('receiver_controls',[]),
                'influence':r.get('influence',{}),
                'warnings':r.get('warnings',[]),
                'interpretation_rules':[
                    'A failed search is not a certificate of robustness.',
                    'Report geometric/schema constraints, not domain realism.',
                    'Separate surrogate flips from verified receiver flips.',
                    'Duplicate references may be unconstrained; never compare them as constrained attacks.',
                    'No claim of the smallest possible poisoning budget.',
                    'These are research labels, not clinical or lending advice.']}

    def attack(self,report_id,method):
        r=self.read(report_id)
        a=next((a for a in r['attacks'] if a.get('method')==method),None)
        if a is None:raise ValueError('method not recorded')
        return {'report_id':report_id,'attack':a,'target':r.get('target',
                 r.get('diagnostics',{}).get('target')),
                'feature_names':r.get('feature_names'),
                'verification':[v for v in r.get('verification',[]) if v.get('attack_method')==method]}

    def markdown(self,report_id):
        f=self.findings(report_id)
        lines=['# Decision stress audit',f'Artifact: `{report_id}`','',
               'This report describes a finite search, not a robustness guarantee.','',
               '| Method | Rows | Valid | Surrogate probability before → after | Decision changed |',
               '|---|---:|---|---|---|']
        for a in f['attacks']:
            lines.append(f"| {a['method']} | {a['n_rows']} | {a['valid']} | "
                         f"{a['p_before']} → {a['p_after']} | {a['prediction_changed']} |")
        lines+=['','## Verification']
        for v in f['verification']:
            lines.append(f"- {v['model']} / {v['attack_method']}: "
                         f"{v.get('p_before')} → {v.get('p_after')}, "
                         f"flip={v.get('flipped')}, valid={v.get('candidate_valid')}")
        lines+=['','## Limitations']+[f'- {w}' for w in f['warnings']]
        return '\n'.join(lines)
