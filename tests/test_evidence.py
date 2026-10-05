import json
import pytest
from auditkit.evidence import EvidenceStore


def report():
    return {'schema_version':'1.0','baseline':{'p_positive':.8},
            'attacks':[{'method':'gradient','n_rows':1,'rows':[[1]],'labels':[0],
             'p_before':.8,'p_after':.3,'valid':True,'flipped':True}],
            'warnings':['No robustness certificate.']}


def test_evidence_uses_real_artifact(tmp_path):
    (tmp_path/'audit.json').write_text(json.dumps(report()))
    s=EvidenceStore(tmp_path)
    assert len(s.reports())==1
    assert s.attack('audit.json','gradient')['attack']['rows']==[[1]]
    assert s.findings('audit.json')['warnings']==['No robustness certificate.']
    assert 'No robustness certificate.' in s.markdown('audit.json')


def test_rejects_traversal_nan_and_wrong_schema(tmp_path):
    s=EvidenceStore(tmp_path)
    with pytest.raises(ValueError):s.read('../secret.json')
    (tmp_path/'bad.json').write_text('{"schema_version":"1.0","x":NaN}')
    with pytest.raises(ValueError):s.read('bad.json')
    (tmp_path/'old.json').write_text('{"schema_version":"0.5"}')
    with pytest.raises(ValueError):s.read('old.json')
