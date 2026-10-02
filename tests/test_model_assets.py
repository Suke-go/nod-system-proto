import json
import pytest
from nod.audio.models import verify_assets
from nod.experiments.common import sha256


def test_model_manifest_rejects_modified_weights_and_wrong_context(tmp_path):
    assets = ['asr/model.bin','asr/config.json','asr/tokenizer.json','timing/model.pt','timing/cpc.pt']
    for relative in assets:
        path=tmp_path/relative; path.parent.mkdir(exist_ok=True); path.write_bytes(b'fixture')
    manifest={'schema_version':1,'asr_model':'base','mode':'bc','language':'jp','frame_rate':10,
              'context_seconds':20,'sha256':{r:sha256(tmp_path/r) for r in assets}}
    path=tmp_path/'manifest.json'; path.write_text(json.dumps(manifest),encoding='utf-8')
    assert verify_assets(tmp_path)['context_seconds'] == 20
    (tmp_path/'timing/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'): verify_assets(tmp_path)
    manifest['context_seconds']=5; path.write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='Unsupported'): verify_assets(tmp_path)


def test_model_manifest_cannot_read_outside_model_root(tmp_path):
    root=tmp_path/'models'; root.mkdir()
    assets=['asr/model.bin','asr/config.json','asr/tokenizer.json','timing/model.pt','timing/cpc.pt']
    manifest={'schema_version':1,'asr_model':'base','mode':'bc','language':'jp','frame_rate':10,
              'context_seconds':20,'sha256':{'../outside':'invalid',**{r:'invalid' for r in assets}}}
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='changed'): verify_assets(root)
