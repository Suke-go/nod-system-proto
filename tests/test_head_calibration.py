"""Synthetic labels below verify algorithms, not empirical listener performance."""
from copy import deepcopy
import math
import json
import pytest
from nod.config import load_config, validate_config
from nod.grounded import configure
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.semantic.sensor import fixture, SCHEMA as OBS_SCHEMA, prompt_digest
from nod.experiments.calibration import (
    SCHEMA, HEADS, digest, jsonl, read, pairs, fit, score, configured, export,
    validate_artifact, scalar_transform,
)


def data(tmp_path, split='calibration', prefix='train'):
    cfg=configure(load_config());annotations=[];predictions=[]
    for i in range(64):
        state={'stable_transcript':f'{prefix}-{i}', 'current_partial':'', 'recent_context':[]}
        labels={head:names[i%len(names)] for head,names in HEADS.items()}
        q=fixture(resolved=.8,completion=.95,demand=.95)
        # Half positive labels: overconfident uninformative binary reports.
        a=dict(schema=SCHEMA,id=f'{prefix}-{i}',session_group=f'{prefix}-session-{i%2}',
               participant=f'{prefix}-person-{i%2}',unit_id=str(i),input_sha256=digest(state),
               language='ja' if i%2 else 'en',split=split,input=state,labels=labels,reviewed=True,annotator='synthetic-test-only')
        binding=digest({k:a[k] for k in ('session_group','participant','unit_id','input_sha256','language','split')})
        predictions.append(dict(schema=SCHEMA,id=a['id'],binding=binding,observation=q,
            model=cfg['semantic']['model'],prompt_sha256=prompt_digest(cfg['semantic']['question_resource'])))
        annotations.append(a)
    ap=tmp_path/f'{prefix}-annotations.jsonl';pp=tmp_path/f'{prefix}-predictions.jsonl'
    jsonl(ap,annotations);jsonl(pp,predictions);return ap,pp


def overwrite(path,rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')


def test_fit_reduces_binary_nll_on_independent_synthetic_test(tmp_path):
    artifact=fit(*data(tmp_path));validate_artifact(artifact)
    result=score(*data(tmp_path,'test','heldout'),artifact)
    assert result['participant_disjoint_verified']
    for head in ('completion','response_demand'):
        m=result['metrics']['all'][head]
        assert m['transformed']['nll']<m['raw']['nll']
    assert artifact['unfitted']==['perception','initial','transition','time_decay','action_losses']


def test_dirichlet_density_fit_matches_analytic_binary_mle(tmp_path):
    ap,pp=data(tmp_path);labels=read(ap);preds=read(pp)
    for a,p in zip(labels,preds):
        correct=a['labels']['interpretability'];other=next(k for k in HEADS['interpretability'] if k!=correct)
        p['observation']['interpretability']={correct:.8,other:.2}
    overwrite(pp,preds)
    artifact=fit(ap,pp)
    assert artifact['concentration']['interpretability']==pytest.approx(-1/math.log(.8)-1,rel=1e-6)


@pytest.mark.parametrize('change',['input','split','participant','session_group'])
def test_immutable_export_binding(tmp_path,change):
    ap,pp=data(tmp_path);rows=read(ap)
    if change=='input':rows[0][change]['stable_transcript']='changed'
    else:rows[0][change]='changed'
    overwrite(ap,rows)
    with pytest.raises(ValueError,match='Changed'):pairs(ap,pp)


def test_unknown_labels_are_missing_not_negative_and_review_is_explicit(tmp_path):
    ap,pp=data(tmp_path);rows=read(ap)
    for r in rows:r['labels']['completion']=None
    overwrite(ap,rows)
    assert score(ap,pp)['metrics']['all']['completion']['raw'] is None
    with pytest.raises(ValueError,match='completion'):fit(ap,pp)
    for r in rows:r['reviewed']=False
    overwrite(ap,rows)
    assert score(ap,pp)['reviewed']==0
    with pytest.raises(ValueError,match='No human-reviewed'):fit(ap,pp)
    rows[0].update(reviewed=True,annotator='')
    overwrite(ap,rows)
    with pytest.raises(ValueError,match='reviewer'):pairs(ap,pp)


@pytest.mark.parametrize('field',['training_sessions','training_inputs','training_participants'])
def test_heldout_leakage_rejected(tmp_path,field):
    artifact=fit(*data(tmp_path));ap,pp=data(tmp_path,'test','heldout');a=read(ap)[0]
    key={'training_sessions':'session_group','training_inputs':'input_sha256','training_participants':'participant'}[field]
    artifact[field].append(a[key])
    with pytest.raises(ValueError,match='leakage|overlaps'):score(ap,pp,artifact)


def test_development_cannot_be_fit_or_be_claimed_heldout(tmp_path):
    ap,pp=data(tmp_path,'development','dev')
    with pytest.raises(ValueError,match='calibration split'):fit(ap,pp)
    with pytest.raises(ValueError,match='test split'):score(ap,pp,fit(*data(tmp_path)))


def test_model_and_prompt_must_match_runtime_and_test(tmp_path):
    artifact=fit(*data(tmp_path));artifact['model']='other'
    with pytest.raises(ValueError,match='match'):configured(artifact)
    with pytest.raises(ValueError,match='mismatch'):score(*data(tmp_path,'test','heldout'),artifact)


def test_partial_fit_preserves_unfitted_parameters_and_transforms_once(tmp_path):
    artifact=fit(*data(tmp_path));cfg=configured(artifact);base=configure(load_config())
    validate_config(cfg)
    for key in ('initial','transition','state_half_life_ms','policy'):
        assert cfg['listener'][key]==base['listener'][key]
    assert cfg['listener']['observation_model']['concentration']['perception']==1
    assert cfg['listener']['observation_model']['status']=='partially_fitted_on_calibration'
    e=create_engine(cfg)
    e.process(Event('asr',0,dict(utterance_id='u',text='I feel tired',source_ms=0,confidence=.99,final=True)))
    q=fixture('negative',completion=.9,demand=.1);before=deepcopy(q)
    request=dict(seq=1,dispatched_ms=10,snapshot=e.units.current.to_dict())
    result=dict(schema=OBS_SCHEMA,observation=q,model=cfg['semantic']['model'])
    event=Event('semantic_result',20,dict(request=request,result=result))
    rows=e.process(event)
    expected=scalar_transform(q,cfg['listener']['scalar_calibration'])
    assert e.semantic['observation']==expected and q==before
    assert next(r for r in rows if r['kind']=='semantic_trace')['observation']==expected
    e.process(Event('semantic_result',21,dict(request=request,result=result)))
    assert e.semantic['observation']==expected
    request['seq']=2;result['model']='unexpected-model'
    rows=e.process(Event('semantic_result',22,dict(request=request,result=result)))
    assert any(r.get('reason')=='calibration_model_mismatch' for r in rows)
    assert e.last_seq==1


def test_live_custom_config_keeps_calibration_and_pins_question(tmp_path):
    from nod.live_session import live_config
    cfg=configured(fit(*data(tmp_path)));before=deepcopy(cfg['listener'])
    assert live_config(cfg,'en',8765,custom=True)['listener']==before
    cfg['semantic']['model']='other'
    with pytest.raises(ValueError,match='mismatch'):validate_config(cfg)


@pytest.mark.parametrize('value',[0,float('nan'),True,21])
def test_invalid_scale_never_reaches_runtime(tmp_path,value):
    cfg=configured(fit(*data(tmp_path)));cfg['listener']['scalar_calibration']['temperature']['completion']=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_export_hides_predictions_and_escapes_script_content(tmp_path):
    cfg=configure(load_config());state=dict(stable_transcript='</script><script>alert(1)</script>',current_partial='',recent_context=[])
    cfg['experiment']={'semantic_prompt_sha256':prompt_digest(cfg['semantic']['question_resource'])}
    result=dict(observation=fixture(),model=cfg['semantic']['model'])
    def row(seq):return dict(event=dict(kind='semantic_result',data=dict(request=dict(seq=seq,state=state,snapshot=dict(utterance_id='u')),result=result)),derived=[dict(kind='semantic_disposition',reason='accepted')])
    path=tmp_path/'session.jsonl';jsonl(path,[dict(config=cfg),row(1),row(2)])
    out=tmp_path/'review';manifest=export([path],out)
    assert manifest['items']==1
    annotations=read(out/'annotations.jsonl');predictions=read(out/'predictions.jsonl')
    assert predictions[0]['source_seq']==2 and not predictions[0]['audio_available']
    assert all(v is None for v in annotations[0]['labels'].values())
    html=(out/'review.html').read_text(encoding='utf-8')
    assert '</script><script>alert(1)' not in html
    assert '"observation"' not in html and '"source_seq"' not in html
    assert score(out/'annotations.jsonl',out/'predictions.jsonl')['reviewed']==0
    with pytest.raises(ValueError,match='Duplicate source'):export([path,path],tmp_path/'duplicate')
    cfg.pop('experiment');missing=tmp_path/'missing.jsonl';jsonl(missing,[dict(config=cfg),row(1)])
    with pytest.raises(ValueError,match='Recorded question'):export([missing],tmp_path/'missing-review')


def test_deleted_rows_are_not_silently_excluded_from_coverage(tmp_path):
    ap,pp=data(tmp_path);overwrite(ap,read(ap)[1:])
    with pytest.raises(ValueError,match='coverage mismatch'):score(ap,pp)
