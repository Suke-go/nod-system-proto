import json
import math
from pathlib import Path
import pytest
from nod.config import load_config
from nod.live_session import live_config
from nod.belief.filter import Transition,BeliefFilter
from nod.core.engine import Engine
from nod.core.events import Event
from nod.telemetry import SessionLog,replay
from nod.experiments.research import ablation_configs,score_annotations,review_session


def make_partial():
    cfg=live_config(load_config(),'ja',0,response='bayes');e=Engine(cfg)
    e.process(Event('asr',0,{'utterance_id':'u','text':'合格しました','source_ms':0,'final':False}))
    return cfg,e,e.transcripts.current.to_dict()


def response(snapshot,at=100,seq=1):
    return Event('semantic_result',at,{'request':{'seq':seq,'snapshot':snapshot,'dispatched_ms':at-10,
        'state':{'stable_transcript':snapshot['stable'],'current_partial':snapshot['text'][len(snapshot['stable']):],'recent_context':[]}},
        'result':{'probabilities':[.02,.03,.1,.85],'model':'fixture','confidence':.9}})


def test_explicit_legacy_bayes_profile_retains_research_equation():
    cfg=live_config(load_config(),'ja',0,response='bayes')
    assert cfg['belief']['observation_rule']=='likelihood_product'
    assert cfg['belief']['initial']==[.25]*4
    assert cfg['research']['calibration']=='identity_unfitted'
    vap=live_config(load_config(),'ja',0,response='bayes-vap')
    assert vap['belief']==cfg['belief'] and not vap['policy']['endpoint_specific_ack']['enabled']
    assert live_config(load_config(),'ja',0,response='responsive')['belief']['observation_rule']=='posterior_blend'


def test_transition_is_a_stochastic_semigroup_and_matches_prediction():
    t=Transition([.25]*4,4000);b=(.6,.25,.1,.05)
    a=t.matrix(300);c=t.matrix(700);combined=t.matrix(1000)
    for row in combined:assert sum(row)==pytest.approx(1.) and all(x>=0 for x in row)
    product=[[sum(a[i][k]*c[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    for i in range(4):assert product[i]==pytest.approx(combined[i])
    assert [sum(b[i]*combined[i][j] for i in range(4)) for j in range(4)]==pytest.approx(t.predict(b,1000))


def test_paper_update_equation_and_reliability_zero():
    cfg=live_config(load_config(),'ja',0,response='bayes')['belief'];f=BeliefFilter(cfg);p=(.02,.03,.1,.85)
    weights=[.25*x**.6 for x in p];expected=[x/sum(weights) for x in weights]
    assert f.observe(p,.6,0)==pytest.approx(expected)
    assert f.observe(p,0,0)==pytest.approx([.25]*4)


def test_final_exact_text_replaces_evidence_once_without_extending_repeatedly():
    cfg,e,s=make_partial();e.process(response(s));oldq=e.semantic['reliability']
    records=e.process(Event('asr',300,{'utterance_id':'u','text':s['text'],'source_ms':300,'final':True}))
    assert any(r['kind']=='semantic_refinement' for r in records)
    assert e.semantic['final'] and e.semantic['original_source_ms']==0 and e.semantic['reliability']>oldq
    fresh=BeliefFilter(cfg['belief'])
    assert e.belief.at(300)==pytest.approx(fresh.observe((.02,.03,.1,.85),e.semantic['reliability'],300))
    records=e.process(Event('asr',500,{'utterance_id':'u','text':s['text'],'source_ms':500,'final':True}))
    assert not any(r['kind']=='semantic_refinement' for r in records)
    assert e.semantic['source_ms']==300


@pytest.mark.parametrize('change',['append','repair','new_utterance','expired'])
def test_final_reuse_never_bypasses_content_or_freshness_checks(change):
    _,e,s=make_partial();e.process(response(s));text=s['text'];uid='u';now=300
    if change=='append':text+='と思ったら違いました'
    if change=='repair':text='不合格でした'
    if change=='new_utterance':uid='u2'
    if change=='expired':now=3500
    records=e.process(Event('asr',now,{'utterance_id':uid,'text':text,'source_ms':now,'final':True}))
    assert not any(r['kind']=='semantic_refinement' for r in records)


def test_inflight_exact_result_uses_final_metadata_already_available():
    _,e,s=make_partial()
    e.process(Event('asr',200,{'utterance_id':'u','text':s['text'],'source_ms':200,'final':True}))
    records=e.process(response(s,300))
    trace=next(r for r in records if r['kind']=='semantic_trace')
    assert trace['source_final'] and trace['final_refined'] and trace['reliability']==1.
    assert e.semantic['original_source_ms']==0


def test_ablation_changes_one_factor_from_bayesian_reference():
    configs=ablation_configs(live_config(load_config(),'ja',0,response='responsive'))
    base=configs['likelihood_product_symmetric'];other=configs['posterior_blend_symmetric']
    assert base['semantic']==other['semantic'] and base['policy']==other['policy']
    b1=dict(base['belief']);b2=dict(other['belief']);b1.pop('observation_rule');b2.pop('observation_rule')
    assert b1==b2
    assert not configs['bayes_without_final_refinement']['semantic']['refine_exact_final']


def test_unreviewed_data_never_produces_accuracy_and_reviewed_ids_are_validated(tmp_path):
    a=tmp_path/'a.jsonl';p=tmp_path/'p.jsonl'
    label={'id':'x','reviewed':False,'label':None,'annotator':None}
    a.write_text(json.dumps(label));p.write_text(json.dumps({'id':'x','raw':[.1,.7,.1,.1],'filtered':[.1,.7,.1,.1]}))
    assert score_annotations(a,p)['metrics'] is None
    label.update(reviewed=True,label='continuer',annotator='rater1');a.write_text(json.dumps(label))
    metrics=score_annotations(a,p)['metrics']['raw']
    assert metrics['accuracy']==1 and metrics['brier_multiclass_sum']==pytest.approx(.12)
    assert metrics['nll']==pytest.approx(-math.log(.7))
    a.write_text(json.dumps(label)+'\n'+json.dumps(label))
    with pytest.raises(ValueError,match='Duplicate'):score_annotations(a,p)


def test_annotation_export_is_blinded_and_research_log_replays(tmp_path):
    cfg,e,s=make_partial();log=SessionLog(tmp_path/'log.jsonl',cfg,'fixture');e=Engine(cfg)
    first=Event('asr',0,{'utterance_id':'u','text':s['text'],'source_ms':0,'final':False})
    for event in (first,response(s),Event('asr',300,{'utterance_id':'u','text':s['text'],'source_ms':300,'final':True})):
        log.event(event,e.process(event))
    log.close();assert replay(tmp_path/'log.jsonl')['matched']
    report=review_session(tmp_path/'log.jsonl',tmp_path/'review');assert report['annotation_items']==1
    item=json.loads((tmp_path/'review/annotations.jsonl').read_text(encoding='utf-8'))
    assert item['label'] is None and not item['reviewed'] and item['split']=='development'
    assert 'raw' not in item and 'filtered' not in item and 'prediction' not in item
    assert item['state']['current_partial']==s['text']
