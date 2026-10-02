"""Controls isolate filter memory; segment closure is not a turn label."""
from copy import deepcopy
import json
import pytest
from nod.config import load_config, validate_config
from nod.live_session import live_config
from nod.belief.filter import BeliefFilter
from nod.core.engine import Engine
from nod.core.events import Event
from nod.telemetry import SessionLog, replay
from nod.experiments.research import review_session


def cfg(profile='bayes'):
    return live_config(load_config(),'en',0,response=profile)


def after_history(profile, first):
    f=BeliefFilter(cfg(profile)['belief'])
    f.observe(first,1,0)
    f.open_epoch(100)
    return f.observe((.05,.10,.40,.45),.8,200)


def test_identical_observation_depends_on_history_only_in_memory_condition():
    positive=(.01,.01,.97,.01);emotional=(.01,.01,.01,.97)
    assert after_history('bayes',positive)!=pytest.approx(after_history('bayes',emotional))
    for profile in ('direct','bayes-no-history'):
        assert after_history(profile,positive)==pytest.approx(after_history(profile,emotional))


def test_memory_ablation_changes_only_history_in_model_and_keeps_other_controls():
    a,b=cfg(),cfg('bayes-no-history')
    for field in ('semantic','timing','policy','motor','prominence','asr'):
        assert a[field]==b[field]
    aa,bb=deepcopy(a['belief']),deepcopy(b['belief'])
    assert aa.pop('history_mode')=='carry' and bb.pop('history_mode')=='none'
    assert aa==bb
    d=cfg('direct')
    for field in ('semantic','timing','policy','motor','prominence','asr'):
        assert d[field]==a[field]
    f=BeliefFilter(d['belief']);p=(.01,.01,.01,.97)
    assert f.observe(p,1,0)==pytest.approx(p)
    assert f.observe(p,.5,0)==pytest.approx([.125+.5*x for x in p])
    assert f.observe(p,0,0)==pytest.approx([.25]*4)


def test_no_history_repair_does_not_restore_previous_utterance():
    f=BeliefFilter(cfg('bayes-no-history')['belief'])
    f.observe((.01,.01,.01,.97),1,0);f.open_epoch(100)
    f.observe((.01,.01,.97,.01),1,200);f.invalidate(300)
    assert f.at(300)==pytest.approx([.25]*4)


def test_direct_cannot_silently_carry_history():
    c=cfg('direct');c['belief']['history_mode']='carry'
    with pytest.raises(ValueError,match='Direct'):validate_config(c)


@pytest.mark.parametrize('profile',['bayes','bayes-no-history','direct'])
@pytest.mark.parametrize('reason',[None,'silence','max_duration','input_end'])
def test_only_observed_silence_can_enable_endpoint_response(profile,reason):
    e=Engine(cfg(profile))
    e.process(Event('controller_ready',0,{'neutral_expression':True}))
    e.process(Event('asr',0,{'utterance_id':'u','text':'I finally passed.','source_ms':0,
                           'final':True,'final_reason':reason}))
    e.process(Event('opportunity',100,{'score':.1,'source_ms':100}))
    records=e.process(Event('semantic_result',110,{'request':{'seq':1,'dispatched_ms':100,
        'snapshot':e.transcripts.current.to_dict()},'result':{'probabilities':[.01,.01,.01,.97]}}))
    commands=[r for r in records if r['kind']=='command']
    assert bool(commands)==(reason=='silence')
    if commands:assert commands[0]['function']=='empathic'


def test_final_refinement_preserves_max_duration_boundary():
    e=Engine(cfg())
    e.process(Event('controller_ready',0,{}))
    e.process(Event('asr',0,{'utterance_id':'u','text':'A story','source_ms':0}))
    e.process(Event('semantic_result',100,{'request':{'seq':1,'dispatched_ms':50,
        'snapshot':e.transcripts.current.to_dict()},'result':{'probabilities':[.01,.01,.97,.01]}}))
    e.process(Event('opportunity',200,{'score':.1,'source_ms':200}))
    records=e.process(Event('asr',210,{'utterance_id':'u','text':'A story','source_ms':210,
                                     'final':True,'final_reason':'max_duration'}))
    assert any(x['kind']=='semantic_refinement' for x in records)
    assert not any(x['kind']=='command' for x in records)
    assert e.semantic['final_reason']=='max_duration'


def test_review_uses_same_items_for_condition_predictions_and_counts_motion(tmp_path):
    c=cfg();e=Engine(c);path=tmp_path/'s.jsonl';log=SessionLog(path,c,'fixture')
    def send(kind,t,data):
        event=Event(kind,t,data);records=e.process(event);log.event(event,records)
    send('controller_ready',0,{'neutral_expression':True})
    send('asr',0,{'utterance_id':'u','text':'I passed.','source_ms':0,'final':True,'final_reason':'silence'})
    send('opportunity',100,{'score':.1,'source_ms':100})
    send('semantic_result',110,{'request':{'seq':1,'dispatched_ms':100,
        'snapshot':e.transcripts.current.to_dict(),'state':{'stable_transcript':'I passed.','recent_context':[]}},
        'result':{'probabilities':[.01,.01,.01,.97]}})
    log.close();assert replay(path)['matched']
    out=tmp_path/'review';review_session(path,out)
    report=json.loads((out/'comparison.json').read_text())
    assert report['final_reason_counts']=={'silence':1}
    for name in ('likelihood_product_symmetric','bayes-no-history','direct'):
        r=report['conditions'][name]
        assert r['function_counts']=={'empathic':1} and r['commanded_motion_ms']>0
        assert r['mean_commanded_intensity']>0
        pred=json.loads((out/'predictions-by-condition'/f'{name}.jsonl').read_text())
        label=json.loads((out/'annotations.jsonl').read_text())
        assert pred['id']==label['id'] and label['label'] is None


def test_segmenter_reports_actual_closure_reason():
    np=pytest.importorskip('numpy')
    from nod.audio.live import Segmenter
    s=Segmenter(partial_ms=200,silence_ms=100,max_ms=400)
    loud=np.ones(320);quiet=np.zeros(320)
    for i in range(20):_,job=s.push(loud,20*(i+1),320*(i+1))
    assert job.final_reason=='max_duration'
    s.push(loud,420,6720)
    for i in range(5):_,job=s.push(quiet,440+i*20,7040+i*320)
    assert job.final_reason=='silence'
    s.push(loud,540,8640)
    assert s.finish(540,8640).final_reason=='input_end'


@pytest.mark.parametrize('blocked',[None,'baseline','unstable','unfinished','ellipsis','uncertain','stale_audio','append','repair','already_acknowledged','cutoff'])
def test_presentation_clause_requires_current_stable_specific_evidence(blocked):
    e=Engine(cfg('bayes' if blocked=='baseline' else 'presentation'))
    text='This explains the contrast.'
    if blocked=='unfinished':text='This explains the'
    if blocked=='ellipsis':text='This explains...'
    e.process(Event('controller_ready',0,{'neutral_expression':True}))
    e.process(Event('asr',0,{'utterance_id':'u','text':text,'source_ms':0}))
    e.process(Event('asr',400,{'utterance_id':'u','text':text,'source_ms':400}))
    snapshot=e.transcripts.current.to_dict()
    if blocked=='unstable':
        from dataclasses import replace
        e.transcripts.current=replace(e.transcripts.current,stable='')
    if blocked in ('append','repair','cutoff'):
        d={'utterance_id':'u','text':text,'source_ms':410}
        if blocked=='append':d['text']+=' However'
        if blocked=='repair':d['text']='This does not explain it.'
        if blocked=='cutoff':d.update(final=True,final_reason='max_duration')
        e.process(Event('asr',410,d))
    if blocked=='already_acknowledged':e.specific_ack_utterance='u'
    e.process(Event('opportunity',420,{'score':.1,'source_ms':0 if blocked=='stale_audio' else 420}))
    p=[.1,.25,.55,.1] if blocked=='uncertain' else [.01,.01,.97,.01]
    records=e.process(Event('semantic_result',450,{'request':{'seq':1,'dispatched_ms':420,'snapshot':snapshot},
                                                'result':{'probabilities':p}}))
    commands=[r for r in records if r['kind']=='command']
    assert bool(commands)==(blocked is None)
    if commands:
        assert commands[0]['trigger']=='stable_clause' and commands[0]['function']=='understanding'
        assert e.specific_ack_utterance=='u'


def test_presentation_changes_timing_extension_without_changing_bayesian_model():
    a,b=cfg(),cfg('presentation')
    for field in ('belief','semantic','timing','motor','prominence'):
        assert a[field]==b[field]
    aa,bb=deepcopy(a['policy']),deepcopy(b['policy'])
    assert aa.pop('stable_clause_ack') is False and bb.pop('stable_clause_ack') is True
    assert aa==bb
