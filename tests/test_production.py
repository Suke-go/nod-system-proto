import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import httpx
import pytest
from nod.config import load_config,validate_config
from nod.production import configure
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.semantic.frames import FRAME_ORDER,FRAMES,SCHEMA,fixture_frame,frame_distribution,marginals
from nod.semantic.backend import JevBackend,SemanticError
from nod.semantic.units import UnitTracker,boundaries
from nod.asr.stability import Snapshot
from nod.belief.interaction import InteractionBelief
from nod.telemetry import SessionLog,replay


def cfg():return configure(load_config())


def send(e,kind,at,**data):return e.process(Event(kind,at,data))
def commands(records):return [r for r in records if r['kind']=='command']


def ready(text='説明は以上です。',final=True,score=.1):
    e=create_engine(cfg())
    send(e,'controller_ready',0,neutral_expression=True,supported_expressions=['neutral','warm','concerned','attentive'])
    send(e,'asr',0,utterance_id='u',text=text,source_ms=0,confidence=.99,final=final,
         **({'final_reason':'silence'} if final else {}))
    send(e,'opportunity',100,source_ms=100,score=score)
    return e


def result(e,frame,at=120,seq=1,snapshot=None):
    return send(e,'semantic_result',at,request={'seq':seq,'dispatched_ms':at-10,
        'snapshot':snapshot or e.units.current.to_dict()},
        result={'schema':SCHEMA,'frames':fixture_frame(frame),'model':'test-fixture','confidence':.98})


def test_joint_distribution_and_marginals_are_consistent():
    p=fixture_frame('negative_experience_complete')
    m=marginals(p)
    for axis in m.values():assert sum(axis.values())==pytest.approx(1)
    assert m['act']['experience']>.98 and m['stance']['negative']>.98
    assert m['function']['empathic']>.98
    with pytest.raises(ValueError):frame_distribution({FRAME_ORDER[0]:1})
    with pytest.raises(ValueError):frame_distribution([float('nan')]*len(FRAMES))


@pytest.mark.parametrize('bad',['missing','choice','nan'])
def test_invalid_typed_observation_is_not_used(bad):
    async def scenario():
        p=dict(zip(FRAME_ORDER,fixture_frame('explanation_complete')))
        a={'type':'choice','choice':'explanation_complete','confidence':.98,'probabilities':p}
        if bad=='missing':p.pop('unclear')
        if bad=='choice':a['choice']='question_open'
        if bad=='nan':p['unclear']=float('nan')
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,
                content=json.dumps({'model':'fixture','answers':{'semantic_frame':a}})))) as client:
            backend=JevBackend(cfg()['semantic'],client=client,api_key='fixture')
            with pytest.raises(SemanticError,match='invalid_response'):await backend.evaluate({})
    asyncio.run(scenario())


def test_typed_request_uses_joint_question_and_preserves_distribution():
    async def scenario():
        def handler(request):
            body=json.loads(request.content)
            assert set(body['questions'])=={'semantic_frame'}
            assert set(body['questions']['semantic_frame']['criteria'])==set(FRAMES)
            return httpx.Response(200,json={'model':'fixture','answers':{'semantic_frame':{
                'type':'choice','choice':'negative_experience_complete','confidence':.98,
                'probabilities':dict(zip(FRAME_ORDER,fixture_frame('negative_experience_complete')))}}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            r=await JevBackend(cfg()['semantic'],client=client,api_key='fixture').evaluate({})
            assert r['schema']==SCHEMA and len(r['frames'])==15
    asyncio.run(scenario())


def test_repeated_correlated_observations_replace_instead_of_sharpening():
    b=InteractionBelief(cfg()['interaction']);p=fixture_frame('explanation_complete')
    b.observe(p,.8,100);once=b.frames_at(100)
    for _ in range(100):b.observe(p,.8,100)
    assert b.frames_at(100)==once
    b.acoustic(.8,100,100);joint=b.joint_at(100)
    for _ in range(100):b.acoustic(.8,100,100)
    assert b.joint_at(100)==joint and sum(joint)==pytest.approx(1)


def test_revoke_restores_pre_observation_anchor_and_stale_audio_removes_readiness():
    b=InteractionBelief(cfg()['interaction']);b.observe(fixture_frame('positive_experience_complete'),1,0)
    b.revoke()
    assert b.frames_at(0)==pytest.approx(b.uniform)
    b.acoustic(.9,100,100)
    assert sum(b.joint_at(100)[1::2])>0
    assert sum(b.joint_at(500)[1::2])==0


@pytest.mark.parametrize('frame,function,expression',[
    ('explanation_complete','understanding','neutral'),
    ('negative_experience_complete','empathic','concerned'),
    ('positive_experience_complete','empathic','warm'),
    ('mixed_experience_complete','empathic','attentive'),
])
def test_meaning_selects_receipt_or_stance_with_low_vap(frame,function,expression):
    e=ready();out=commands(result(e,frame))
    assert len(out)==1
    assert out[0]['function']==function and out[0]['command']['expression']==expression
    assert out[0]['trigger']=='joint_state' and out[0]['unit_id']=='u@0'


@pytest.mark.parametrize('frame',['unclear','question_open','question_complete','repair_open'])
def test_question_unclear_and_unfinished_repair_do_not_produce_false_acknowledgement(frame):
    e=ready(score=.9)
    assert not commands(result(e,frame))


def test_missing_and_malformed_semantics_never_become_fake_evidence():
    e=ready(score=.9)
    assert not commands(send(e,'tick',110))
    records=send(e,'semantic_result',120,request={'seq':1,'dispatched_ms':110,'snapshot':e.units.current.to_dict()},
                 result={'probabilities':[0,0,1,0]})
    assert records[0]['reason']=='incompatible_semantic_schema' and e.semantic is None
    assert not commands(records)


def test_append_after_classification_suppresses_specific_feedback_until_updated():
    e=ready();snap=e.units.current.to_dict()
    send(e,'asr',110,utterance_id='u',text=snap['text']+'ただしまだ',source_ms=110,confidence=.99)
    # This result refers to an older unit, or an incomplete prefix, never the new content.
    assert not commands(result(e,'positive_experience_complete',snapshot=snap))


def test_asr_repair_revokes_old_result_and_records_irreversible_feedback():
    e=ready('We succeeded.')
    cmd=commands(result(e,'positive_experience_complete'))[0]['command']
    snap=e.units.current.to_dict()
    records=send(e,'asr',140,utterance_id='u',text='We did not succeed.',source_ms=140,confidence=.99)
    assert any(r['kind']=='feedback_evidence_revoked' for r in records)
    assert e.history[0]['invalidated'] and e.semantic is None
    records=result(e,'positive_experience_complete',at=150,seq=2,snapshot=snap)
    assert records[0]['reason']=='repaired_input' and not commands(records)
    assert e.motor.command['action_id']==cmd['action_id']


def test_acknowledged_content_not_repeated_after_cooldown():
    e=ready();cmd=commands(result(e,'explanation_complete'))[0]['command']
    send(e,'feedback',130,action_id=cmd['action_id'],status='completed')
    send(e,'opportunity',2100,score=.95,source_ms=2100)
    assert not commands(result(e,'explanation_complete',at=2110,seq=2))
    assert len(e.history)==1


def test_new_meaning_unit_inside_one_asr_utterance_can_receive_its_own_response():
    e=ready('最初の説明です。');cmd=commands(result(e,'explanation_complete'))[0]['command']
    send(e,'feedback',130,action_id=cmd['action_id'],status='completed')
    send(e,'asr',2100,utterance_id='u',text='最初の説明です。次の説明です。',source_ms=2100,
         final=True,final_reason='silence',confidence=.99)
    send(e,'opportunity',2110,score=.1,source_ms=2110)
    out=commands(result(e,'explanation_complete',at=2130,seq=2))
    assert out and out[0]['unit_id']!='u@0' and len(e.history)==2
    assert e.units.context[-1]['text']=='最初の説明です。'


def test_units_do_not_split_unstable_words_abbreviations_or_decimals():
    assert list(boundaries('Dr. Smith paid 3.14 dollars.'))==[28]
    u=UnitTracker(2000)
    snap=Snapshot('u',1,0,'First sentence. Second clause','First',0,0,.8)
    s,new,_=u.update(snap,[]);assert s.text==snap.text
    s,new,_=u.update(replace(snap,stable='First sentence. '),[])
    assert new and s.text=='Second clause' and u.context[-1]['text']=='First sentence. '


@pytest.mark.parametrize('blocked',['audio_stale','semantic_stale','input_unhealthy','disconnected'])
def test_freshness_and_input_state_constrain_actions(blocked):
    e=ready()
    if blocked=='input_unhealthy':send(e,'input_health',110,healthy=False)
    if blocked=='disconnected':send(e,'controller_disconnected',110)
    at=500 if blocked=='audio_stale' else 4000 if blocked=='semantic_stale' else 120
    assert not commands(result(e,'explanation_complete',at=at))


def test_production_log_is_exactly_replayable(tmp_path):
    c=cfg();e=create_engine(c);log=SessionLog(tmp_path/'s.jsonl',c,'fixture')
    def record(kind,at,**data):
        ev=Event(kind,at,data);out=e.process(ev);log.event(ev,out);return out
    record('controller_ready',0,neutral_expression=True)
    record('asr',0,utterance_id='u',text='I finally passed.',source_ms=0,final=True,final_reason='silence')
    record('opportunity',100,score=.1,source_ms=100)
    record('semantic_result',120,request={'seq':1,'dispatched_ms':100,'snapshot':e.units.current.to_dict()},
           result={'schema':SCHEMA,'frames':fixture_frame('positive_experience_complete')})
    log.close()
    assert replay(tmp_path/'s.jsonl')['matched']


def test_default_live_profile_is_production_and_legacy_is_explicit():
    from nod.live_session import live_config
    c=live_config(load_config(),'ja',0,response='legacy-production')
    assert c['interaction']['enabled'] and c['semantic']['schema']==SCHEMA
    assert 'interaction' not in live_config(load_config(),'ja',0,response='bayes')


@pytest.mark.parametrize('key,value',[('temperature',0),('unit_carry',2),('semantic_half_life_ms',float('nan'))])
def test_invalid_parameters_rejected(key,value):
    c=cfg();c['interaction'][key]=value
    with pytest.raises(ValueError):validate_config(c)
