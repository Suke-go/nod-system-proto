from copy import deepcopy
import math
import pytest
from nod.config import load_config,validate_config
from nod.incremental import configure
from nod.asr.incremental import IncrementalTranscriptTracker
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.semantic.evidence import EvidenceMemory, words
from nod.semantic.sensor import fixture, SCHEMA, compatibility
from nod.policy.incremental import IncrementalPolicy


def cfg():return configure(load_config())
def send(e,k,t,**data):return e.process(Event(k,t,data))
def commands(rows):return [r for r in rows if r['kind']=='command']
def engine():
    e=create_engine(cfg())
    send(e,'controller_ready',0,neutral_expression=True,supported_expressions=['neutral','warm','concerned','attentive'])
    send(e,'asr',0,utterance_id='u',text='I am disappointed',source_ms=0,confidence=.99,final=True)
    send(e,'opportunity',100,score=.01,source_ms=100)
    return e
def semantic(e,q,t=120,seq=1,snapshot=None):
    return send(e,'semantic_result',t,request={'seq':seq,'dispatched_ms':t-10,'snapshot':snapshot or e.units.current.to_dict()},
        result={'schema':SCHEMA,'observation':q,'probabilities':compatibility(q),'model':'fixture'})


def test_open_negative_content_can_express_without_claiming_full_receipt():
    e=engine();r=commands(semantic(e,fixture('negative',completion=.1,demand=.9)))
    assert len(r)==1 and r[0]['function']=='empathic'
    assert r[0]['command']['expression']=='concerned'
    assert 'understanding' not in r[0]['decision_context']['allowed']
    assert r[0]['command']['intensity']<=.4


def test_question_does_not_automatically_trigger_empathy_or_agreement():
    e=engine();r=commands(semantic(e,fixture('neutral',completion=.95,demand=.95)))
    assert not any(x['function'] in ('empathic','understanding') for x in r)


def test_missing_expression_capability_does_not_fallback_to_strong_nod():
    e=engine();e.neutral_expression=False
    assert not any(x['function']=='empathic' for x in commands(semantic(e,fixture('negative',completion=.1))))


def test_semantic_prefix_survives_unfinished_append_and_is_source_anchored():
    e=engine();e.cfg['listener']['policy']['minimum_gain']=100
    semantic(e,fixture('negative',completion=.4));item=e.memory.items[0];ident=item['id']
    send(e,'asr',250,utterance_id='u',text='I am disappointed because',source_ms=250,confidence=.99)
    assert e.belief.packet[1] is None
    assert e.memory.items[0]['id']==ident and e.memory.items[0]['source_ms']==0
    e.cfg['listener']['policy']['minimum_gain']=.08
    r=commands(send(e,'opportunity',260,source_ms=260,score=.01))
    assert r[0]['function']=='empathic' and r[0]['retained_evidence']
    assert r[0]['evidence_text']=='I am disappointed'


def test_unresolved_larger_scope_does_not_delete_interpreted_prefix():
    e=engine();e.cfg['listener']['policy']['minimum_gain']=100
    semantic(e,fixture('negative',completion=.4))
    send(e,'asr',250,utterance_id='u',text='I am disappointed because',source_ms=250,confidence=.99)
    semantic(e,fixture('negative',resolved=.2,completion=.1),280,2)
    assert len(e.memory.items)==1 and e.memory.items[0]['text']=='I am disappointed'


def test_same_scope_replacement_with_uncertainty_is_not_confidence_cherry_picking():
    e=engine();e.cfg['listener']['policy']['minimum_gain']=100
    semantic(e,fixture('negative'))
    semantic(e,fixture('negative',resolved=.1),150,2)
    assert not e.memory.items


@pytest.mark.parametrize('text',['I am not disappointed','Someone else is disappointed'])
def test_lexical_revision_revokes_dependent_memory(text):
    e=engine();semantic(e,fixture('negative'))
    r=send(e,'asr',250,utterance_id='u',text=text,source_ms=250,confidence=.99,final=True)
    assert not e.memory.items
    assert any(x['kind']=='interpretations_revoked' for x in r)
    assert any(x['kind']=='feedback_evidence_revoked' for x in r)


def test_conflicting_larger_interpretation_supersedes_old_appraisal():
    e=engine();e.cfg['listener']['policy']['minimum_gain']=100
    semantic(e,fixture('negative',completion=.4))
    send(e,'asr',250,utterance_id='u',text='I am disappointed but actually relieved',source_ms=250,confidence=.99)
    semantic(e,fixture('positive',resolved=.8),280,2)
    assert len(e.memory.items)==1 and e.memory.items[0]['observation']['appraisal']['positive']>.9


def test_ticks_do_not_refresh_old_semantics_and_new_turn_clears_memory():
    e=engine();e.cfg['listener']['policy']['minimum_gain']=100
    semantic(e,fixture('negative'));b=e.memory.items[0]['belief'];before=b.posterior(120)
    for _ in range(20):send(e,'tick',120)
    assert b.posterior(120)==before
    send(e,'tick',3600);assert not e.memory.items
    send(e,'asr',3700,utterance_id='v',text='New topic',source_ms=3700,confidence=.99)
    assert not e.memory.items


def test_delayed_old_request_cannot_survive_negation_repair():
    e=engine();old=e.units.current.to_dict()
    send(e,'asr',150,utterance_id='u',text='I am not disappointed',source_ms=150,confidence=.99)
    r=semantic(e,fixture('negative'),180,snapshot=old)
    assert next(x for x in r if x['kind']=='semantic_disposition')['reason']=='repaired_input'
    assert not e.memory.items


def test_same_unit_expression_does_not_repeat_but_receipt_is_separate():
    p=IncrementalPolicy(cfg()['listener']['policy'])
    hist=[{'function':'empathic','unit_id':'u','evidence_id':'e1','status':'completed','at_ms':0}]
    b=[.01,.01,.01,.95,.01,.01]
    f,_,c=p.choose(b,hist,2500,'u',fixture('positive'),1,True,True,evidence_id='e2')
    assert 'empathic' not in c['allowed']
    b=[.01,.01,.95,.01,.01,.01]
    f,_,c=p.choose(b,hist,2500,'u',fixture(),1,True,True,evidence_id='e3')
    assert f=='understanding'


def test_retained_scope_never_claims_understanding_of_new_unfinished_tail():
    p=IncrementalPolicy(cfg()['listener']['policy'])
    f,_,c=p.choose([.01,.01,.95,.01,.01,.01],[],0,'u',fixture(),1,True,True,retained=True)
    assert 'understanding' not in c['allowed'] and f=='none'


def test_stability_ignores_formatting_but_not_question_or_negation():
    t=IncrementalTranscriptTracker(cfg())
    t.update('u','今日は忙しいです',0,0)
    a=t.update('u','今日は、忙しいです。',500,500)
    assert a.reliability==pytest.approx(1.) and a.repair_epoch==0
    b=t.update('u','今日は忙しくないです。',600,600)
    assert b.repair_epoch==1 and b.reliability<a.reliability
    assert words('Are you ready?')!=words('Are you ready.')


def test_no_history_condition_disables_retained_prefixes():
    e=engine();e.cfg['listener']['condition']='no_history';e.cfg['listener']['policy']['minimum_gain']=100
    semantic(e,fixture('negative'))
    assert not e.memory.candidates(e.transcripts.current,130)


@pytest.mark.parametrize('key,value',[('retention_ms',999999),('max_items',True),('retain_reliability',float('nan'))])
def test_incremental_configuration_validation(key,value):
    c=cfg();c['listener']['incremental'][key]=value
    with pytest.raises(ValueError):validate_config(c)


def test_live_default_uses_new_engine_while_v2_replays_keep_old_engine():
    from nod.live_session import live_config
    from nod.listener import configure as old
    assert type(create_engine(live_config(load_config(),'ja',0))).__name__=='IncrementalListenerEngine'
    assert type(create_engine(old(load_config()))).__name__=='ListenerEngine'
