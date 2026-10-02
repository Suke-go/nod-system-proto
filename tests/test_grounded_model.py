from copy import deepcopy
import math
import pytest
from nod.config import load_config,validate_config
from nod.grounded import configure
from nod.policy.risk import BayesRiskPolicy,ACTIONS
from nod.belief.listener import ListenerBelief
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.semantic.sensor import fixture,SCHEMA,compatibility


def config():return configure(load_config())
def policy():return BayesRiskPolicy(config()['listener']['policy'])
def send(e,k,t,**data):return e.process(Event(k,t,data))
def semantic(e,q,t,seq=1):
    return send(e,'semantic_result',t,request={'seq':seq,'dispatched_ms':t-10,'snapshot':e.units.current.to_dict()},
        result={'schema':SCHEMA,'observation':q,'probabilities':compatibility(q),'model':'fixture'})
def choose(b,q,**kw):return policy().choose(b,[],0,'u',q,.01,False,False,**kw)


@pytest.mark.parametrize('kind,index',[('warm',3),('concerned',4)])
def test_the_selected_expression_is_the_one_whose_risk_is_minimized(kind,index):
    b=[.01]*6;b[index]=.95
    q=fixture('positive' if kind=='warm' else 'negative',completion=.3)
    p=policy();f,u,c=p.choose(b,[],0,'u',q,.01,False,False)
    assert c['selected_realization']==kind and f=='empathic'
    assert c['risks'][kind]==min(c['risks'][a] for a in c['allowed_realizations'])
    summary={'posterior':b,'appraisal':dict(zip(('neutral','positive','negative','mixed'),b[2:])),
             'perception':1-b[0],'understanding':sum(b[2:])}
    assert p.realize(f,summary,0,True,c)[2]==kind
    assert u[f]==pytest.approx(-sum(c['risk_terms'][kind].values()))


def test_uncertain_emotional_direction_prefers_nondirectional_expression():
    f,_,c=choose([.01,.01,.02,.46,.46,.04],fixture('mixed',completion=.1))
    assert f=='empathic' and c['selected_realization']=='attentive'
    assert c['risks']['attentive']<c['risks']['warm'] and c['risks']['attentive']<c['risks']['concerned']


def test_capability_is_checked_before_choice_not_silently_changed_afterwards():
    f,_,c=choose([.01,.01,.01,.95,.01,.01],fixture('positive',completion=.1),supported_expressions=['neutral','attentive'])
    assert c['selected_realization']=='attentive'
    assert c['excluded_realizations']['warm']==['expression_unavailable']
    f,_,c=choose([.01,.01,.01,.95,.01,.01],fixture('positive',completion=.1),supported_expressions=['neutral'])
    assert f!='empathic'


def test_receipt_uses_integrated_belief_not_duplicate_raw_threshold():
    # No hidden 0.65 threshold, no 0.85 readiness override; ASR support is in b.
    b=[.10,.10,.77,.01,.01,.01]
    q=fixture('neutral',resolved=.64,completion=.99,demand=.01)
    f,_,c=choose(b,q)
    assert f=='understanding'
    assert 'readiness' not in c and 'local_receipt_supported' not in c
    assert c['selected_realization']=='receipt'


def test_risks_change_continuously_across_previous_thresholds():
    b=[.1,.1,.77,.01,.01,.01]
    contexts=[choose(b,fixture('neutral',completion=x,demand=.1))[2] for x in (.64999,.65001)]
    assert abs(contexts[0]['risks']['receipt']-contexts[1]['risks']['receipt'])<.0001
    assert contexts[0]['allowed_realizations']==contexts[1]['allowed_realizations']


def test_incomplete_and_substantive_question_costs_prevent_false_receipt():
    b=[0.,0.,1.,0.,0.,0.]
    for q in (fixture('neutral',completion=0.,demand=0.),fixture('neutral',completion=1.,demand=1.)):
        f,_,c=choose(b,q)
        assert f!='understanding' and c['risks']['receipt']>c['risks']['none']
    assert choose(b,fixture('neutral',completion=1.,demand=0.))[0]=='understanding'


def test_perception_failure_does_not_produce_strong_semantic_cues():
    for kind in ('neutral','positive','negative','mixed'):
        f,_,c=choose([.96,.01,.01,.01,.005,.005],fixture(kind))
        assert f=='none'


def test_semantics_missing_or_retained_cannot_claim_current_receipt():
    b=[.01,.01,.95,.01,.01,.01]
    assert 'receipt' not in choose(b,None)[2]['allowed_realizations']
    assert 'receipt' not in choose(b,fixture('neutral'),retained=True)[2]['allowed_realizations']


def test_invalidated_evidence_does_not_erase_physical_action_budget():
    p=policy();b=[.01,.01,.95,.01,.01,.01]
    h=[{'at_ms':i,'unit_id':f'u{i}','function':'continuer','status':'completed','invalidated':True} for i in range(4)]
    f,_,c=p.choose(b,h,500,'new',fixture('neutral'),1,True,False)
    assert f=='none' and c['reason']=='action_budget'
    assert c['allowed_realizations']==['none']


def test_expression_refinement_does_not_repeat_the_same_realization():
    p=policy();b=[.01,.01,.01,.95,.01,.01]
    h=[dict(at_ms=0,unit_id='u',function='empathic',expression='attentive',status='completed')]
    f,_,c=p.choose(b,h,2000,'u',fixture('positive',completion=.1),.1,False,False)
    assert c['selected_realization']=='warm'
    h.append(dict(at_ms=2100,unit_id='u',function='empathic',expression='warm',status='completed'))
    assert 'warm' not in p.choose(b,h,2300,'u',fixture('positive'),.1,False,False)[2]['allowed_realizations']


def test_time_propagation_obeys_one_semigroup_with_or_without_stored_packet():
    cfg=config()['listener'];b=ListenerBelief(cfg)
    b.observe(.9,fixture('negative'),100)
    at_900=b.posterior(900)
    propagated=ListenerBelief(cfg);propagated.anchor=at_900;propagated.anchor_ms=900
    assert propagated.posterior(1800)==pytest.approx(b.posterior(1800))
    for t in (100,500,1000,5000):
        p=b.posterior(t)
        assert sum(p)==pytest.approx(1.) and all(math.isfinite(x) and x>=0 for x in p)


def test_punctuation_and_repairs_never_commit_overlapping_asr_evidence():
    cfg=config();cfg['listener']['policy']['decision_model']['minimum_gain']=100
    e=create_engine(cfg)
    send(e,'asr',0,utterance_id='u',text='I am happy. Then',source_ms=0,confidence=.99)
    anchor=e.belief.anchor;anchor_ms=e.belief.anchor_ms
    semantic(e,fixture('positive'),50)
    send(e,'asr',500,utterance_id='u',text='I am happy. Then it failed.',source_ms=500,confidence=.99)
    assert e.units.start>0
    assert e.belief.anchor==anchor and e.belief.anchor_ms==anchor_ms
    semantic(e,fixture('negative'),550,2)
    send(e,'asr',600,utterance_id='u',text='I am not happy. Then it failed.',source_ms=600,confidence=.99)
    assert e.belief.anchor==anchor and e.belief.anchor_ms==anchor_ms
    assert e.belief.packet[1] is None and not e.memory.items
    send(e,'asr',700,utterance_id='v',text='',source_ms=700,final=False)
    assert e.belief.anchor_ms==700


def test_no_history_ablation_keeps_same_scope_revision_memory():
    cfg=config();cfg['listener']['condition']='no_history'
    cfg['listener']['policy']['decision_model']['minimum_gain']=100
    e=create_engine(cfg)
    send(e,'asr',0,utterance_id='u',text='I am disappointed',source_ms=0,confidence=.99,final=True)
    semantic(e,fixture('negative'),50)
    send(e,'asr',100,utterance_id='u',text='I am disappointed because',source_ms=100,confidence=.99)
    assert e.memory.candidates(e.transcripts.current,100)
    e.belief.anchor=(1.,0.,0.,0.,0.,0.)
    assert e.belief.predicted(100)==tuple(cfg['listener']['initial'])


@pytest.mark.parametrize('value',[float('nan'),-1,True])
def test_invalid_risk_costs_rejected(value):
    cfg=config();cfg['listener']['policy']['decision_model']['loss']['warm'][4]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_conflicting_emergency_override_cannot_be_silently_enabled():
    cfg=config();cfg['listener']['policy']['local_receipt']={'readiness':.85,'minimum_resolved':.65,'suppress_filler_only':True}
    with pytest.raises(ValueError):validate_config(cfg)
