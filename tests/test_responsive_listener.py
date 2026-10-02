import math
from copy import deepcopy
import pytest
from nod.config import load_config,validate_config
from nod.responsive import configure
from nod.belief.listener import ListenerBelief
from nod.belief.reports import report_logs
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.semantic.sensor import fixture,SCHEMA,compatibility


def cfg():return configure(load_config())
def send(e,k,t,**data):return e.process(Event(k,t,data))
def commands(rows):return [r for r in rows if r['kind']=='command']
def semantic(e,q,t,seq=1):
    return send(e,'semantic_result',t,request={'seq':seq,'dispatched_ms':t-10,'snapshot':e.units.current.to_dict()},
        result={'schema':SCHEMA,'observation':q,'probabilities':compatibility(q),'model':'fixture'})
def engine():
    e=create_engine(cfg())
    send(e,'controller_ready',0,neutral_expression=True,supported_expressions=['neutral','warm','concerned','attentive'])
    send(e,'asr',0,utterance_id='u',text='I am disappointed',source_ms=0,confidence=.99,final=True)
    send(e,'opportunity',10,score=.01,source_ms=10)
    return e


def test_sparse_negative_report_not_reversed_by_unrelated_zeros():
    c=cfg();b=ListenerBelief(c['listener']);q=fixture('negative',resolved=.89)
    q['appraisal']={'neutral':.14,'positive':0.,'negative':.86,'mixed':0.}
    b.observe(.85,q,0);s=b.summary(0)
    assert s['appraisal']['negative']>s['appraisal']['neutral']*3
    assert s['understanding']<=s['perception']
    assert s['observation_model_status']=='designed_unfitted'


def test_conditional_appraisal_odds_follow_report_ratio():
    b=ListenerBelief(cfg()['listener']);b.anchor=(.1,.1,.2,.2,.2,.2)
    q=fixture();q['appraisal']={'neutral':.2,'positive':.1,'negative':.6,'mixed':.1}
    b.observe(.8,q,0);p=b.posterior(0)
    assert p[4]/p[2]==pytest.approx(3.)
    prior=0
    for negative in (.01,.1,.3,.6,.9,.99):
        q['appraisal']={'neutral':1-negative,'positive':0.,'negative':negative,'mixed':0.}
        b.observe(.8,q,0);p=b.posterior(0)
        assert p[4]>prior;prior=p[4]


def test_appraisal_label_permutation_equivariance():
    m=cfg()['listener']['observation_model']
    a=report_logs([.8,.8,.1,.2,.6,.1],m)
    b=report_logs([.8,.8,.6,.1,.2,.1],m)
    assert a[:2]==pytest.approx(b[:2])
    assert [a[4],a[2],a[3],a[5]]==pytest.approx(b[2:])


def test_undefined_appraisal_is_marginalized_not_neutral_evidence_for_u():
    b=ListenerBelief(cfg()['listener'])
    q=fixture('neutral',resolved=.01);b.observe(.95,q,0)
    assert b.summary(0)['understanding']<.03
    logs=[report_logs([.7,.7,*p],b.cfg['observation_model'])[:2]
          for p in ([1.,0.,0.,0.],[.25]*4,[0.,0.,1.,0.])]
    for x in logs[1:]:assert x==pytest.approx(logs[0])


def test_missing_semantics_keeps_conditional_understanding_prior():
    b=ListenerBelief(cfg()['listener']);b.observe(.95,None,0);p=b.posterior(0)
    assert sum(p[2:])/(1-p[0])==pytest.approx(sum(b.anchor[2:])/(1-b.anchor[0]))
    assert all(math.isfinite(x) for x in p)


def test_same_packet_and_ticks_do_not_accumulate_confidence():
    b=ListenerBelief(cfg()['listener']);q=fixture('negative');b.observe(.85,q,0);p=b.posterior(100)
    for _ in range(10):b.observe(.85,q,0);assert b.posterior(100)==p


def test_empathic_upgrade_after_completed_continuer_without_full_cooldown():
    e=engine();rows=commands(send(e,'opportunity',100,score=.99,source_ms=100))
    assert rows[0]['function']=='continuer';a=rows[0]['command']['action_id']
    send(e,'feedback',110,action_id=a,status='started')
    assert not commands(semantic(e,fixture('negative'),120))
    send(e,'feedback',400,action_id=a,status='completed')
    assert not commands(send(e,'opportunity',450,score=.1,source_ms=450))
    rows=commands(send(e,'opportunity',550,score=.1,source_ms=550))
    assert len(rows)==1 and rows[0]['function']=='empathic'
    assert rows[0]['command']['expression']=='concerned'


def test_continuer_needs_new_content_and_neutral_cannot_bypass_cooldown():
    e=engine();a=commands(send(e,'opportunity',100,score=.99,source_ms=100))[0]['command']['action_id']
    send(e,'feedback',400,action_id=a,status='completed')
    assert not commands(semantic(e,fixture('neutral',resolved=.1),600))
    assert e.motor.state=='COOLDOWN'
    rows=send(e,'opportunity',2300,score=.99,source_ms=2300)
    assert not commands(rows)
    rows=send(e,'asr',2350,utterance_id='u',text='I am disappointed today',source_ms=2350,confidence=.99,final=True)
    assert commands(rows)


def test_presentation_receipt_is_idempotent_and_not_motor_feedback():
    e=engine();a=commands(semantic(e,fixture('negative'),120))[0]['command']['action_id']
    before=e.belief.posterior(150)
    rows=send(e,'presentation',150,action_id=a)
    assert rows[0]['dispatch_to_receipt_ms']==30
    assert e.motor.state=='SENT' and e.belief.posterior(150)==before
    assert send(e,'presentation',160,action_id=a)==[]
    assert send(e,'presentation',170,action_id='a999')==[]


def test_report_family_guard_and_invalid_concentrations():
    c=cfg();validate_config(c)
    for v in (0,True,float('nan'),21):
        x=deepcopy(c);x['listener']['observation_model']['concentration']['appraisal']=v
        with pytest.raises(ValueError):validate_config(x)
    from nod.incremental import configure as old
    x=old(load_config());x['listener']['observation_model']=c['listener']['observation_model']
    with pytest.raises(ValueError):validate_config(x)


def test_expression_can_refine_when_new_content_resolves_stance_but_not_repeat():
    from nod.policy.incremental import IncrementalPolicy
    p=IncrementalPolicy(cfg()['listener']['policy'])
    history=[dict(function='empathic',unit_id='u',status='completed',at_ms=0,expression='attentive')]
    belief=[.01,.01,.01,.01,.95,.01]
    f,_,_=p.choose(belief,history,2000,'u',fixture('negative'),.01,False,True)
    assert f=='empathic'
    history.append(dict(function='empathic',unit_id='u',status='completed',at_ms=1800,expression='concerned'))
    f,_,c=p.choose(belief,history,2200,'u',fixture('negative'),.01,False,True)
    assert 'empathic' not in c['allowed']


def test_prior_affect_does_not_justify_expression_on_neutral_new_observation():
    from nod.policy.incremental import IncrementalPolicy
    p=IncrementalPolicy(cfg()['listener']['policy'])
    f,_,c=p.choose([.01,.01,.01,.95,.01,.01],[],0,'u',fixture('neutral'),.01,False,True)
    assert 'empathic' not in c['allowed']


def test_continuer_is_not_emitted_late_or_after_specific_feedback():
    from nod.policy.incremental import IncrementalPolicy
    p=IncrementalPolicy(cfg()['listener']['policy']);b=[.01,.95,.01,.01,.01,.01]
    f,_,c=p.choose(b,[],1500,'u',None,1,True,False,evidence_text='hello',evidence_age_ms=1500)
    assert f=='none' and 'continuer' not in c['allowed']
    h=[dict(function='empathic',unit_id='u',at_ms=0,status='completed')]
    f,_,c=p.choose(b,h,1500,'u',None,1,True,False,evidence_text='hello again')
    assert 'continuer' not in c['allowed']


def test_sensevoice_format_normalization_preserves_english_and_negation():
    from nod.audio.sensevoice import normalize_text
    assert normalize_text('嬉しい わけ では ない。','ja')=='嬉しいわけではない。'
    assert normalize_text('I am not happy.','en')=='I am not happy.'


def test_fit_requires_reviewed_calibration_and_matching_report_family(tmp_path):
    import json
    from nod.experiments.listener import fit
    from nod.belief.listener import STATES
    from nod.belief.reports import FAMILY
    a=[];p=[]
    for state in STATES:
        for j in range(8):
            ident=f'{state}-{j}'
            a.append(dict(id=ident,schema=SCHEMA,reviewed=True,state=state,annotator='test fixture',split='calibration'))
            p.append(dict(id=ident,schema=SCHEMA,observation_family=FAMILY,probability_report=[.8,.8,.1,.1,.7,.1],prompt_sha256='test',model='fixture'))
    ann,pred=tmp_path/'a.jsonl',tmp_path/'p.jsonl'
    ann.write_text('\n'.join(map(json.dumps,a)),encoding='utf-8');pred.write_text('\n'.join(map(json.dumps,p)),encoding='utf-8')
    result=fit(ann,pred)
    assert result['n']==48 and result['observation_model']['family']==FAMILY
    assert all(.05<=x<=20 for x in result['observation_model']['concentration'].values())
    a[0]['split']='test';ann.write_text('\n'.join(map(json.dumps,a)),encoding='utf-8')
    with pytest.raises(ValueError):fit(ann,pred)
