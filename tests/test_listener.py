import asyncio
from copy import deepcopy
import json
import math
import httpx
import pytest
from nod.config import load_config,validate_config
from nod.listener import configure
from nod.belief.listener import ListenerBelief,STATES,gaussian_logs
from nod.semantic.sensor import SCHEMA,fixture,compatibility,parse,features,prompt_digest
from nod.semantic.backend import JevBackend,SemanticError
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.runtime import Runtime
from nod.telemetry import replay
from nod.experiments.listener import review,fit,score,load_model


def config():return configure(load_config())
def send(e,kind,at,**data):return e.process(Event(kind,at,data))
def actions(records):return [r for r in records if r['kind']=='command']
def ready():
    e=create_engine(config())
    send(e,'controller_ready',0,neutral_expression=True,supported_expressions=['neutral','warm','concerned','attentive'])
    send(e,'asr',0,utterance_id='u',text='A complete statement.',source_ms=0,confidence=.99,final=True,final_reason='silence')
    send(e,'opportunity',100,score=.1,source_ms=100)
    return e
def result(e,q,at=120,seq=1,snapshot=None):
    return send(e,'semantic_result',at,request={'seq':seq,'dispatched_ms':at-10,'snapshot':snapshot or e.units.current.to_dict()},
        result={'schema':SCHEMA,'observation':q,'probabilities':compatibility(q),'model':'fixture'})


def test_filter_uses_joint_density_not_classifier_posterior_as_likelihood():
    b=ListenerBelief(config()['listener']);q=fixture('negative');b.observe(.9,q,0)
    logs=gaussian_logs(features(.9,q),b.cfg['observation_model'])
    expected=[p*math.exp(l) for p,l in zip(b.cfg['initial'],logs)]
    assert b.posterior(0)==pytest.approx([p/sum(expected) for p in expected])
    assert sum(b.posterior(0))==pytest.approx(1)
    assert b.summary(0)['understanding']<=b.summary(0)['perception']


def test_density_marginalizes_missing_semantics_and_models_correlation():
    c=config()['listener'];m=c['observation_model'];x=[1.,None,None,None,None]
    logs=gaussian_logs(x,m)
    expected=-.5*math.log(2*math.pi*m['covariance'][0][0])-.5*(1-m['means'][0][0])**2/m['covariance'][0][0]
    assert logs[0]==pytest.approx(expected)
    independent=deepcopy(m)
    independent['covariance']=[[m['covariance'][i][j] if i==j else 0 for j in range(5)] for i in range(5)]
    assert gaussian_logs(features(.9,fixture()),m)!=gaussian_logs(features(.9,fixture()),independent)


def test_duplicate_packets_do_not_accumulate_evidence_and_revoke_restores_checkpoint():
    b=ListenerBelief(config()['listener']);q=fixture()
    b.observe(.9,q,100);first=b.posterior(200)
    for _ in range(100):b.observe(.9,q,100)
    assert b.posterior(200)==first
    b.revoke();assert b.posterior(200)==pytest.approx(b.cfg['initial'])


def test_transition_is_applied_only_when_unit_commits_and_time_decays():
    b=ListenerBelief(config()['listener']);b.observe(.9,fixture('positive'),0)
    old=b.posterior(100)
    b.open_unit(100)
    assert b.anchor==pytest.approx([sum(old[i]*b.cfg['transition'][i][j] for i in range(6)) for j in range(6)])
    assert b.posterior(100000)==pytest.approx(b.cfg['initial'],abs=1e-5)


def test_neutral_sensor_certainty_does_not_establish_understanding():
    b=ListenerBelief(config()['listener']);q=fixture(resolved=.15)
    q['appraisal']={'neutral':1.,'positive':0.,'negative':0.,'mixed':0.}
    b.observe(.99,q,0)
    assert b.summary(0)['perception']>.8
    assert b.summary(0)['understanding']<.2


@pytest.mark.parametrize('appraisal,function,expression',[
    ('neutral','understanding','neutral'),('positive','empathic','warm'),
    ('negative','empathic','concerned'),('mixed','empathic','attentive')])
def test_sensor_drives_listener_state_then_differentiated_action(appraisal,function,expression):
    e=ready();r=actions(result(e,fixture(appraisal)))
    assert len(r)==1 and r[0]['function']==function
    assert r[0]['command']['expression']==expression
    assert r[0]['trigger']=='listener_belief' and e.belief.summary(120)['understanding']>.65


@pytest.mark.parametrize('q',[fixture(resolved=.05),fixture(completion=.05),fixture(demand=.95)])
def test_no_specific_ack_for_unresolved_open_or_answer_demand(q):
    e=ready();assert not any(r['function'] in ('understanding','empathic') for r in actions(result(e,q)))


def test_ack_does_not_create_grounding_and_revised_evidence_is_revoked():
    e=ready();r=actions(result(e,fixture('negative')))[0];before=e.belief.posterior(120)
    for status in ('accepted','started','completed'):
        send(e,'feedback',120,action_id=r['command']['action_id'],status=status)
    assert e.belief.posterior(120)==before
    records=send(e,'asr',130,utterance_id='u',text='That was incorrect.',source_ms=130,final=True)
    assert any(x['kind']=='feedback_evidence_revoked' for x in records)
    assert e.semantic is None and e.belief.packet[1] is None


def test_append_discards_old_semantic_factor_until_latest_text_is_interpreted():
    e=ready();result(e,fixture(completion=.05))
    send(e,'asr',140,utterance_id='u',text='A complete statement. However',source_ms=140,final=False)
    assert e.belief.packet[1] is None


@pytest.mark.parametrize('mutation',[lambda c:c['listener']['transition'][0].__setitem__(0,2),
    lambda c:c['listener']['observation_model']['covariance'][0].__setitem__(0,-1),
    lambda c:c['listener']['observation_model']['means'][0].__setitem__(0,float('nan'))])
def test_model_validation_rejects_invalid_math(mutation):
    c=config();mutation(c)
    with pytest.raises(ValueError):validate_config(c)


def response_body(q):
    answers={}
    for key in ('interpretability','appraisal'):
        answers[key]={'type':'choice','probabilities':q[key],'choice':max(q[key],key=q[key].get),'confidence':.9}
    for key in ('completion','response_demand'):answers[key]={'type':'noul','noul':q[key]}
    return {'model':'fixture','answers':answers}


def test_provider_adapter_uses_typed_sensor_and_rejects_incomplete_response():
    async def run():
        def handler(request):
            assert set(json.loads(request.content)['questions'])==set(fixture())
            return httpx.Response(200,json=response_body(fixture('negative')))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            r=await JevBackend(config()['semantic'],client=client,api_key='fixture').evaluate({})
            assert r['observation']['appraisal']['negative']==.94 and r['schema']==SCHEMA
    asyncio.run(run())
    invalid=response_body(fixture());del invalid['answers']['completion']
    with pytest.raises(KeyError):parse(invalid)


def test_runtime_websocket_replay_and_blind_comparison(tmp_path):
    class Backend:
        async def evaluate(self,state):
            q=fixture('negative' if 'lost' in state['stable_transcript'] else 'neutral')
            return {'schema':SCHEMA,'observation':q,'probabilities':compatibility(q),'model':'fixture'}
        async def close(self):pass
    async def run():
        c=config();c['output']['port']=0;c['motor']['cooldown_after_completion_ms']=50
        r=Runtime(c,'mock',tmp_path/'log.jsonl');r.semantic.backend=Backend()
        async def feed():
            for uid,text in [('a','A complete explanation.'),('b','I lost my work and feel terrible.')]:
                r.emit('asr',{'utterance_id':uid,'text':text,'source_ms':r.clock.now_ms(),'final':True,'final_reason':'silence','confidence':.99})
                for _ in range(15):
                    r.emit('opportunity',{'score':.1,'source_ms':r.clock.now_ms()});await asyncio.sleep(.1)
        return await r.run(feed)
    r=asyncio.run(run());assert r['functions']==['understanding','empathic']
    assert replay(tmp_path/'log.jsonl')['matched']
    exported=review(tmp_path/'log.jsonl',tmp_path/'review');assert exported['annotation_items']==2
    annotations=tmp_path/'review/annotations.jsonl';preds=tmp_path/'review/predictions.jsonl'
    assert score(annotations,preds)['reviewed']==0
    rows=[json.loads(s) for s in annotations.read_text(encoding='utf-8').splitlines()]
    assert 'posterior' not in rows[0] and 'observation' not in rows[0]


def test_acoustic_only_requires_no_api_key_and_makes_no_classifier_calls(tmp_path,monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    c=configure(load_config(),'acoustic_only');c['output']['port']=0
    r=Runtime(c,'jev',tmp_path/'acoustic.jsonl')
    assert not r.semantic_enabled
    async def run():
        async def feed():
            r.emit('asr',{'utterance_id':'u','text':'A statement.','source_ms':r.clock.now_ms(),'final':True,'confidence':.99})
            for _ in range(10):
                r.emit('opportunity',{'score':.9,'source_ms':r.clock.now_ms()});await asyncio.sleep(.1)
        result=await r.run(feed)
        assert r.semantic.sequence==0 and result['actions']>=1
    asyncio.run(run())


def test_observation_fit_requires_reviewed_calibration_states_and_matching_prompt(tmp_path):
    labels=[];preds=[]
    for i,state in enumerate(STATES):
        for k in range(8):
            ident=f'{i}-{k}'
            labels.append({'id':ident,'schema':SCHEMA,'state':state,'reviewed':True,'annotator':'fixture','split':'calibration'})
            preds.append({'id':ident,'schema':SCHEMA,'features':[v+k*.01 for v in config()['listener']['observation_model']['means'][i]],
                          'model':config()['semantic']['model'],'prompt_sha256':prompt_digest()})
    a=tmp_path/'a.jsonl';p=tmp_path/'p.jsonl'
    def write():
        a.write_text('\n'.join(json.dumps(x) for x in labels),encoding='utf-8')
        p.write_text('\n'.join(json.dumps(x) for x in preds),encoding='utf-8')
    write();artifact=fit(a,p);model=tmp_path/'model.json';model.write_text(json.dumps(artifact),encoding='utf-8')
    c=config();load_model(c,model);assert c['listener']['observation_model']['status']=='fitted_on_calibration'
    artifact['prompt_sha256']='different';model.write_text(json.dumps(artifact),encoding='utf-8')
    with pytest.raises(ValueError,match='match'):load_model(c,model)
    labels[0]['split']='test';write()
    with pytest.raises(ValueError,match='calibration'):fit(a,p)
