from dataclasses import replace
import pytest
from nod.config import load_config
from nod.live_session import live_config
from nod.belief.filter import BeliefFilter
from nod.core.engine import Engine
from nod.core.events import Event
from nod.policy.selector import ExpectedUtilityPolicy
from nod.telemetry import SessionLog,replay


def setup_engine():
    cfg=live_config(load_config(),'ja',0,response='responsive')
    engine=Engine(cfg)
    engine.process(Event('controller_ready',0,{'neutral_expression':True}))
    engine.process(Event('asr',0,{'utterance_id':'u','text':'話の結末です','final':True,'source_ms':0}))
    engine.process(Event('opportunity',100,{'score':.1,'source_ms':100}))
    return cfg,engine


def result(engine,p,now=120,seq=1,snapshot=None):
    return Event('semantic_result',now,{'request':{'seq':seq,'dispatched_ms':now-10,
        'snapshot':snapshot or engine.transcripts.current.to_dict()},
        'result':{'probabilities':p,'model':'fixture','confidence':.9}})


def commands(records):return [r for r in records if r['kind']=='command']


def test_jev_distribution_is_not_multiplied_by_another_class_prior():
    cfg=live_config(load_config(),'ja',0,response='responsive')['belief'];p=(.02,.03,.08,.87)
    f=BeliefFilter(cfg)
    assert f.observe(p,1,0)==pytest.approx(p)
    assert f.observe(p,1,0)==pytest.approx(p)  # overlapping requests never compound
    f=BeliefFilter(cfg)
    assert f.observe(p,.5,0)==pytest.approx(tuple((a+b)/2 for a,b in zip(cfg['initial'],p)))
    assert f.observe(p,0,0)==pytest.approx(cfg['initial'])


@pytest.mark.parametrize('p,function,action',[
    ((.02,.03,.9,.05),'understanding','STRONG_NOD'),
    ((.02,.03,.05,.9),'empathic','EMPATHIC_EXPRESSION')])
def test_clear_final_semantics_can_acknowledge_without_a_vap_peak(p,function,action):
    _,e=setup_engine();cmd=commands(e.process(result(e,p)))[0]
    assert cmd['function']==function and cmd['command']['action']==action
    assert cmd['trigger']=='utterance_end' and cmd['prominence_degraded']
    assert cmd['command']['intensity']<.6  # meaning determines gesture, not prominence


@pytest.mark.parametrize('blocked',['partial','unfinalized_request','low_reliability','uncertain','no_bc','continuer','stale_audio','input_stopped','disconnected'])
def test_endpoint_is_not_an_unconditional_silence_fallback(blocked):
    _,e=setup_engine();p=(.02,.03,.05,.9);snapshot=e.transcripts.current.to_dict();now=120
    if blocked=='partial':e.transcripts.current=replace(e.transcripts.current,final=False)
    if blocked=='unfinalized_request':snapshot['final']=False
    if blocked=='low_reliability':snapshot['reliability']=.4
    if blocked=='uncertain':p=(.1,.1,.4,.4)
    if blocked=='no_bc':p=(.9,.04,.03,.03)
    if blocked=='continuer':p=(.02,.92,.03,.03)
    if blocked=='stale_audio':now=400
    if blocked=='input_stopped':e.process(Event('input_health',110,{'healthy':False}))
    if blocked=='disconnected':e.process(Event('controller_disconnected',110))
    assert not commands(e.process(result(e,p,now=now,snapshot=snapshot)))


def test_specific_ack_occurs_once_per_utterance_even_after_cooldown():
    _,e=setup_engine();p=(.02,.03,.05,.9)
    cmd=commands(e.process(result(e,p)))[0]['command']
    e.process(Event('feedback',130,{'action_id':cmd['action_id'],'status':'started'}))
    e.process(Event('feedback',730,{'action_id':cmd['action_id'],'status':'completed'}))
    e.process(Event('asr',2600,{'utterance_id':'u','text':'話の結末です','final':True,'source_ms':2600}))
    e.process(Event('opportunity',2700,{'score':.8,'source_ms':2700}))
    assert not commands(e.process(result(e,p,now=2710,seq=2)))
    assert not commands(e.process(Event('opportunity',2800,{'score':.8,'source_ms':2800})))


def test_new_specific_log_replays_and_legacy_mapping_remains_available(tmp_path):
    cfg=live_config(load_config(),'ja',0,response='responsive');e=Engine(cfg);log=SessionLog(tmp_path/'s.jsonl',cfg,'mock')
    events=[Event('controller_ready',0,{'neutral_expression':True}),
            Event('asr',0,{'utterance_id':'u','text':'説明です','source_ms':0,'final':True}),
            Event('opportunity',100,{'score':.1,'source_ms':100})]
    for event in events:log.event(event,e.process(event))
    event=result(e,(.01,.02,.95,.02));log.event(event,e.process(event));log.close()
    assert replay(tmp_path/'s.jsonl')['matched']
    old=ExpectedUtilityPolicy(load_config()['policy'])
    new=ExpectedUtilityPolicy(cfg['policy'])
    assert old.embody('understanding',.25)[0]=='SMALL_NOD'
    assert new.embody('understanding',.25)[0]=='STRONG_NOD'
    assert old.embody('continuer',.25)==new.embody('continuer',.25)


@pytest.mark.parametrize('profile',['responsive','bayes','bayes-no-history','direct'])
def test_specific_commands_cross_real_websocket_with_fixture_semantics(tmp_path,profile):
    import asyncio
    from nod.runtime import Runtime
    class Backend:
        async def evaluate(self,state):
            p=(.01,.01,.97,.01) if '説明' in state['stable_transcript'] else (.01,.01,.01,.97)
            return {'probabilities':p,'model':'fixture','confidence':1.}
        async def close(self):pass
    async def scenario():
        runtime=Runtime(live_config(load_config(),'ja',0,response=profile),'mock',tmp_path/'wire.jsonl')
        runtime.semantic.backend=Backend()
        async def feeder():
            for uid,text in [('u1','説明を終えました'),('u2','ようやく合格できました')]:
                runtime.emit('asr',{'utterance_id':uid,'text':text,'final':True,'final_reason':'silence','source_ms':runtime.clock.now_ms()})
                for _ in range(28):
                    runtime.emit('opportunity',{'score':.1,'source_ms':runtime.clock.now_ms()})
                    await asyncio.sleep(.1)
        result=await runtime.run(feeder)
        return result,runtime.commands
    result,records=asyncio.run(scenario())
    assert result['functions']==['understanding','empathic']
    assert [r['command']['action'] for r in records]==['STRONG_NOD','EMPATHIC_EXPRESSION']
    assert all(r['trigger']=='utterance_end' for r in records)
    assert result['motor_state']!='FAULT' and replay(tmp_path/'wire.jsonl')['matched']
