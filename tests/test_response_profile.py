import pytest
from nod.config import load_config,validate_config
from nod.live_session import live_config
from nod.policy.selector import ExpectedUtilityPolicy
from nod.core.engine import Engine
from nod.core.events import Event
from nod.telemetry import SessionLog,replay


def test_responsive_profile_changes_timing_without_changing_semantics_or_freshness():
    old=live_config(load_config(),'ja',0,response='conservative')
    new=live_config(load_config(),'ja',0,response='responsive')
    assert new['semantic']==old['semantic']
    assert new['belief']['observation_rule']=='posterior_blend'
    assert old['belief'].get('observation_rule','likelihood_product')=='likelihood_product'
    assert new['timing']['open_threshold']==.4
    assert new['motor']['cooldown_after_completion_ms']==1800
    assert old['timing']['open_threshold']==.65
    assert new['policy']['minimum_asr_reliability']==old['policy']['minimum_asr_reliability']
    assert new['policy']['opportunity_weighting']=='gate_only'


def test_gate_conditioned_policy_uses_semantics_but_does_not_double_discount_timing():
    cfg=load_config()['policy'];legacy=ExpectedUtilityPolicy(cfg)
    responsive=ExpectedUtilityPolicy(dict(cfg,opportunity_weighting='gate_only'))
    b=(.1,.85,.03,.02)
    assert legacy.choose(.45,b).function=='none'
    assert responsive.choose(.45,b).function=='continuer'
    assert responsive.choose(.45,(.9,.07,.02,.01)).function=='none'
    assert responsive.choose(0,b).function=='none'
    assert responsive.choose(.5,(0,0,0,1),specific=False).function!='empathic'


def test_custom_configuration_is_preserved_and_conflicts_are_explicit():
    custom=load_config();custom['timing']['open_threshold']=.77
    result=live_config(custom,'en',0,custom=True)
    assert result['timing']['open_threshold']==.77
    with pytest.raises(ValueError,match='either'):
        live_config(load_config(),'ja',0,custom=True,response='responsive')
    bad=load_config();bad['policy']['opportunity_weighting']='invalid'
    with pytest.raises(ValueError,match='weighting'):validate_config(bad)


def test_responsive_still_requires_fresh_audio_and_semantics_and_replays(tmp_path):
    cfg=live_config(load_config(),'ja',0,response='responsive')
    engine=Engine(cfg);path=tmp_path/'responsive.jsonl';log=SessionLog(path,cfg,'mock')
    def send(kind,at,data):
        event=Event(kind,at,data);records=engine.process(event);log.event(event,records)
        return [r for r in records if r['kind']=='command']
    send('controller_ready',0,{'neutral_expression':True})
    send('asr',0,{'utterance_id':'u1','text':'説明です','source_ms':0,'final':True})
    assert not send('opportunity',100,{'score':.45,'source_ms':100})
    assert not send('opportunity',200,{'score':.45,'source_ms':200})  # no semantics
    snapshot=engine.transcripts.current.to_dict()
    cmds=send('semantic_result',220,{'request':{'seq':1,'snapshot':snapshot,'dispatched_ms':200},
        'result':{'probabilities':[.001,.995,.003,.001]}})
    assert len(cmds)==1
    action=cmds[0]['command']['action_id']
    send('feedback',230,{'action_id':action,'status':'started'})
    send('feedback',600,{'action_id':action,'status':'completed'})
    # Sustained score never creates repeated output even after cooldown.
    for now in range(700,2701,100):
        assert not send('opportunity',now,{'score':.45,'source_ms':now})
    for now in (2800,2900,3000):send('opportunity',now,{'score':.1,'source_ms':now})
    send('opportunity',3100,{'score':.45,'source_ms':3100})
    assert not send('opportunity',3200,{'score':.45,'source_ms':3200})  # semantic expired
    log.close()
    assert replay(path)['matched']
