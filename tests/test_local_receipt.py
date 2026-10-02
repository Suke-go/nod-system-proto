import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
from types import SimpleNamespace
import pytest
from nod.config import load_config,validate_config
from nod.responsive import configure
from nod.policy.incremental import IncrementalPolicy,filler_only
from nod.semantic.sensor import fixture,SCHEMA,compatibility
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.audio.live import LiveInput,ASRJob


def config():return configure(load_config())
def send(e,kind,t,**data):return e.process(Event(kind,t,data))
def commands(rows):return [r for r in rows if r['kind']=='command']
def semantic(e,q,t,seq=1,snapshot=None):
    return send(e,'semantic_result',t,request={'seq':seq,'dispatched_ms':t-10,
        'snapshot':snapshot or e.units.current.to_dict()},result={'schema':SCHEMA,
        'observation':q,'probabilities':compatibility(q),'model':'fixture'})


def test_local_receipt_retains_bayesian_requirements_without_duplicate_asr_gate():
    p=IncrementalPolicy(config()['listener']['policy'])
    # Approximate recorded robot state; ASR .616 was already used to obtain it.
    b=[.21,.09,.68,.005,.005,.01]
    q=fixture('neutral',resolved=.91,completion=.66,demand=.11)
    f,_,ctx=p.choose(b,[],0,'u',q,.01,False,False,evidence_text='説明するわけですよ。')
    assert f=='understanding' and ctx['local_receipt_supported']
    old=deepcopy(p.cfg);old.pop('local_receipt')
    f,_,ctx=IncrementalPolicy(old).choose(b,[],0,'u',q,.01,False,False)
    assert f=='none' and 'understanding' not in ctx['allowed']


@pytest.mark.parametrize('change',['incomplete','question','unresolved','weak_belief','retained'])
def test_local_receipt_never_bypasses_semantic_requirements(change):
    p=IncrementalPolicy(config()['listener']['policy'])
    b=[.05,.05,.85,.01,.02,.02];q=fixture('neutral',resolved=.9,completion=.9,demand=.1)
    if change=='incomplete':q['completion']=.2
    if change=='question':q['response_demand']=.95
    if change=='unresolved':q['interpretability']={'unresolved':.9,'resolved':.1}
    if change=='weak_belief':b=[.5,.4,.07,.01,.01,.01]
    f,_,ctx=p.choose(b,[],0,'u',q,.01,False,False,retained=change=='retained')
    assert f!='understanding' and not ctx['local_receipt_supported']


@pytest.mark.parametrize('text',['あのう。','えーと…','う。','で？','と。','あの、えっと','um...','uh, well','so…'])
def test_filler_only_waits_even_with_high_acoustic_readiness(text):
    p=IncrementalPolicy(config()['listener']['policy'])
    f,_,ctx=p.choose([.01,.95,.01,.01,.01,.01],[],0,'u',None,.99,True,False,evidence_text=text)
    assert f=='none' and ctx['reason']=='filler_only_wait'


@pytest.mark.parametrize('text',['はい。','うれしい。','しんどい。','あの、実験が終わりました。',
                                'Well, the experiment worked.','I am sad.','yes','some data'])
def test_filler_guard_does_not_remove_substantive_content(text):assert not filler_only(text)


def prepared_engine():
    e=create_engine(config())
    send(e,'controller_ready',0,neutral_expression=True,supported_expressions=['neutral','attentive','warm','concerned'])
    send(e,'asr',0,utterance_id='u',text='The experiment worked.',source_ms=0,confidence=.99,final=True)
    send(e,'opportunity',10,score=.01,source_ms=10)
    return e


def test_receipt_upgrade_waits_for_motor_completion_and_new_semantics():
    e=prepared_engine()
    a=commands(send(e,'opportunity',100,score=.99,source_ms=100))[0]['command']['action_id']
    send(e,'feedback',110,action_id=a,status='started')
    assert not commands(semantic(e,fixture('neutral'),120))
    send(e,'feedback',400,action_id=a,status='completed')
    assert not commands(send(e,'opportunity',500,score=.01,source_ms=500))
    rows=commands(send(e,'opportunity',550,score=.01,source_ms=550))
    assert len(rows)==1 and rows[0]['function']=='understanding'
    a=rows[0]['command']['action_id']
    send(e,'feedback',560,action_id=a,status='started')
    send(e,'feedback',1000,action_id=a,status='completed')
    assert not commands(send(e,'opportunity',2850,score=.99,source_ms=2850))


def test_repair_and_new_unit_invalidate_receipt_evidence():
    e=prepared_engine();old=e.units.current.to_dict()
    send(e,'asr',100,utterance_id='u',text='The experiment did not work.',source_ms=100,confidence=.99,final=True)
    assert not commands(semantic(e,fixture('neutral'),120,snapshot=old))
    assert e.semantic is None
    send(e,'asr',130,utterance_id='next',text='',source_ms=130,final=False)
    assert not commands(semantic(e,fixture('neutral'),140,seq=2,snapshot=old))
    assert e.semantic is None


def test_boundary_does_not_create_semantics_or_claim_max_duration_is_silence():
    e=prepared_engine()
    send(e,'asr_boundary',20,utterance_id='u',source_ms=20,final_reason='max_duration')
    assert e.semantic is None
    assert e.decision_context['readiness']<.2
    send(e,'asr_boundary',30,utterance_id='u',source_ms=30,final_reason='silence')
    assert e.semantic is None and e.belief.summary(30)['understanding']<.65
    send(e,'asr',40,utterance_id='next',text='',source_ms=40,final=False)
    assert e.asr_boundary is None
    send(e,'asr_boundary',50,utterance_id='u',source_ms=30,final_reason='silence')
    assert e.asr_boundary is None


def test_lifecycle_events_never_advance_or_resurrect_listener_state():
    e=prepared_engine();before=e.belief.posterior(20)
    assert send(e,'asr_job',20,utterance_id='old',source_ms=0,final=True,state='archived',text='I am happy')==[]
    assert e.transcripts.current.utterance_id=='u' and e.semantic is None
    assert e.belief.posterior(20)==before


def test_final_queue_survives_onset_and_late_final_is_archived_not_current():
    np=pytest.importorskip('numpy')
    async def run():
        started=threading.Event();release=threading.Event();events=[];calls=[]
        class ASR:
            def infer(self,a):
                calls.append(int(a[0]))
                if len(calls)==1:started.set();release.wait(2)
                return {'text':str(int(a[0]))}
        runtime=SimpleNamespace(cfg=config(),clock=SimpleNamespace(now_ms=lambda:1000),
            emit=lambda k,d:events.append((k,d)))
        live=LiveInput(runtime,ASR(),None);live.current_uid='old'
        pool=ThreadPoolExecutor(1);task=asyncio.create_task(live._asr_worker(pool))
        def job(uid,n,final=False):return ASRJob(uid,np.full(320,n),n*20,n*320,final,'silence' if final else None)
        try:
            live.offer(job('old',1))
            for _ in range(200):
                if started.is_set():break
                await asyncio.sleep(.005)
            assert started.is_set()
            live.offer(job('old',2));live.offer(job('old',3,True))
            live._onset('new',80,1280);live.offer(job('new',4))
            release.set()
            for _ in range(200):
                if any(k=='asr' and d.get('text')=='4' for k,d in events):break
                await asyncio.sleep(.005)
            assert calls==[1,3,4]
            assert [(d['utterance_id'],d['text']) for k,d in events if k=='asr' and d['text']]==[('new','4')]
            archived=[d for k,d in events if k=='asr_job' and d['state']=='archived']
            assert len(archived)==1 and archived[0]['text']=='3' and archived[0]['final']
            assert any(d.get('reason')=='superseded_by_final' for k,d in events)
        finally:
            release.set();task.cancel();await asyncio.gather(task,return_exceptions=True);pool.shutdown()
    asyncio.run(run())


def test_final_queue_overload_is_bounded_and_explicit():
    events=[];runtime=SimpleNamespace(cfg=config(),emit=lambda k,d:events.append((k,d)))
    live=LiveInput(runtime,None,None)
    for n in range(4):live.offer(ASRJob(str(n),[],n*20,n*320,True,'silence'))
    assert len(live.final_jobs)==2
    drops=[d for k,d in events if d.get('reason')=='final_queue_overload']
    assert [d['utterance_id'] for d in drops]==['0','1']


@pytest.mark.parametrize('value',[0,True,5,1.5])
def test_final_queue_configuration_is_bounded(value):
    c=config();c['asr']['final_queue_capacity']=value
    with pytest.raises(ValueError):validate_config(c)
