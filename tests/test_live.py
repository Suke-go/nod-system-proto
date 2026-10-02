import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace
import urllib.request
import urllib.error
import wave
import pytest
np=pytest.importorskip('numpy')
from nod.audio.live import Segmenter,LiveInput,ASRJob
from nod.config import load_config
from nod.dashboard import Dashboard
from nod.runtime import Runtime
from nod.telemetry import replay
from nod.core.clock import VirtualClock
from nod.semantic.backend import MockBackend
from nod.semantic.coordinator import Coordinator
from nod.asr.stability import TranscriptTracker


def test_endpointing_preserves_preroll_and_never_includes_future_audio():
    s=Segmenter(partial_ms=200,silence_ms=100,max_ms=1000)
    quiet=np.zeros(320,dtype=np.float32); speech=np.full(320,.1,dtype=np.float32)
    for i in range(15): assert s.push(quiet,(i+1)*20,(i+1)*320)==(False,None)
    onset,job=s.push(speech,320,5120)
    assert onset and job and len(job.audio)==3520 and not job.final
    for i in range(5): onset,job=s.push(quiet,340+i*20,5440+i*320)
    assert job.final and job.source_ms==420 and len(job.audio)==5120
    onset,job=s.push(speech,440,7040)
    assert onset and s.uid==2


def test_continuous_speech_is_bounded():
    s=Segmenter(partial_ms=200,max_ms=400)
    for i in range(20): _,job=s.push(np.ones(320),20*(i+1),320*(i+1))
    assert job.final and len(job.audio)==6400 and not s.active


def test_empty_onset_clears_pending_semantic_request():
    cfg=load_config(); tracker=TranscriptTracker(cfg)
    c=Coordinator(cfg['semantic'],MockBackend(),VirtualClock(),lambda e:None)
    c.offer(tracker.update('old','hello',0,0,None,True),[])
    assert c.pending
    c.offer(tracker.update('new','',1,1,None,False),[])
    assert c.pending is None


def test_late_asr_is_discarded_after_new_utterance():
    async def scenario():
        started=threading.Event(); release=threading.Event(); events=[]
        class ASR:
            def infer(self,a): started.set(); release.wait(2); return {'text':'old'}
        runtime=SimpleNamespace(clock=SimpleNamespace(now_ms=lambda:100),emit=lambda *a:events.append(a))
        live=LiveInput(runtime,ASR(),None); live.current_uid='old'
        pool=ThreadPoolExecutor(1)
        task=asyncio.create_task(live._asr_worker(pool))
        try:
            live.offer(ASRJob('old',np.zeros(320),20,320,False))
            for _ in range(100):
                if started.is_set(): break
                await asyncio.sleep(.005)
            assert started.is_set()
            live.current_uid='new'; release.set(); await asyncio.sleep(.05)
            assert events==[]
        finally:
            release.set(); task.cancel(); await asyncio.gather(task,return_exceptions=True); pool.shutdown()
    asyncio.run(scenario())


def test_capture_discontinuity_stops_output():
    events=[]
    runtime=SimpleNamespace(emit=lambda *args:events.append(args))
    live=LiveInput(runtime,None,None)
    loop=SimpleNamespace(call_soon_threadsafe=lambda fn,*args:fn(*args))
    live._callback(loop)(np.zeros((320,1)),320,None,True)
    assert live.stop.is_set() and events[-1]==('input_health',{'healthy':False})


def test_early_timer_wake_never_emits_future_audio(monkeypatch):
    async def scenario():
        now=[0]; sleeps=[];events=[]
        async def early_sleep(delay):
            sleeps.append(delay);now[0]+=5
        runtime=SimpleNamespace(clock=SimpleNamespace(now_ms=lambda:now[0]),
                                emit=lambda kind,data:events.append((now[0],data)))
        live=LiveInput(runtime,None,None)
        live.capture_queue.put_nowait((np.ones(320,dtype=np.float32),20,320,False))
        live.capture_queue.put_nowait(None)
        monkeypatch.setattr('nod.audio.live.asyncio.sleep',early_sleep)
        await live._consume()
        assert len(sleeps)==4 and events
        assert all(data['source_ms']<=at_ms for at_ms,data in events)
        assert live.pending.end_sample==320 and live.pending.source_ms==20
    asyncio.run(scenario())


@pytest.mark.parametrize('responsive',[False,True])
def test_live_wav_drives_engine_without_real_models_or_api(tmp_path,responsive):
    path=tmp_path/'audio.wav'
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes(np.full(16000,4000,dtype='<i2').tobytes())
    class ASR:
        def infer(self,a): return {'text':'This is a test.'}
    class Timing:
        def infer(self,a): return .95
    async def scenario():
        cfg=load_config();cfg['output']['port']=0
        if responsive:
            from nod.responsive import configure
            cfg=configure(cfg)
        runtime=Runtime(cfg,'mock',tmp_path/'session.jsonl')
        live=LiveInput(runtime,ASR(),Timing(),wav=path,duration=2,partial_ms=200)
        return await runtime.run(live.run)
    result=asyncio.run(scenario())
    records=[json.loads(line) for line in (tmp_path/'session.jsonl').read_text().splitlines()]
    events=[r['event'] for r in records[1:]]
    assert any(e['kind']=='asr' and e['data']['text'] for e in events)
    assert any(e['kind']=='semantic_result' for e in events)
    if responsive:
        assert any(e['kind']=='asr_job' for e in events)
        assert any(e['kind']=='asr_boundary' for e in events)
    assert result['actions']>=1
    assert replay(tmp_path/'session.jsonl')['matched']


def test_dashboard_stop_requires_origin_and_token():
    stopped=[]; d=Dashboard('ja','mock',0,lambda:stopped.append(True));d.start()
    url=f'http://127.0.0.1:{d.port}'
    try:
        d.publish(text='<script>unsafe</script>')
        with urllib.request.urlopen(url+'/api/state') as r:
            state=json.load(r);assert state['text']=='<script>unsafe</script>'
            assert 'token' not in state and 'api_key' not in state
        request=urllib.request.Request(url+'/api/stop',method='POST')
        with pytest.raises(urllib.error.HTTPError) as exc: urllib.request.urlopen(request)
        assert exc.value.code==403 and not stopped
        request=urllib.request.Request(url+'/api/stop',method='POST',headers={'Origin':url,'X-Control-Token':d.token})
        with urllib.request.urlopen(request) as r: assert r.status==200
        assert stopped==[True]
        request=urllib.request.Request(url+'/api/state',headers={'Host':'attacker.invalid'})
        with pytest.raises(urllib.error.HTTPError) as exc: urllib.request.urlopen(request)
        assert exc.value.code==403
    finally: d.close()


def test_dotenv_loads_only_key_without_executing_text(tmp_path,monkeypatch):
    from nod.secrets import load_api_key_file
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    path=tmp_path/'.env';path.write_text('UNRELATED=anything\nTYPESAFE_API_KEY="fake-unit-test"\n')
    load_api_key_file(path)
    import os
    assert os.environ['TYPESAFE_API_KEY']=='fake-unit-test' and 'UNRELATED' not in os.environ
    path.write_text('TYPESAFE_API_KEY=do-not-overwrite\n');load_api_key_file(path)
    assert os.environ['TYPESAFE_API_KEY']=='fake-unit-test'


def test_dashboard_render_reports_require_origin_token_and_bounded_schema():
    reported=[];d=Dashboard('ja','mock',0,rendered=reported.append);d.start()
    url=f'http://127.0.0.1:{d.port}'
    try:
        def post(body,auth=True):
            headers={'Origin':url,'X-Control-Token':d.token} if auth else {}
            return urllib.request.urlopen(urllib.request.Request(url+'/api/rendered',data=json.dumps(body).encode(),headers=headers,method='POST'))
        with pytest.raises(urllib.error.HTTPError) as exc:post({'action_id':'a1'},False)
        assert exc.value.code==403 and not reported
        with pytest.raises(urllib.error.HTTPError) as exc:post({'action_id':'a1','text':'unrelated'})
        assert exc.value.code==400 and not reported
        with post({'action_id':'a1'}) as response:assert response.status==200
        assert reported==[{'action_id':'a1'}]
    finally:d.close()
