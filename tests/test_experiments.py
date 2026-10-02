import asyncio
import json
import wave
from pathlib import Path
import pytest
np = pytest.importorskip('numpy')
from nod.config import load_config
from nod.audio.recording import load_wave
from nod.experiments.evaluate import load_cases,summarize,evaluate_cases
from nod.experiments.timeline import build_timeline,load_timeline
from nod.experiments.report import analyze_session
from nod.experiments.common import save_json
from nod.semantic.backend import MockBackend,SemanticError


def case(label='continuer',status='draft'):
    return {'id':'c1','state':{'stable_transcript':'それで','current_partial':'','recent_context':[]},
            'reference_label':label,'label_status':status}


def test_draft_labels_never_reported_as_ground_truth():
    row = {'status':'ok','label_status':'draft','reference_label':'continuer','prediction':'continuer',
           'probabilities':[.1,.7,.1,.1],'latency_ms':10}
    assert summarize([row])['accuracy'] is None
    row['label_status'] = 'reviewed'
    report = summarize([row])
    assert report['accuracy'] == 1
    assert report['brier_multiclass_sum'] == pytest.approx(.12)
    assert report['negative_log_likelihood'] == pytest.approx(-np.log(.7))


def test_evaluation_stops_on_error_and_writes_partial_report(tmp_path):
    class Failed:
        async def evaluate(self,state): raise SemanticError('http_401',permanent=True)
        async def close(self): pass
    dataset = tmp_path/'cases.jsonl'
    dataset.write_text(json.dumps(case())+'\n'+json.dumps(dict(case(),id='c2'))+'\n',encoding='utf-8')
    result = asyncio.run(evaluate_cases(load_config(),dataset,tmp_path/'run',limit=2,backend=Failed()))
    assert result['attempted'] == 1 and result['not_attempted'] == 1 and result['failed'] == 1
    assert (tmp_path/'run'/'summary.json').is_file()


def test_mock_evaluation_is_marked_not_model_quality(tmp_path):
    dataset = tmp_path/'cases.jsonl'; dataset.write_text(json.dumps(case(status='reviewed'))+'\n',encoding='utf-8')
    result = asyncio.run(evaluate_cases(load_config(),dataset,tmp_path/'run',limit=1))
    assert result['succeeded'] == 1 and not result['model_quality_measured']
    with pytest.raises(FileExistsError):
        asyncio.run(evaluate_cases(load_config(),dataset,tmp_path/'run',limit=1))


def test_cases_reject_duplicate_ids_and_state_leak(tmp_path):
    dataset = tmp_path/'cases.jsonl'
    dataset.write_text(json.dumps(case())+'\n'+json.dumps(case())+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='unique'): load_cases(dataset)
    item = case(); item['state']['opportunity'] = .9
    dataset.write_text(json.dumps(item)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='state'): load_cases(dataset)


def test_causal_audio_prefixes_and_no_tail_padding_or_overlap(tmp_path):
    class ASR:
        def __init__(self): self.lengths = []
        def infer(self,samples):
            self.lengths.append(len(samples)); return {'text':'音声','words':[]}
    class Timing:
        def __init__(self): self.chunks=[]
        def infer(self,samples): self.chunks.append(samples.copy()); return .2
    asr,timing = ASR(),Timing()
    audio = np.arange(16500,dtype=np.float32)
    result = build_timeline(audio,asr,timing)
    assert asr.lengths == [16000,16500]
    assert np.array_equal(np.concatenate(timing.chunks),audio[:16000])
    for event in result['events']:
        assert event['data']['audio_end_sample']/16 <= event['data']['source_ms'] <= event['at_ms']
    save_json(tmp_path/'timeline.json',result)
    assert load_timeline(tmp_path/'timeline.json')['events'] == result['events']


def test_timeline_rejects_lookahead_and_internal_control_events(tmp_path):
    base = {'schema_version':1,'kind':'nod_audio_timeline','events':[
        {'kind':'asr','at_ms':100,'data':{'text':'future','utterance_id':'u','source_ms':200,'audio_end_sample':3200}}]}
    save_json(tmp_path/'bad.json',base)
    with pytest.raises(ValueError,match='before'): load_timeline(tmp_path/'bad.json')
    base['events'][0]['kind'] = 'controller_ready'
    save_json(tmp_path/'control.json',base)
    with pytest.raises(ValueError,match='external'): load_timeline(tmp_path/'control.json')


def test_wave_contract_rejects_implicit_resampling(tmp_path):
    path = tmp_path/'test.wav'
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000); f.writeframes(bytes(16000))
    with pytest.raises(ValueError,match='16kHz'): load_wave(path)


def test_record_is_never_started_by_device_enumeration(monkeypatch):
    sd = pytest.importorskip('sounddevice')
    from nod.audio.recording import devices
    monkeypatch.setattr(sd,'query_devices',lambda:[{'name':'test','max_input_channels':1,'default_samplerate':16000}])
    monkeypatch.setattr(sd,'rec',lambda *args,**kwargs:pytest.fail('Unexpected recording'))
    assert devices()[0]['name'] == 'test'


def test_processing_stall_is_not_saved_as_a_usable_timeline(monkeypatch):
    from nod.experiments import timeline
    class Timing:
        def infer(self,samples): return .2
    times = iter([0,6])
    monkeypatch.setattr(timeline.time,'perf_counter',lambda:next(times))
    with pytest.raises(RuntimeError,match='5s behind'):
        build_timeline(np.zeros(1600),None,Timing())


def test_session_analysis_reports_missing_completion(tmp_path):
    path = tmp_path/'session.jsonl'
    records = [{'record':'header','backend':'mock','config':{}},
        {'ingest_seq':1,'event':{'kind':'tick','at_ms':100,'data':{}},'derived':[
            {'kind':'command','command':{'action_id':'a1','at_ms':100},'function':'continuer','prominence_degraded':True}]},
        {'ingest_seq':2,'event':{'kind':'feedback','at_ms':120,'data':{'action_id':'a1','status':'started'}},'derived':[]}]
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
    report = analyze_session(path)
    assert report['actions_without_completion'] == ['a1']
    assert report['command_to_started_ms']['p50'] == 20
    assert not report['measures_real_animation_onset']
