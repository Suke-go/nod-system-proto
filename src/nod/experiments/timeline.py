import asyncio
import math
import time
from pathlib import Path
from nod.config import probability, strict_json
from nod.experiments.common import quantiles, save_json, sha256


def build_timeline(samples, asr, timing, partial_ms=1000):
    """Causal-prefix offline experiment. Independent serialized worker clocks, not a live latency benchmark."""
    if not isinstance(partial_ms,int) or partial_ms < 500 or partial_ms % 100:
        raise ValueError('ASR partial interval must be a multiple of 100ms and >=500ms')
    length = len(samples); duration_ms = math.ceil(length/16)
    rows, words, asr_times, timing_times = [], [], [], []
    asr_free = timing_free = 0
    # Both models see only data up to each source timestamp. ASR jobs have no backlog.
    for start in range(0,length,1600):
        end = min(start+1600,length); source = math.ceil(end/16)
        chunk = samples[start:end]
        if len(chunk) == 1600:
            begin = time.perf_counter()
            score = timing.infer(chunk)
            elapsed = max(1,math.ceil((time.perf_counter()-begin)*1000)); timing_times.append(elapsed)
            timing_free = max(source,timing_free)+elapsed
            if timing_free-source > 5000:
                raise RuntimeError('Timing worker fell over 5s behind audio; retry when the PC is idle. No usable timeline was saved.')
            rows.append({'kind':'opportunity','at_ms':timing_free,
                         'data':{'source_ms':source,'score':probability(score),'audio_end_sample':end}})
        final = end == length
        if (source % partial_ms == 0 and source >= asr_free) or final:
            begin = time.perf_counter()
            result = asr.infer(samples[:end])
            elapsed = max(1,math.ceil((time.perf_counter()-begin)*1000)); asr_times.append(elapsed)
            asr_free = max(source,asr_free)+elapsed
            if asr_free-source > 5000:
                raise RuntimeError('ASR worker fell over 5s behind audio; reduce load or shorten the clip. No usable timeline was saved.')
            rows.append({'kind':'asr','at_ms':asr_free,'data':{
                'utterance_id':'u1','text':result['text'],'final':final,
                'source_ms':source,'audio_end_sample':end}})
            if final:
                words = result.get('words',[])
    rows.sort(key=lambda row:row['at_ms'])
    return {'schema_version':1,'kind':'nod_audio_timeline','duration_ms':duration_ms,
            'timing_mode':'causal_prefix_offline_measured_independent_workers',
            'prominence':'missing_default_strength','events':rows,'final_words':words,
            'latency_ms':{'asr':quantiles(asr_times),'timing':quantiles(timing_times)},
            'asr_partial_interval_ms':partial_ms}


def load_timeline(path):
    timeline = strict_json(Path(path).read_text(encoding='utf-8-sig'))
    if timeline.get('schema_version') != 1 or timeline.get('kind') != 'nod_audio_timeline':
        raise ValueError('Unsupported audio timeline')
    previous = -1
    for row in timeline['events']:
        at, data = row['at_ms'],row['data']
        if isinstance(at,bool) or not isinstance(at,int) or at < previous:
            raise ValueError('Invalid timeline order')
        previous = at
        if row['kind'] not in ('asr','opportunity','prominence','input_health'):
            raise ValueError('Timeline may only contain external observations')
        if row['kind'] != 'input_health':
            source = data['source_ms']
            if isinstance(source,bool) or not isinstance(source,(int,float)) or not math.isfinite(source) or not 0 <= source <= at:
                raise ValueError('Observation cannot arrive before its source audio')
            end = data.get('audio_end_sample')
            if not isinstance(end,int) or isinstance(end,bool) or end < 0 or end/16 > source:
                raise ValueError('Audio sample and source timestamp mismatch')
        if row['kind'] == 'opportunity': probability(data['score'])
        elif row['kind'] == 'prominence': probability(data['strength'])
        elif row['kind'] == 'asr':
            if not isinstance(data['text'],str) or len(data['text']) > 2000 or not isinstance(data['utterance_id'],str) or not data['utterance_id']:
                raise ValueError('Invalid ASR observation')
    if not timeline['events']:
        raise ValueError('Empty audio timeline')
    return timeline


async def feed_timeline(runtime, timeline):
    origin = runtime.clock.now_ms()
    for row in timeline['events']:
        await asyncio.sleep(max(0,(origin+row['at_ms']-runtime.clock.now_ms())/1000))
        data = dict(row['data'])
        if 'source_ms' in data: data['source_ms'] += origin
        runtime.emit(row['kind'],data)
    # Let the last semantic job resolve, then close timing input before shutdown.
    await asyncio.sleep(1.3)
    runtime.emit('input_health',{'healthy':False})


def prepare_audio(wave_path, model_dir, output_path, partial_ms=1000):
    from nod.audio.recording import load_wave
    from nod.audio.models import LocalASR, LocalTiming, verify_assets
    if Path(output_path).exists(): raise FileExistsError(output_path)
    samples = load_wave(wave_path)
    manifest = verify_assets(model_dir)
    asr, timing = LocalASR(model_dir), LocalTiming(model_dir)
    result = build_timeline(samples,asr,timing,partial_ms)
    result.update(audio_sha256=sha256(wave_path),models=manifest)
    save_json(output_path,result)
    return {'timeline':str(Path(output_path).resolve()),'events':len(result['events']),
            'latency_ms':result['latency_ms'],'duration_ms':result['duration_ms']}
