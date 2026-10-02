"""Paired local CPU benchmark on authored, referenced speech; never captures audio."""
import argparse, json, time, wave, unicodedata
from pathlib import Path
import numpy as np
from nod.audio.models import LocalASR


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1): nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]


def units(text,lang):
    text=unicodedata.normalize('NFKC',text).lower()
    text=''.join(c if not unicodedata.category(c).startswith(('P','S')) else ' ' for c in text)
    return list(''.join(text.split())) if lang=='ja' else text.split()


def main():
    p=argparse.ArgumentParser();p.add_argument('--audio',required=True);p.add_argument('--models',required=True)
    p.add_argument('--sensevoice',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    import sherpa_onnx
    root=Path(args.audio); refs=json.loads((root/'authored-text.json').read_text(encoding='utf-8'))
    rows=[]
    for lang in ('ja','en'):
        clips=[]
        for i,reference in enumerate(refs[lang]):
            with wave.open(str(root/f'{lang}-{i}.wav')) as f:
                assert f.getframerate()==16000 and f.getnchannels()==1 and f.getsampwidth()==2
                audio=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
            clips.append((i,reference,audio))
        for name,threads in [('whisper',4),('sensevoice',1),('sensevoice',2),('sensevoice',4)]:
            if name=='whisper':
                model=LocalASR(args.models,lang);infer=lambda x:model.infer(x)['text']
            else:
                model=sherpa_onnx.OfflineRecognizer.from_sense_voice(model=str(Path(args.sensevoice)/'model.int8.onnx'),
                    tokens=str(Path(args.sensevoice)/'tokens.txt'),num_threads=threads,language=lang,use_itn=True)
                def infer(x):
                    stream=model.create_stream();stream.accept_waveform(16000,x);model.decode_stream(stream)
                    return stream.result.text
            infer(np.zeros(16000,dtype=np.float32))
            for repeat in range(2):
                for i,reference,audio in clips:
                    start=time.perf_counter();text=infer(audio);ms=(time.perf_counter()-start)*1000
                    ref,hyp=units(reference,lang),units(text,lang)
                    row=dict(language=lang,backend=name,threads=threads,repeat=repeat,clip=i,
                        duration_ms=len(audio)/16,inference_ms=ms,reference=reference,text=text,
                        errors=distance(ref,hyp),reference_units=len(ref),metric='CER' if lang=='ja' else 'WER')
                    rows.append(row)
            # Partial outputs are separate from full-reference accuracy scores.
            for seconds in (.8,1.6,2.4,3.2):
                start=time.perf_counter();text=infer(clips[1][2][:round(seconds*16000)])
                rows.append(dict(language=lang,backend=name,threads=threads,partial_seconds=seconds,
                    inference_ms=(time.perf_counter()-start)*1000,text=text))
            print(json.dumps({'language':lang,'backend':name,'threads':threads,
                'median_ms':float(np.median([r['inference_ms'] for r in rows if r['language']==lang and r['backend']==name and r['threads']==threads and 'repeat' in r]))}),flush=True)
            del model
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps({'scope':'six authored synthetic clips, CPU standalone; not natural speech validation','rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
