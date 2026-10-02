"""Typed semantic evaluation, calibration and offline controlled replays."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from nod.config import strict_json,validate_config
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.semantic.backend import JevBackend,MockBackend,SemanticError
from nod.semantic.frames import SCHEMA,FRAME_ORDER,frame_distribution,marginals
from nod.belief.interaction import temper


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def metrics(rows):
    if not rows:return None
    nll=brier=correct=0.
    confusion=[[0]*len(FRAME_ORDER) for _ in FRAME_ORDER]
    for row in rows:
        p=frame_distribution(row['frames']);y=FRAME_ORDER.index(row['label'])
        best=max(range(len(p)),key=p.__getitem__)
        correct+=best==y;nll-=math.log(max(p[y],1e-12))
        brier+=sum((value-(i==y))**2 for i,value in enumerate(p))
        confusion[y][best]+=1
    return {'n':len(rows),'accuracy':correct/len(rows),'nll':nll/len(rows),
            'brier_multiclass_sum':brier/len(rows),'class_order':FRAME_ORDER,
            'confusion_rows_true_columns_predicted':confusion}


async def evaluate(cfg,cases,out,backend='jev',limit=20):
    if cfg.get('semantic',{}).get('schema')!=SCHEMA:raise ValueError('Use the production configuration')
    rows=[strict_json(line) for line in Path(cases).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not 1<=limit<=1000:raise ValueError('Evaluation limit must be 1–1000')
    selected=rows[:limit]
    if len({row['id'] for row in selected})!=len(selected):raise ValueError('Duplicate case IDs')
    for row in selected:
        if not isinstance(row['state'],dict) or row['label'] not in FRAME_ORDER:raise ValueError('Invalid semantic case')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    model=JevBackend(cfg['semantic']) if backend=='jev' else MockBackend(cfg['semantic'])
    results=[]
    try:
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as stream:
            for row in selected:
                start=time.perf_counter()
                result={'id':row['id'],'language':row['language'],'label':row['label'],
                        'label_origin':row.get('label_origin','unspecified')}
                try:
                    answer=await model.evaluate(row['state'])
                    result.update(status='ok',frames=answer['frames'],model=answer['model'])
                except SemanticError as error:
                    result.update(status='error',reason=error.reason)
                result['latency_ms']=round((time.perf_counter()-start)*1000,2)
                results.append(result);stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                if result['status']=='error':break
    finally:await model.close()
    good=[r for r in results if r['status']=='ok'];latencies=sorted(r['latency_ms'] for r in good)
    summary={'schema':SCHEMA,'backend':backend,'cases':len(results),'errors':len(results)-len(good),
        'scope':'authored diagnostic cases; not independent human validation',
        'metrics':metrics(good),'by_language':{lang:metrics([r for r in good if r['language']==lang])
                                             for lang in sorted({r['language'] for r in good})},
        'latency_p50_ms':latencies[len(latencies)//2] if latencies else None,
        'latency_p95_ms':latencies[min(len(latencies)-1,math.ceil(len(latencies)*.95)-1)] if latencies else None}
    save(out/'summary.json',summary)
    return {'out':str(out.resolve()),**summary}


def review(path,out):
    from nod.experiments.research import simulate
    raw=Path(path).read_bytes();rows=[strict_json(line) for line in raw.decode('utf-8').splitlines()]
    base=rows[0]['config'];validate_config(base)
    if not base.get('interaction',{}).get('enabled'):raise ValueError('Expected a production session')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    configs={};results={}
    for condition in ('full','no_history','direct','argmax','acoustic_only'):
        c=deepcopy(base);c['interaction']['condition']=condition;c['research']['condition']=condition
        configs[condition]=c;results[condition]=simulate(rows,c)
    save(out/'conditions.json',configs)
    save(out/'comparison.json',{'conditions':results,'source_sha256':hashlib.sha256(raw).hexdigest(),
        'quality_measured':False,'controller':'simulated acknowledgements',
        'note':'Identical input stream; acoustic_only does not require semantics. Counts are not HRI outcomes.'})
    engine=create_engine(base);count=0
    with (out/'annotations.jsonl').open('x',encoding='utf-8') as labels,(out/'predictions.jsonl').open('x',encoding='utf-8') as preds:
        for row in rows[1:]:
            event=Event(**row['event']);derived=engine.process(event)
            if event.kind!='semantic_result' or not any(r['kind']=='semantic_disposition' and r['reason']=='accepted' for r in derived):continue
            data=event.data;request=data['request']
            ident=hashlib.sha256(raw+str(request['seq']).encode()).hexdigest()[:20]
            labels.write(json.dumps({'id':ident,'schema':SCHEMA,'state':request.get('state'),
                'unit_id':request['snapshot']['utterance_id'],'language':base['asr']['language'],
                'split':'development','label':None,'reviewed':False,'annotator':None},ensure_ascii=False)+'\n')
            preds.write(json.dumps({'id':ident,'frames':data['result']['frames'],
                'filtered_frames':engine.belief.frames_at(event.at_ms)})+'\n');count+=1
    return {'out':str(out.resolve()),'annotation_items':count,'quality_measured':False,
            'functions':{name:r['function_counts'] for name,r in results.items()}}


def score(annotations,predictions,calibrate=False):
    predictions_path=Path(predictions)
    labels=[strict_json(x) for x in Path(annotations).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    predictions=[strict_json(x) for x in Path(predictions).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if len({r['id'] for r in labels})!=len(labels) or len({r['id'] for r in predictions})!=len(predictions):
        raise ValueError('Duplicate item IDs')
    lookup={r['id']:r for r in predictions};rows=[]
    for row in labels:
        if row.get('reviewed') is not True:continue
        if row.get('label') not in FRAME_ORDER or not row.get('annotator') or row['id'] not in lookup:
            raise ValueError('Reviewed items require a frame label, annotator and matching prediction')
        if calibrate and row.get('split')!='calibration':raise ValueError('Temperature fitting accepts only the calibration split')
        rows.append({'id':row['id'],'label':row['label'],'frames':frame_distribution(lookup[row['id']]['frames'])})
    result={'reviewed':len(rows),'unreviewed':len(labels)-len(rows),'metrics':metrics(rows)}
    if calibrate:
        if len(rows)<20:raise ValueError('At least 20 reviewed calibration items are required')
        candidates=[.5+i*.05 for i in range(91)]
        def loss(t):return metrics([{**r,'frames':temper(r['frames'],t)} for r in rows])['nll']
        temperature=min(candidates,key=loss)
        result.update(temperature=temperature,calibration='temperature_fitted',
            calibration_nll=loss(temperature),scope='fit on calibration split; evaluate separately',
            annotations_sha256=hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
            predictions_sha256=hashlib.sha256(predictions_path.read_bytes()).hexdigest(),semantic_schema=SCHEMA)
    return result
