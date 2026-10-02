"""Listener-state review, joint observation-density fitting, and API diagnostics."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from nod.config import strict_json, validate_config
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.belief.listener import STATES, gaussian_logs
from nod.semantic.sensor import SCHEMA, FEATURES, observation, prompt_digest
from nod.listener import CONDITIONS


def save(path, data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)


def read_rows(path):
    return [strict_json(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def read_pairs(annotations, predictions):
    labels, predictions = read_rows(annotations), read_rows(predictions)
    if len({r['id'] for r in labels})!=len(labels) or len({r['id'] for r in predictions})!=len(predictions):
        raise ValueError('Duplicate review IDs')
    lookup={r['id']:r for r in predictions};pairs=[]
    for label in labels:
        if label.get('reviewed') is not True:continue
        if label.get('state') not in STATES or not label.get('annotator') or label['id'] not in lookup:
            raise ValueError('Reviewed rows require a state, annotator, and matching prediction')
        if label.get('schema')!=SCHEMA:raise ValueError('Annotation schema mismatch')
        pred=lookup[label['id']]
        if pred.get('schema')!=SCHEMA:raise ValueError('Prediction schema mismatch')
        pairs.append((label,pred))
    return labels,pairs


def score(annotations,predictions):
    labels,pairs=read_pairs(annotations,predictions)
    if not pairs:return {'reviewed':0,'unreviewed':len(labels),'metrics':None}
    correct=nll=brier=0.;confusion=[[0]*6 for _ in STATES]
    for a,p in pairs:
        values=p['posterior']
        if len(values)!=6 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in values) or abs(sum(values)-1)>1e-6:
            raise ValueError('Invalid state posterior')
        y=STATES.index(a['state']);best=max(range(6),key=values.__getitem__)
        correct+=best==y;nll-=math.log(max(values[y],1e-15))
        brier+=sum((v-(i==y))**2 for i,v in enumerate(values));confusion[y][best]+=1
    n=len(pairs)
    return {'reviewed':n,'unreviewed':len(labels)-n,'state_order':STATES,
            'metrics':{'accuracy':correct/n,'nll':nll/n,'brier':brier/n,'confusion':confusion}}


def fit(annotations,predictions):
    _,pairs=read_pairs(annotations,predictions)
    if any(a.get('split')!='calibration' for a,_ in pairs):raise ValueError('Use only the calibration split')
    if any(p.get('observation_family')=='hierarchical_dirichlet_report' for _,p in pairs):
        from nod.experiments.report_fit import fit_reports
        metadata={(p['prompt_sha256'],p['model']) for _,p in pairs}
        if len(metadata)!=1:raise ValueError('Do not mix prompts or classifier models during calibration')
        model,counts=fit_reports(pairs);prompt,backend=next(iter(metadata))
        return {'schema':SCHEMA,'state_order':STATES,'prompt_sha256':prompt,'model':backend,
            'observation_model':model,'n':len(pairs),'counts':counts,
            'estimator':'bounded maximum likelihood, concentration [0.05,20], uniform nuisance axes',
            'annotations_sha256':hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
            'predictions_sha256':hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
            'scope':'reviewed calibration only; held-out sessions and participant validation required'}
    by={s:[] for s in STATES};metadata=set()
    for a,p in pairs:
        x=p['features']
        if len(x)!=5 or any(v is None or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in x):
            raise ValueError('Fitting requires complete finite observation features')
        metadata.add((p['prompt_sha256'],p['model']))
        by[a['state']].append(x)
    if any(len(rows)<8 for rows in by.values()):raise ValueError('At least 8 reviewed calibration examples per state are required')
    if len(metadata)!=1:raise ValueError('Do not mix prompts or classifier models during calibration')
    means=[[sum(x[j] for x in by[s])/len(by[s]) for j in range(5)] for s in STATES]
    n=sum(map(len,by.values()));cov=[[0.]*5 for _ in range(5)]
    for i,s in enumerate(STATES):
        for x in by[s]:
            for j in range(5):
                for k in range(5):cov[j][k]+=(x[j]-means[i][j])*(x[k]-means[i][k])/(n-6)
    # A fixed, reported shrinkage estimator keeps the joint covariance invertible.
    cov=[[.8*cov[j][k]+(.2*cov[j][j]+.1 if j==k else 0) for k in range(5)] for j in range(5)]
    model={'family':'joint_logistic_normal_appraisal_marginalized','features':list(FEATURES),
           'means':means,'covariance':cov,'status':'fitted_on_calibration'}
    gaussian_logs([0.]*5,model)
    prompt,backend=next(iter(metadata))
    return {'schema':SCHEMA,'state_order':STATES,'prompt_sha256':prompt,'model':backend,
            'observation_model':model,'n':n,'counts':{s:len(v) for s,v in by.items()},
            'regularization':{'diagonal_shrinkage':.2,'ridge':.1},
            'annotations_sha256':hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
            'predictions_sha256':hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
            'scope':'calibration only; independent test and participant validation still required'}


def load_model(cfg,path):
    a=strict_json(Path(path).read_text(encoding='utf-8-sig'))
    if ('listener' not in cfg or a.get('schema')!=SCHEMA or a.get('state_order')!=list(STATES)
        or a.get('prompt_sha256')!=prompt_digest(cfg['semantic']['question_resource']) or a.get('model')!=cfg['semantic']['model']
        or a.get('observation_model',{}).get('status')!='fitted_on_calibration'):
        raise ValueError('Observation model must match the schema, prompt and pinned classifier model')
    candidate=deepcopy(cfg);candidate['listener']['observation_model']=a['observation_model'];validate_config(candidate)
    cfg['listener']['observation_model']=a['observation_model']
    cfg['research']['observation_artifact_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cfg['research']['parameters_status']='observation_fitted_transition_and_utility_designed'


def review(path,out):
    from nod.experiments.research import simulate
    raw=Path(path).read_bytes();rows=read_rows(path);cfg=rows[0]['config'];validate_config(cfg)
    if 'listener' not in cfg:raise ValueError('Expected listener-v2 log')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    configs={};results={}
    for name in CONDITIONS:
        c=deepcopy(cfg);c['listener']['condition']=name;c['research']['condition']=name
        configs[name]=c;results[name]=simulate(rows,c)
    save(out/'conditions.json',configs)
    save(out/'comparison.json',{'conditions':results,'source_sha256':hashlib.sha256(raw).hexdigest(),
        'quality_measured':False,'controller':'simulated acknowledgements',
        'scope':'same observations, simulated policy comparison; not human outcomes'})
    engine=create_engine(cfg);annotations=[];predictions=[]
    for row in rows[1:]:
        e=Event(**row['event']);records=engine.process(e)
        if e.kind!='semantic_result' or not any(r['kind']=='semantic_disposition' and r['reason']=='accepted' for r in records):continue
        request=e.data['request'];seq=request['seq']
        ident=hashlib.sha256(raw+str(seq).encode()).hexdigest()[:20]
        annotations.append({'id':ident,'schema':SCHEMA,'input':request.get('state'),
            'perception_reliability':request['snapshot']['reliability'] if cfg['listener']['version']>=3 else engine.units.current.reliability,
            'unit_id':engine.units.current.utterance_id,'session_group':hashlib.sha256(raw).hexdigest()[:16],
            'language':cfg['asr']['language'],'split':'development','state':None,'reviewed':False,'annotator':None})
        summary=engine.semantic_trace['state'] if cfg['listener']['version']>=3 else engine.belief.summary(e.at_ms)
        if e.data['result']['schema']!=SCHEMA:continue
        predictions.append({'id':ident,'schema':SCHEMA,'observation':e.data['result']['observation'],
            'features':summary['features'],'posterior':summary['posterior'],
            'model':e.data['result'].get('model'),
            'prompt_sha256':cfg.get('experiment',{}).get('semantic_prompt_sha256',prompt_digest(cfg['semantic']['question_resource']))})
        if cfg['listener']['version']==4:
            predictions[-1].update(observation_family=summary['observation_family'],probability_report=summary['probability_report'])
    for name,data in (('annotations',annotations),('predictions',predictions)):
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8')
    return {'out':str(out.resolve()),'annotation_items':len(annotations),'quality_measured':False,
            'functions':{name:r['function_counts'] for name,r in results.items()}}


async def evaluate(cfg,cases,out,backend='jev',limit=20):
    from nod.semantic.backend import JevBackend,MockBackend,SemanticError
    rows=read_rows(cases)
    if not 1<=limit<=1000:raise ValueError('Invalid diagnostic limit')
    rows=rows[:limit]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate diagnostic IDs')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);results=[]
    model=JevBackend(cfg['semantic']) if backend=='jev' else MockBackend(cfg['semantic'])
    try:
        for row in rows:
            start=time.perf_counter();result={'id':row['id'],'language':row['language']}
            try:
                r=await model.evaluate(row['state']);q=observation(r['observation'])
                result.update(status='ok',observation=q,model=r['model'],
                    expected=row.get('expected'),scope='authored developmental diagnostic')
            except SemanticError as e:result.update(status='error',reason=e.reason)
            result['latency_ms']=round((time.perf_counter()-start)*1000,2);results.append(result)
            if result['status']=='error':break
    finally:await model.close()
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    lat=sorted(r['latency_ms'] for r in results if r['status']=='ok')
    summary={'schema':SCHEMA,'prompt_sha256':prompt_digest(cfg['semantic']['question_resource']),'attempted':len(results),
        'errors':sum(r['status']=='error' for r in results),'latency_p50_ms':lat[len(lat)//2] if lat else None,
        'latency_p95_ms':lat[min(len(lat)-1,math.ceil(len(lat)*.95)-1)] if lat else None,
        'scope':'authored diagnostics, not independent semantic or psychological validation'}
    save(out/'summary.json',summary)
    return {'out':str(out.resolve()),**summary}
