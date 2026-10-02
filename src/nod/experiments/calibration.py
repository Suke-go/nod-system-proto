"""Human-reviewed head calibration. No API calls, hidden-state guesses, or auto labels."""
import argparse
from collections import Counter
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path

from nod.config import strict_json,validate_config,load_config
from nod.semantic.sensor import observation,prompt_digest,scale_report as power,calibrate_scalars as scalar_transform

SCHEMA='listener-head-review-v1'
HEADS={'interpretability':('unresolved','resolved'),
       'appraisal':('neutral','positive','negative','mixed'),
       'completion':('incomplete','complete'),
       'response_demand':('not_required','required')}


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def read(path):return [strict_json(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
def jsonl(path,rows):
    with Path(path).open('x',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def head_values(obs,head):
    if head in ('interpretability','appraisal'):return [obs[head][name] for name in HEADS[head]]
    return [1-obs[head],obs[head]]


def export(logs,out,split='development',participant=None):
    if split not in ('development','calibration','test'):raise ValueError('Invalid split')
    annotations=[];predictions=[];sources=[]
    for path in logs:
        raw=Path(path).read_bytes();group=hashlib.sha256(raw).hexdigest()
        if group in sources:raise ValueError('Duplicate source session')
        rows=[strict_json(line) for line in raw.decode('utf-8').splitlines()];header=rows[0];cfg=header['config']
        validate_config(cfg)
        if cfg.get('listener',{}).get('version')!=4:raise ValueError('Export requires v4 typed observations')
        if cfg['listener'].get('scalar_calibration'):raise ValueError('Export raw, uncalibrated sessions only')
        question_hash=cfg.get('experiment',{}).get('semantic_prompt_sha256')
        if not isinstance(question_hash,str) or len(question_hash)!=64 or any(c not in '0123456789abcdef' for c in question_hash):
            raise ValueError('Recorded question digest required; do not infer historical prompts from current files')
        selected={}
        for row in rows[1:]:
            event=row['event']
            if event['kind']!='semantic_result' or not any(d['kind']=='semantic_disposition' and d['reason']=='accepted' for d in row['derived']):continue
            request=event['data']['request'];selected[request['snapshot']['utterance_id']]=event
        for unit,event in selected.items():
            req=event['data']['request'];result=event['data']['result'];obs=observation(result['observation'])
            state=req.get('state')
            if not isinstance(state,dict):raise ValueError('Recorded model input is required')
            identity=digest([group,unit,req['seq']]);input_hash=digest(state)
            a={'schema':SCHEMA,'id':identity,'session_group':group,'participant':participant,
               'unit_id':unit,'input_sha256':input_hash,'language':cfg['asr']['language'],'split':split,
               'input':state,'labels':dict.fromkeys(HEADS),'reviewed':False,'annotator':None}
            meta={'model':result['model'],'prompt_sha256':question_hash}
            binding=digest({k:a[k] for k in ('session_group','participant','unit_id','input_sha256','language','split')})
            predictions.append({'schema':SCHEMA,'id':identity,'binding':binding,**meta,'observation':obs,
                                'source_seq':req['seq'],'audio_available':header.get('audio_recorded',False)})
            annotations.append(a)
        sources.append(group)
    if not annotations:raise ValueError('No accepted typed observations')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    jsonl(out/'annotations.jsonl',annotations);jsonl(out/'predictions.jsonl',predictions)
    payload=json.dumps(annotations,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    template=files('nod').joinpath('resources/review-heads.html').read_text(encoding='utf-8')
    (out/'review.html').write_text(template.replace('__REVIEW_DATA__',payload),encoding='utf-8')
    manifest={'schema':SCHEMA,'sessions':sources,'items':len(annotations),'split':split,
              'selection':'last accepted observation per semantic unit; overlapping units remain session-grouped',
              'scope':'text semantic judgments only; no perception or timing ground truth',
              'annotated':False}
    write(out/'manifest.json',manifest)
    return {'out':str(out.resolve()),**manifest}


def pairs(annotations,predictions):
    labels=read(annotations);preds=read(predictions)
    if len({r['id'] for r in labels})!=len(labels) or len({r['id'] for r in preds})!=len(preds):raise ValueError('Duplicate item IDs')
    if {r['id'] for r in labels}!={r['id'] for r in preds}:raise ValueError('Annotation/prediction coverage mismatch; retain unreviewed rows')
    lookup={r['id']:r for r in preds};result=[]
    for a in labels:
        if a.get('schema')!=SCHEMA or a['id'] not in lookup:raise ValueError('Missing matching head prediction')
        p=lookup[a['id']]
        binding=digest({k:a[k] for k in ('session_group','participant','unit_id','input_sha256','language','split')})
        if p.get('schema')!=SCHEMA or p.get('binding')!=binding or digest(a['input'])!=a['input_sha256']:
            raise ValueError('Changed input, grouping, language, participant or split')
        p['observation']=observation(p['observation'])
        if set(a['labels'])!=set(HEADS):raise ValueError('Expected four semantic head labels')
        for head,value in a['labels'].items():
            if value is not None and value not in HEADS[head]:raise ValueError('Unknown human label')
        if a.get('reviewed') is not True:continue
        if not isinstance(a.get('annotator'),str) or not a['annotator'].strip():raise ValueError('Human reviewer identifier required')
        result.append((a,p))
    return labels,result


def metadata(rows):
    values={(p['model'],p['prompt_sha256']) for _,p in rows}
    if len(values)!=1:raise ValueError('Use one classifier model and prompt per calibration')
    model,prompt=next(iter(values));return {'model':model,'prompt_sha256':prompt}


def bounded_minimum(fn,lo,hi):
    # Both implemented objectives are convex in their positive scale parameter.
    for _ in range(100):
        left=lo+(hi-lo)/3;right=hi-(hi-lo)/3
        if fn(left)<fn(right):hi=right
        else:lo=left
    return (lo+hi)/2


def fit(annotations,predictions):
    _,rows=pairs(annotations,predictions)
    if not rows:raise ValueError('No human-reviewed items; no parameters were fitted')
    if any(a['split']!='calibration' for a,_ in rows):raise ValueError('Fitting requires an explicitly exported calibration split')
    groups={a['session_group'] for a,_ in rows}
    if len(groups)<2:raise ValueError('At least two source sessions are required; this is not a power calculation')
    meta=metadata(rows);scales={};counts={}
    for head,names in HEADS.items():
        sample=[(head_values(p['observation'],head),names.index(a['labels'][head])) for a,p in rows if a['labels'][head] is not None]
        count=Counter(y for _,y in sample);counts[head]={name:count[i] for i,name in enumerate(names)}
        if any(count[i]<8 for i in range(len(names))):raise ValueError(f'{head}: need at least 8 reviewed examples per category; unavailable labels are not negatives')
        if head in ('interpretability','appraisal'):
            # log Gamma(K+k)-log Gamma(1+k) = sum_{j=1}^{K-1} log(k+j).
            logs=sum(math.log(max(1e-6,q[y])/sum(max(1e-6,v) for v in q)) for q,y in sample)
            k=len(names);n=len(sample)
            fn=lambda scale: -n*sum(math.log(scale+j) for j in range(1,k))-scale*logs
        else:
            fn=lambda scale: -sum(math.log(max(1e-15,power(q,scale)[y])) for q,y in sample)
        scales[head]=bounded_minimum(fn,.05,20.)
    return {'schema':'listener-head-calibration-v1',**meta,'status':'partially_fitted',
        'concentration':{'interpretability':scales['interpretability'],'appraisal':scales['appraisal']},
        'temperature':{head:1/scales[head] for head in ('completion','response_demand')},
        'training_sessions':sorted(groups),'training_inputs':sorted({a['input_sha256'] for a,_ in rows}),
        'training_participants':sorted({a['participant'] for a,_ in rows if a['participant']}),
        'counts':counts,'unfitted':['perception','initial','transition','time_decay','action_losses'],
        'annotations_sha256':hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
        'predictions_sha256':hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
        'estimators':{'choice':'Dirichlet report-density MLE','noul':'temperature scaling NLL'},
        'scope':'reviewed calibration fit; held-out evaluation required; not a human mental-state measurement'}


def validate_artifact(a):
    if a.get('schema')!='listener-head-calibration-v1' or a.get('status')!='partially_fitted':raise ValueError('Invalid head calibration')
    for name,keys in [('concentration',{'interpretability','appraisal'}),('temperature',{'completion','response_demand'})]:
        if set(a[name])!=keys:raise ValueError('Unexpected calibration axes')
        for value in a[name].values():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not .05<=value<=20.:
                raise ValueError('Calibration scale outside [0.05,20]')
    for key in ('training_sessions','training_inputs','training_participants'):
        if not isinstance(a.get(key),list) or any(not isinstance(v,str) or not v.strip() for v in a[key]):
            raise ValueError('Invalid calibration provenance')
    if len(set(a['training_sessions']))<2 or not a['training_inputs']:raise ValueError('Calibration provenance missing')
    if not isinstance(a.get('model'),str) or not a['model']:raise ValueError('Missing classifier model')
    checksum=a.get('prompt_sha256','')
    if not isinstance(checksum,str) or len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Missing question digest')


def score(annotations,predictions,artifact=None):
    labels,rows=pairs(annotations,predictions)
    if artifact:
        validate_artifact(artifact)
        if any(a['split']!='test' for a,_ in rows):raise ValueError('Calibrated scoring requires the test split')
        for a,p in rows:
            if p['model']!=artifact['model'] or p['prompt_sha256']!=artifact['prompt_sha256']:raise ValueError('Model or prompt mismatch')
            if a['session_group'] in artifact['training_sessions'] or a['input_sha256'] in artifact['training_inputs']:
                raise ValueError('Calibration/test leakage')
            if a['participant'] and a['participant'] in artifact['training_participants']:raise ValueError('Participant overlaps calibration')
    def metrics(head,subset,calibrated):
        items=[(a,p) for a,p in subset if a['labels'][head] is not None]
        if not items:return None
        brier=nll=correct=0.
        for a,p in items:
            q=head_values(p['observation'],head)
            if calibrated:
                scale=artifact['concentration'][head] if head in artifact['concentration'] else 1/artifact['temperature'][head]
                q=power(q,scale)
            y=HEADS[head].index(a['labels'][head]);correct+=max(range(len(q)),key=q.__getitem__)==y
            nll-=math.log(max(1e-15,q[y]));brier+=sum((v-(i==y))**2 for i,v in enumerate(q))
        n=len(items);return {'n':n,'accuracy':correct/n,'nll':nll/n,'brier':brier/n}
    slices={'all':rows}
    slices.update({f'language:{lang}':[(a,p) for a,p in rows if a['language']==lang] for lang in sorted({a['language'] for a,_ in rows})})
    slices.update({f'session:{group}':[(a,p) for a,p in rows if a['session_group']==group] for group in sorted({a['session_group'] for a,_ in rows})})
    return {'reviewed':len(rows),'unreviewed':len(labels)-len(rows),
        'metrics':{name:{head:{'raw':metrics(head,subset,False),
                   **({'transformed':metrics(head,subset,True)} if artifact else {})} for head in HEADS} for name,subset in slices.items()},
        'participant_disjoint_verified':bool(rows and artifact and all(a['participant'] for a,_ in rows)),
        'scope':'semantic heads on text; choice transformed score uses uniform reference class prior, not the six-state listener posterior; no perception, timing or HRI outcome evaluation'}


def configured(artifact):
    from nod.grounded import configure
    validate_artifact(artifact);cfg=configure(load_config())
    if artifact['model']!=cfg['semantic']['model'] or artifact['prompt_sha256']!=prompt_digest(cfg['semantic']['question_resource']):
        raise ValueError('Calibration must match the deployed model and question definitions')
    model=cfg['listener']['observation_model'];model['concentration'].update(artifact['concentration'])
    model.update(status='partially_fitted_on_calibration',fitted_axes=['interpretability','appraisal'])
    cfg['listener']['scalar_calibration']={'temperature':artifact['temperature'],'artifact_sha256':digest(artifact),
        'model':artifact['model'],'prompt_sha256':artifact['prompt_sha256']}
    cfg['research'].update(parameters_status='meaning_axes_fitted_perception_transition_loss_unfitted',
        calibration_training_sessions=artifact['training_sessions'],calibration_artifact_sha256=digest(artifact))
    validate_config(cfg);return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    exp=commands.add_parser('export');exp.add_argument('logs',nargs='+');exp.add_argument('--out',required=True)
    exp.add_argument('--split',choices=['development','calibration','test'],default='development');exp.add_argument('--participant')
    for name in ('fit','score'):
        p=commands.add_parser(name);p.add_argument('annotations');p.add_argument('predictions');p.add_argument('--out',required=True)
        if name=='score':p.add_argument('--artifact')
    p=commands.add_parser('config');p.add_argument('artifact');p.add_argument('--out',required=True)
    args=parser.parse_args()
    try:
        if args.command=='export':result=export(args.logs,args.out,args.split,args.participant)
        elif args.command=='fit':result=fit(args.annotations,args.predictions);write(args.out,result)
        elif args.command=='score':
            result=score(args.annotations,args.predictions,strict_json(Path(args.artifact).read_text(encoding='utf-8')) if args.artifact else None);write(args.out,result)
        else:result=configured(strict_json(Path(args.artifact).read_text(encoding='utf-8')));write(args.out,result);result={'config':str(Path(args.out).resolve())}
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,KeyError,TypeError,OSError) as error:parser.exit(2,str(error)+'\n')


if __name__=='__main__':main()
