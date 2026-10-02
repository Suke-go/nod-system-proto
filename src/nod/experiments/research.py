"""Development-only diagnostics and ablations. Never calls a model or actuator."""
from collections import Counter
from copy import deepcopy
import hashlib
import heapq
import json
from pathlib import Path
from nod.config import CLASSES,strict_json,distribution
from nod.core.engine import create_engine as Engine
from nod.core.events import Event
from nod.live_session import live_config


def simulate(rows,cfg,predictions=None):
    e=Engine(cfg);queue=[];sequence=0;actions=[];raw=Counter();filtered=Counter();refined=0
    for row in rows[1:]:
        event=row['event']
        if event['kind']=='feedback':continue
        sequence+=1;heapq.heappush(queue,(event['at_ms'],sequence,event))
    while queue:
        _,_,event=heapq.heappop(queue)
        records=e.process(Event(**event))
        for record in records:
            if record['kind']=='semantic_refinement':refined+=1
            if record['kind']=='semantic_disposition' and record['reason']=='accepted':
                p=event['data']['result']['probabilities'];b=e.belief.at(e.now)
                raw[CLASSES[max(range(4),key=lambda i:p[i])]]+=1
                filtered[CLASSES[max(range(4),key=lambda i:b[i])]]+=1
                if predictions is not None:
                    predictions[event['data']['request']['seq']]={'raw':p,'filtered':b}
            if record['kind']!='command':continue
            command=record['command']
            actions.append({'at_ms':event['at_ms'],'function':record['function'],
                            'action':command['action'],'trigger':record.get('trigger','vap'),
                            'utterance_id':e.transcripts.current.utterance_id,
                            'intensity':command['intensity'],'duration_ms':command['duration_ms'],
                            'semantic_age_ms':e.now-e.semantic['source_ms'] if e.semantic else None})
            for delay,status in [(10,'accepted'),(20,'started'),(20+command['duration_ms'],'completed')]:
                sequence+=1;t=event['at_ms']+delay
                heapq.heappush(queue,(t,sequence,{'kind':'feedback','at_ms':t,
                    'data':{'action_id':command['action_id'],'status':status}}))
    duration_ms=rows[-1]['event']['at_ms'] if len(rows)>1 else 0
    return {'actions':actions,'function_counts':dict(Counter(x['function'] for x in actions)),
            'trigger_counts':dict(Counter(x['trigger'] for x in actions)),
            'raw_top_class_counts':dict(raw),'filtered_top_class_counts':dict(filtered),
            'actions_per_minute':len(actions)*60000/duration_ms if duration_ms else None,
            'commanded_motion_ms':sum(x['duration_ms'] for x in actions),
            'mean_commanded_intensity':sum(x['intensity'] for x in actions)/len(actions) if actions else None,
            'exact_final_refinements':refined,'motor_state':e.motor.state}


def ablation_configs(recorded):
    base=live_config(deepcopy(recorded),recorded['asr']['language'],0,response='bayes')
    configs={'recorded_config':deepcopy(recorded)}
    for rule in ('likelihood_product','posterior_blend'):
        for prior_name,prior in [('symmetric',[.25]*4),('original',[.6,.25,.1,.05])]:
            cfg=deepcopy(base);cfg['belief']['observation_rule']=rule
            cfg['belief']['initial']=prior;cfg['belief']['transition']['stationary']=prior
            cfg['research']['method']='generalized_bayes_pseudo_likelihood' if rule=='likelihood_product' else 'posterior_blend'
            cfg['research']['prior']=prior_name+'_assumption_not_estimated'
            configs[f'{rule}_{prior_name}']=cfg
    for name in ('bayes-no-history','direct','presentation'):
        configs[name]=live_config(deepcopy(recorded),recorded['asr']['language'],0,response=name)
    no_endpoint=deepcopy(base);no_endpoint['policy']['endpoint_specific_ack']['enabled']=False
    configs['bayes_without_endpoint']=no_endpoint
    no_refinement=deepcopy(base);no_refinement['semantic']['refine_exact_final']=False
    configs['bayes_without_final_refinement']=no_refinement
    for name,cfg in configs.items():
        if name=='recorded_config':continue
        cfg['profile']='ablation_'+name
        cfg['research']['condition']=name
        cfg['research']['endpoint_extension']=cfg['policy']['endpoint_specific_ack']['enabled']
    return configs


def review_session(path,out):
    raw=Path(path).read_bytes();rows=[strict_json(line) for line in raw.decode('utf-8').splitlines()]
    if not rows or rows[0].get('record')!='header':raise ValueError('Expected a session log')
    if rows[0]['config'].get('listener',{}).get('enabled'):
        from nod.experiments.listener import review
        return review(path,out)
    if rows[0]['config'].get('interaction',{}).get('enabled'):
        from nod.experiments.interaction import review
        return review(path,out)
    digest=hashlib.sha256(raw).hexdigest();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    configs=ablation_configs(rows[0]['config'])
    by_condition={name:{} for name in configs}
    boundary_counts=Counter(row['event']['data'].get('final_reason') or 'unknown'
        for row in rows[1:] if row['event']['kind']=='asr' and row['event']['data'].get('final'))
    result={'kind':'counterfactual_development_not_quality_evaluation','source_sha256':digest,
        'language':rows[0]['config']['asr']['language'],
        'duration_ms':rows[-1]['event']['at_ms'],'feedback':'simulated 10ms accepted, 20ms started, duration completion',
        'final_reason_counts':dict(boundary_counts),
        'boundary_limitation':'Unknown legacy boundaries are not treated as silence in research-v2.',
        'primary_memory_comparison':['likelihood_product_symmetric','bayes-no-history'],
        'direct_comparison':['likelihood_product_symmetric','direct'],
        'shared_semantic_inputs':True,'quality_measured':False,
        'conditions':{name:simulate(rows,cfg,by_condition[name]) for name,cfg in configs.items()}}
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'conditions.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2),encoding='utf-8')
    # One accepted request per utterance. Human annotators see precisely the input
    # provided to that request, without model outputs or later transcript revisions.
    engine=Engine(rows[0]['config']);chosen={}
    for row in rows[1:]:
        event=row['event'];records=engine.process(Event(**event))
        if event['kind']!='semantic_result' or records[0]['reason']!='accepted':continue
        request=event['data']['request'];snapshot=request['snapshot']
        rank=(bool(snapshot.get('final')),snapshot['reliability'],snapshot['source_ms'])
        uid=snapshot['utterance_id']
        if uid not in chosen or rank>chosen[uid][0]:
            chosen[uid]=(rank,request,event['data']['result']['probabilities'],engine.belief.at(engine.now))
    selected=[]
    with (out/'annotations.jsonl').open('x',encoding='utf-8') as labels, (out/'predictions.jsonl').open('x',encoding='utf-8') as predictions:
        for uid,(_,request,p,b) in chosen.items():
            ident=hashlib.sha256(f'{digest}:{uid}:{request["seq"]}'.encode()).hexdigest()[:20]
            item={'id':ident,'session_sha256':digest,'utterance_id':uid,
                  'language':rows[0]['config']['asr']['language'],'state':request['state'],
                  'source_ms':request['snapshot']['source_ms'],'split':'development',
                  'label':None,'reviewed':False,'annotator':None}
            labels.write(json.dumps(item,ensure_ascii=False)+'\n')
            predictions.write(json.dumps({'id':ident,'raw':p,'filtered':b})+'\n')
            selected.append((ident,request['seq']))
    (out/'predictions-by-condition').mkdir()
    for name,values in by_condition.items():
        with (out/'predictions-by-condition'/f'{name}.jsonl').open('x',encoding='utf-8') as stream:
            for ident,seq in selected:
                if seq in values:
                    stream.write(json.dumps({'id':ident,**values[seq]})+'\n')
    return {'out':str(out.resolve()),'annotation_items':len(chosen),'quality_measured':False,
            'function_counts':{name:r['function_counts'] for name,r in result['conditions'].items()}}


def score_annotations(annotations,predictions):
    import math
    labels=[strict_json(x) for x in Path(annotations).read_text(encoding='utf-8').splitlines()]
    predicted=[strict_json(x) for x in Path(predictions).read_text(encoding='utf-8').splitlines()]
    if len({x['id'] for x in labels})!=len(labels) or len({x['id'] for x in predicted})!=len(predicted):
        raise ValueError('Duplicate item IDs')
    by_id={x['id']:x for x in predicted};reviewed=[]
    for x in labels:
        if x.get('reviewed') is not True:continue
        if x.get('label') not in CLASSES or not isinstance(x.get('annotator'),str) or not x['annotator'].strip():
            raise ValueError('Reviewed labels require a class and an annotator')
        if x['id'] not in by_id:raise ValueError('Missing prediction for a reviewed item')
        reviewed.append(x)
    result={'reviewed':len(reviewed),'excluded_unreviewed':len(labels)-len(reviewed),
            'scope':'human-labelled development data; no generalization claim','metrics':None}
    if not reviewed:return result
    result['metrics']={}
    for field in ('raw','filtered'):
        nll=brier=correct=0.;matrix=[[0]*4 for _ in range(4)]
        for x in reviewed:
            p=distribution(by_id[x['id']][field]);y=CLASSES.index(x['label']);guess=max(range(4),key=lambda i:p[i])
            correct+=guess==y;nll-=math.log(max(p[y],1e-12))
            brier+=sum((p[i]-(i==y))**2 for i in range(4));matrix[y][guess]+=1
        n=len(reviewed);result['metrics'][field]={'accuracy':correct/n,'nll':nll/n,
            'brier_multiclass_sum':brier/n,'confusion_rows_true_columns_predicted':matrix,'class_order':CLASSES}
    return result
