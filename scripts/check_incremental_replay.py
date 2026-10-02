"""Local-only diagnostic: frozen recorded observations, simulated controller ACKs."""
from collections import Counter
from copy import deepcopy
import hashlib
import heapq
import json
from pathlib import Path
import sys
from nod.config import validate_config
from nod.core.engine import create_engine
from nod.core.events import Event
from nod.incremental import configure
from nod.telemetry import SessionLog,replay
from nod.provenance import source_digest


def counterfactual(rows,cfg,path):
    queue=[];serial=0;engine=create_engine(cfg);actions=[];dispositions=Counter()
    for row in rows[1:]:
        event=row['event']
        if event['kind']=='feedback':continue
        serial+=1;heapq.heappush(queue,(event['at_ms'],serial,event))
    log=SessionLog(path,cfg,'frozen_recorded_observations_simulated_ack')
    try:
        while queue:
            _,_,raw=heapq.heappop(queue);event=Event(**raw);records=engine.process(event);log.event(event,records)
            for record in records:
                if record['kind']=='semantic_disposition':dispositions[record['reason']]+=1
                if record['kind']!='command':continue
                c=record['command']
                actions.append({'at_ms':c['at_ms'],'function':record['function'],'expression':c['expression'],
                    'evidence_text':record.get('evidence_text'),'retained':record.get('retained_evidence',False)})
                for delay,status in [(10,'accepted'),(20,'started'),(20+c['duration_ms'],'completed')]:
                    serial+=1;t=event.at_ms+delay
                    heapq.heappush(queue,(t,serial,{'kind':'feedback','at_ms':t,'data':{'action_id':c['action_id'],'status':status}}))
    finally:log.close()
    result=replay(path)
    return {'replay_matched':result['matched'],'events':result['events'],
        'functions':dict(Counter(a['function'] for a in actions)),
        'semantic_dispositions':dict(dispositions),'actions':actions}


def main():
    source,out=map(Path,sys.argv[1:]);raw=source.read_bytes()
    rows=[json.loads(s) for s in raw.decode('utf-8').splitlines()]
    out.mkdir(parents=True,exist_ok=False)
    configs={'v2':rows[0]['config'],'v3':configure(rows[0]['config'])}
    report={'source_sha256':hashlib.sha256(raw).hexdigest(),
            'implementation_sha256':source_digest(),
            'scope':'development counterfactual; frozen old API responses and simulated ACKs; not a new user experiment',
            'conditions':{}}
    for name,cfg in configs.items():
        cfg=deepcopy(cfg);cfg['experiment']={**cfg.get('experiment',{}),'counterfactual':True,'source_sha256':report['source_sha256']}
        validate_config(cfg)
        report['conditions'][name]=counterfactual(rows,cfg,out/(name+'.jsonl'))
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{'functions':v['functions'],'replay_matched':v['replay_matched'],'semantic_dispositions':v['semantic_dispositions']} for k,v in report['conditions'].items()}))


if __name__=='__main__':main()
