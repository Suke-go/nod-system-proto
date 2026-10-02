"""Offline policy comparison: recorded observations, simulated action acknowledgements.

No models, API calls, audio playback or actual motor commands are involved.
This is a counterfactual sensitivity check, not a measurement of new live performance.
"""
import argparse
import copy
import hashlib
import heapq
import json
from pathlib import Path
from nod.core.engine import Engine
from nod.core.events import Event
from nod.live_session import live_config


def simulate(rows, cfg):
    engine=Engine(cfg); queue=[]; sequence=0; actions=[]
    for row in rows[1:]:
        event=row['event']
        if event['kind']=='feedback': continue
        sequence+=1;heapq.heappush(queue,(event['at_ms'],sequence,event))
    while queue:
        _,_,event=heapq.heappop(queue)
        for record in engine.process(Event(**event)):
            if record['kind']!='command': continue
            command=record['command']
            actions.append({'at_ms':event['at_ms'],'function':record['function'],
                            'action':command['action'],'trigger':record.get('trigger','vap')})
            for delay,status in [(10,'accepted'),(20,'started'),(20+command['duration_ms'],'completed')]:
                sequence+=1; now=event['at_ms']+delay
                heapq.heappush(queue,(now,sequence,{'kind':'feedback','at_ms':now,
                    'data':{'action_id':command['action_id'],'status':status}}))
    return {'actions':len(actions),'decisions':actions,'motor_state':engine.motor.state}


def compare(path):
    raw=Path(path).read_bytes(); rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    old=rows[0]['config']; new=live_config(copy.deepcopy(old),old['asr']['language'],0,response='responsive')
    result={'kind':'counterfactual_not_live_measurement','source_sha256':hashlib.sha256(raw).hexdigest(),
            'duration_ms':rows[-1]['event']['at_ms'],'feedback':'simulated_for_each_new_action',
            'recorded_config':simulate(rows,old),'responsive':simulate(rows,new)}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log');parser.add_argument('--out')
    args=parser.parse_args(); result=compare(args.log)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.out: Path(args.out).write_text(text+'\n',encoding='utf-8')
    print(text)
