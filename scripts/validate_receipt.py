"""Frozen observations, simulated ACKs: engineering regression, not HRI efficacy."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from check_incremental_replay import counterfactual
from nod.config import validate_config
from nod.responsive import configure
from nod.telemetry import replay


def main():
    source,out=map(Path,sys.argv[1:])
    raw=source.read_bytes();rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    original=replay(source)
    # Original presentation receipts belong to original commands only.
    frozen=[rows[0]]+[row for row in rows[1:] if row['event']['kind']!='presentation']
    out.mkdir(parents=True,exist_ok=False)
    before=deepcopy(rows[0]['config'])
    after=deepcopy(before)
    new=configure(deepcopy(before))
    after['listener']['policy']['local_receipt']=new['listener']['policy']['local_receipt']
    after['asr'].update(preserve_final_jobs=True,final_queue_capacity=2)
    after['research']['policy_revision']='local-receipt-1'
    report={'source_sha256':hashlib.sha256(raw).hexdigest(),
        'original_replay_matched':original['matched'],
        'scope':'frozen ASR/Jev/VAP; simulated ACKs; no new Jev calls, no reconstructed ASR endpoints; development regression only',
        'conditions':{}}
    for name,cfg in [('before',before),('after',after)]:
        cfg['experiment']['counterfactual']=True
        validate_config(cfg)
        result=counterfactual(frozen,cfg,out/(name+'.jsonl'))
        result['expressions']=dict(Counter(a['expression'] for a in result['actions']))
        report['conditions'][name]=result
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{'functions':v['functions'],'expressions':v['expressions'],
        'replay_matched':v['replay_matched'],
        'receipts':[a for a in v['actions'] if a['function']=='understanding']}
        for k,v in report['conditions'].items()},ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
