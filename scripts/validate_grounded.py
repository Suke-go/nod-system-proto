"""Frozen development observations; ablations and sensitivity, never human outcomes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from check_incremental_replay import counterfactual
from nod.config import validate_config
from nod.grounded import configure
from nod.responsive import configure as previous
from nod.telemetry import replay


def main():
    source,out=map(Path,sys.argv[1:]);raw=source.read_bytes()
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    original=replay(source)
    frozen=[rows[0]]+[row for row in rows[1:] if row['event']['kind']!='presentation']
    out.mkdir(parents=True,exist_ok=False)
    base=deepcopy(rows[0]['config'])
    full=configure(base)
    configs={'previous_091':previous(base),'grounded':full}
    for condition in ('no_history','argmax'):
        cfg=deepcopy(full);cfg['listener']['condition']=condition;cfg['research']['condition']=condition
        configs[condition]=cfg
    # Relative cost sensitivity, not parameter fitting on this development log.
    for scale in (.5,2.):
        cfg=deepcopy(full);m=cfg['listener']['policy']['decision_model']
        for key in ('incomplete_receipt_cost','substantive_demand_cost'):m[key]*=scale
        configs[f'receipt_context_cost_x{scale}']=cfg
    report={'source_sha256':hashlib.sha256(raw).hexdigest(),'original_replay_matched':original['matched'],
        'scope':'development replay: frozen old ASR/Jev/VAP, simulated ACKs; no new audio endpoints, no human outcomes, no calibration',
        'conditions':{}}
    for name,cfg in configs.items():
        cfg['experiment']['counterfactual']=True;validate_config(cfg)
        result=counterfactual(frozen,cfg,out/(name+'.jsonl'))
        report['conditions'][name]=result
        print(json.dumps({'condition':name,'functions':result['functions'],'replay_matched':result['replay_matched']},ensure_ascii=False),flush=True)
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
