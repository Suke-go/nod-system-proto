"""Local deployment with before/after hash checks and recoverable backups."""
from datetime import datetime
import hashlib,json,shutil,sys
from pathlib import Path

stage=Path(__file__).resolve().parents[1]
target=Path(r'C:\Users\kosuk\nod').resolve()
plan_file=stage/'deployment-plan.json'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def inside(root,relative):
    result=(root/relative).resolve()
    if not result.is_relative_to(root):raise ValueError('Outside deployment root')
    return result

if sys.argv[1]=='plan':
    paths=[p for name in ('src/nod','tests','configs','docs','scripts') for p in (stage/name).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    paths += [stage/x for x in ('README.md','pyproject.toml','requirements-audio.lock',
        'experiments/pilot/responsive-cases.jsonl','models/sensevoice/model.int8.onnx',
        'models/sensevoice/tokens.txt','models/sensevoice/manifest.json')]
    changes=[]
    for p in paths:
        rel=p.relative_to(stage).as_posix();before,after=digest(inside(target,rel)),digest(p)
        if before!=after:changes.append(dict(path=rel,before_sha256=before,after_sha256=after))
    backup=target/'runs'/('speed-upgrade-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    plan=dict(target=str(target),backup=str(backup),changes=changes)
    plan_file.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({'changes':len(changes),'backup':str(backup)}))
elif sys.argv[1]=='apply':
    plan=json.loads(plan_file.read_text(encoding='utf-8'));backup=Path(plan['backup']).resolve()
    if Path(plan['target']).resolve()!=target or not backup.is_relative_to(target/'runs') or backup.exists():raise ValueError('Invalid target or backup')
    for entry in plan['changes']:
        if digest(inside(target,entry['path']))!=entry['before_sha256']:raise ValueError('Target changed: '+entry['path'])
        if digest(inside(stage,entry['path']))!=entry['after_sha256']:raise ValueError('Source changed: '+entry['path'])
    backup.mkdir(parents=True);shutil.copy2(plan_file,backup/'manifest.json')
    for entry in plan['changes']:
        rel=entry['path'];destination=inside(target,rel)
        if destination.exists():
            saved=inside(backup/'backup',rel);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,saved)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(inside(stage,rel),destination)
        if digest(destination)!=entry['after_sha256']:raise ValueError('Copy verification failed')
    validation=backup/'validation';validation.mkdir()
    for name in ('asr-benchmark.json','speed-tests.xml','validation-summary.json','semantic-verified',
        'live-verified-ja.jsonl','live-verified-ja-result.json','live-verified-en.jsonl','live-verified-en-result.json',
        'counterfactual-verified.jsonl'):
        source=stage/'outputs'/name
        if source.is_dir():shutil.copytree(source,validation/name)
        else:shutil.copy2(source,validation/name)
    shutil.copytree(stage/'dist',backup/'dist')
    print(json.dumps({'copied':len(plan['changes']),'backup':str(backup),'verified':True}))
else:raise ValueError('Use plan or apply')
