"""Deploy reviewed local changes with source/target hashes and a complete backup."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

stage=Path(__file__).resolve().parents[1]
target=Path(r'C:\Users\kosuk\nod').resolve()
plan_path=stage/'calibration-deployment.json'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def inside(root,rel):
    path=(root/rel).resolve()
    if not path.is_relative_to(root):raise ValueError('Outside root')
    return path

if sys.argv[1]=='plan':
    paths=[p for name in ('src/nod','tests','docs','scripts','experiments/review-20260920')
           for p in (stage/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    paths.append(stage/'pyproject.toml');changes=[]
    for path in paths:
        rel=path.relative_to(stage).as_posix();before=sha(inside(target,rel));after=sha(path)
        if before!=after:changes.append(dict(path=rel,before_sha256=before,after_sha256=after))
    backup=target/'runs'/('calibration-upgrade-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    plan=dict(target=str(target),backup=str(backup),changes=changes)
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps(plan,indent=2))
elif sys.argv[1]=='apply':
    plan=json.loads(plan_path.read_text(encoding='utf-8'));backup=Path(plan['backup']).resolve()
    if Path(plan['target']).resolve()!=target or not backup.is_relative_to(target/'runs') or backup.exists():raise ValueError('Invalid destination')
    for item in plan['changes']:
        if sha(inside(target,item['path']))!=item['before_sha256']:raise ValueError('Target changed: '+item['path'])
        if sha(inside(stage,item['path']))!=item['after_sha256']:raise ValueError('Stage changed: '+item['path'])
    backup.mkdir(parents=True);shutil.copy2(plan_path,backup/'manifest.json')
    for item in plan['changes']:
        original=inside(target,item['path'])
        if original.is_file():
            saved=inside(backup/'before',item['path']);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,saved)
    shutil.copytree(stage/'outputs',backup/'validation')
    for item in plan['changes']:
        destination=inside(target,item['path']);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(inside(stage,item['path']),destination)
        if sha(destination)!=item['after_sha256']:raise ValueError('Copy mismatch')
    print(json.dumps(dict(copied=len(plan['changes']),backup=str(backup),verified=True)))
else:raise ValueError('Use plan or apply')
