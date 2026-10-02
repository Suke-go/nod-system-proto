"""Offline release checks. Never loads credentials or contacts the classifier."""
import json
from pathlib import Path
from importlib.resources import files
from nod import __version__
from nod.config import load_config
from nod.grounded import configure
from nod.experiments.calibration import read,score
from nod.telemetry import replay

root=Path(__file__).resolve().parents[1]
out=root/'outputs';out.mkdir(exist_ok=True)
bundle=root/'experiments/review-20260920'
# Refresh only the generated page, preserving original data and any separate human labels.
data=json.dumps(read(bundle/'annotations.jsonl'),ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
(bundle/'review.html').write_text(files('nod').joinpath('resources/review-heads.html').read_text(encoding='utf-8').replace('__REVIEW_DATA__',data),encoding='utf-8')
result=score(bundle/'annotations.jsonl',bundle/'predictions.jsonl')
cfg=configure(load_config())
old=replay(r'C:\Users\kosuk\nod\sessions\20260919-201238-298917-live.jsonl')
previous=Path(r'C:\Users\kosuk\nod\runs\model-upgrade-20260919-220839\validation')
checks={}
for name in ('grounded-reviewed/grounded.jsonl','receipt-audio-ja.jsonl','receipt-audio-en.jsonl'):
    replayed=replay(previous/name)
    checks[name]={'events':replayed['events'],'actions':len(replayed['actions']),'matched':replayed['matched']}
result.update(version=__version__,default_calibration=cfg['listener'].get('scalar_calibration'),
              default_observation_status=cfg['listener']['observation_model']['status'],
              regression_replays=checks,
              previous_session_replay={'events':old['events'],'actions':len(old['actions']),'matched':old['matched']})
(out/'calibration-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
