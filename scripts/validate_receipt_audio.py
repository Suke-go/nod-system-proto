"""Local authored JA/EN WAVs, real ASR/VAP and mock semantics. No microphone/API."""
from argparse import Namespace
from collections import Counter
import json
from pathlib import Path
import sys
from nod.config import load_config
from nod.live_session import run_live
from nod.telemetry import replay

sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parents[1]
audio=Path(r'C:\Users\kosuk\nod\runs\incremental-upgrade-20260919-170108\validation\authored-speech')
for lang in ('ja','en'):
    path=root/'outputs'/f'receipt-audio-{lang}.jsonl'
    args=Namespace(seconds=60,dashboard_port=0,models=r'C:\Users\kosuk\nod\models\local-audio',
        language=lang,asr_backend='sensevoice',sensevoice_models=r'C:\Users\kosuk\nod\models\sensevoice',
        partial_ms=None,port=0,config=None,response=None,condition=None,observation_model=None,calibration=None,
        wav=str(audio/f'{lang}.wav'),external_controller=False,rms_threshold=.012,semantic='mock',device=None,open_browser=False)
    result=run_live(args,load_config(),str(path))
    result['replay_matched']=replay(path)['matched']
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()][1:]
    result['asr_job_states']=dict(Counter(row['event']['data']['state'] for row in rows if row['event']['kind']=='asr_job'))
    result['scope']='authored WAVs; real local SenseVoice/VAP; mock semantic observations; not semantic accuracy or HRI validation'
    (path.parent/(path.stem+'-result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
