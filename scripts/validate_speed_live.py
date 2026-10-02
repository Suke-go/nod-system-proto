"""Explicit authored WAV experiment. Never opens the microphone."""
import json,sys
from pathlib import Path
from argparse import Namespace
from nod.config import load_config
from nod.live_session import run_live
from nod.secrets import load_api_key_file
from nod.telemetry import replay

root=Path(__file__).resolve().parents[1]
load_api_key_file(r'C:\Users\kosuk\nod\.env')
audio=Path(r'C:\Users\kosuk\nod\runs\incremental-upgrade-20260919-170108\validation\authored-speech')
prefix=sys.argv[1] if len(sys.argv)>1 else 'verified'
for lang in ('ja','en'):
    path=root/'outputs'/f'live-{prefix}-{lang}.jsonl'
    args=Namespace(seconds=60,dashboard_port=0,models=r'C:\Users\kosuk\nod\models\local-audio',
        language=lang,asr_backend='sensevoice',sensevoice_models=str(root/'models/sensevoice'),
        partial_ms=None,port=0,config=None,response=None,condition=None,observation_model=None,calibration=None,
        wav=str(audio/f'{lang}.wav'),external_controller=False,rms_threshold=.012,semantic='jev',device=None,open_browser=False)
    result=run_live(args,load_config(),str(path));result['replay']=replay(path)
    (root/'outputs'/f'live-{prefix}-{lang}-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'language':lang,'functions':result['functions'],'replay_matched':result['replay']['matched']},ensure_ascii=False),flush=True)
