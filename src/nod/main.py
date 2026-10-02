import argparse
import asyncio
from datetime import datetime
import json
import os
import getpass
import sys
from pathlib import Path
from nod.config import load_config, strict_json
from nod.output.transport import SimulatedController
from nod.runtime import Runtime
from nod.semantic.backend import JevBackend, SemanticError
from nod.telemetry import replay


def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='nod: reproducible local-audio and Jev experiments')
    parser.add_argument('--config', help='Full runtime JSON configuration')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Synthetic audio observations through a real local WebSocket')
    demo.add_argument('--semantic', choices=['mock', 'jev'], default='mock')
    demo.add_argument('--log', default=None)
    demo.add_argument('--external-controller', action='store_true')
    demo.add_argument('--port', type=int, help='Use 0 for an automatically allocated port')
    demo.add_argument('--prompt-key', action='store_true', help='Read the API key without echoing or saving it')
    rep = commands.add_parser('replay', help='Offline deterministic verification; never sends commands')
    rep.add_argument('path')
    commands.add_parser('doctor', help='Show configuration availability without secrets')
    check_parser = commands.add_parser('jev-check', help='One online call using a synthetic Japanese utterance')
    check_parser.add_argument('--prompt-key', action='store_true')
    controller = commands.add_parser('controller', help='Run the timer-based controller simulator')
    controller.add_argument('--url', default='ws://127.0.0.1:8765')
    commands.add_parser('audio-devices', help='List microphone inputs; never records')
    rec = commands.add_parser('record', help='Record a short local 16kHz mono WAV')
    rec.add_argument('path'); rec.add_argument('--seconds',type=int,default=12)
    rec.add_argument('--device',type=int)
    models = commands.add_parser('models-prepare',help='Download and verify local ASR and VAP-BC assets')
    models.add_argument('--models',default='models/local-audio')
    models.add_argument('--language',choices=['ja','en'],default='ja')
    fast_models=commands.add_parser('sensevoice-prepare',help='Explicitly download and verify the pinned local SenseVoice model')
    fast_models.add_argument('--models',default='models/sensevoice')
    live = commands.add_parser('live',help='Local microphone/WAV ASR + VAP, Jev semantics and a PC avatar')
    live.add_argument('--language',choices=['ja','en'],default='ja')
    live.add_argument('--semantic',choices=['jev','mock'],default='jev')
    live.add_argument('--response',choices=['production','legacy-production','bayes','bayes-vap','bayes-no-history','direct','presentation','responsive','conservative'],default=None,
                      help='Default: production (typed semantics, joint state, revision-aware feedback). Other profiles retain the legacy experiments.')
    live.add_argument('--models',default='models/local-audio')
    source=live.add_mutually_exclusive_group()
    source.add_argument('--device',type=int)
    source.add_argument('--wav',help='Play a 16kHz mono PCM16 file through the live pipeline at wall-clock speed')
    live.add_argument('--seconds',type=int,default=300)
    live.add_argument('--partial-ms',type=int,default=None,help='Default 400 ms for SenseVoice, 800 ms for Whisper')
    live.add_argument('--asr-backend',choices=['auto','sensevoice','whisper'],default='auto')
    live.add_argument('--sensevoice-models',default='models/sensevoice')
    live.add_argument('--rms-threshold',type=float,default=.012)
    live.add_argument('--dashboard-port',type=int,default=8766)
    live.add_argument('--port',type=int,default=8765)
    live.add_argument('--external-controller',action='store_true')
    live.add_argument('--open-browser',action='store_true')
    live.add_argument('--log')
    live.add_argument('--prompt-key',action='store_true')
    live.add_argument('--condition',choices=['full','no_history','argmax','acoustic_only'])
    live.add_argument('--observation-model',help='Fitted listener observation-density artifact')
    live.add_argument('--calibration',help='Reviewed semantic-calibrate result for this schema')
    prep = commands.add_parser('prepare-audio',help='Build a causal-prefix timeline from a 1–30s recording')
    prep.add_argument('wav'); prep.add_argument('--out',required=True)
    prep.add_argument('--models',default='models/local-audio')
    prep.add_argument('--partial-ms',type=int,default=1000)
    exp = commands.add_parser('experiment',help='Run a prepared audio timeline with Jev or mock')
    exp.add_argument('timeline'); exp.add_argument('--semantic',choices=['mock','jev'],default='mock')
    exp.add_argument('--log',default=None); exp.add_argument('--prompt-key',action='store_true')
    exp.add_argument('--port',type=int,default=0)
    evaluation = commands.add_parser('jev-eval',help='Evaluate independent text cases, stopping on first failure')
    evaluation.add_argument('cases'); evaluation.add_argument('--out',required=True)
    evaluation.add_argument('--semantic',choices=['mock','jev'],default='mock')
    evaluation.add_argument('--limit',type=int,default=10)
    evaluation.add_argument('--prompt-key',action='store_true')
    analyze = commands.add_parser('analyze',help='Summarize semantic failures, timing and simulated actions')
    analyze.add_argument('log'); analyze.add_argument('--out')
    research=commands.add_parser('research-review',help='Offline controlled ablations and blinded annotation export')
    research.add_argument('log');research.add_argument('--out',required=True)
    scoring=commands.add_parser('research-score',help='Score only human-reviewed semantic labels')
    scoring.add_argument('annotations');scoring.add_argument('predictions')
    typed_eval=commands.add_parser('semantic-eval',help='Evaluate the typed semantic model on bilingual diagnostic cases')
    typed_eval.add_argument('cases');typed_eval.add_argument('--out',required=True)
    typed_eval.add_argument('--semantic',choices=['jev','mock'],default='jev')
    typed_eval.add_argument('--limit',type=int,default=20)
    for name in ('semantic-score','semantic-calibrate'):
        typed_score=commands.add_parser(name,help='Score reviewed frame labels or fit temperature on the calibration split')
        typed_score.add_argument('annotations');typed_score.add_argument('predictions');typed_score.add_argument('--out')
    listener_eval=commands.add_parser('listener-eval',help='Typed sensor diagnostics on authored cases')
    listener_eval.add_argument('cases');listener_eval.add_argument('--out',required=True)
    listener_eval.add_argument('--semantic',choices=['jev','mock'],default='jev')
    listener_eval.add_argument('--limit',type=int,default=20)
    for name in ('listener-score','listener-fit'):
        listener_score=commands.add_parser(name,help='Score reviewed states or fit joint observation density')
        listener_score.add_argument('annotations');listener_score.add_argument('predictions')
        listener_score.add_argument('--out',required=True)
    args = parser.parse_args()
    try:
        from nod.secrets import load_api_key_file
        load_api_key_file()
        cfg = load_config(args.config)
        if getattr(args, 'prompt_key', False) and not os.environ.get(cfg['semantic']['api_key_env']):
            os.environ[cfg['semantic']['api_key_env']] = getpass.getpass('Jev API key (not saved): ')
        if args.command == 'demo':
            if args.port is not None:
                cfg['output']['port'] = args.port
            path = args.log or str(Path('sessions') / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl'))
            result = asyncio.run(Runtime(cfg, args.semantic, path).demo(args.external_controller))
            result['log'] = str(Path(path).resolve())
            result['inputs'] = 'synthetic'
            result['controller'] = 'external' if args.external_controller else 'timer_simulator'
        elif args.command == 'audio-devices':
            from nod.audio.recording import devices
            result = devices()
        elif args.command == 'record':
            from nod.audio.recording import record
            result = record(args.path,args.seconds,args.device)
        elif args.command == 'models-prepare':
            from nod.audio.models import prepare_models,prepare_english
            result = prepare_models(args.models)
            if args.language=='en': result=prepare_english(args.models)
        elif args.command == 'sensevoice-prepare':
            from nod.audio.sensevoice import prepare
            result=prepare(args.models)
        elif args.command == 'live':
            from nod.live_session import run_live
            path=args.log or str(Path('sessions')/(datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-live.jsonl'))
            result=run_live(args,cfg,path)
        elif args.command == 'prepare-audio':
            from nod.experiments.timeline import prepare_audio
            result = prepare_audio(args.wav,args.models,args.out,args.partial_ms)
        elif args.command == 'experiment':
            from nod.experiments.timeline import load_timeline,feed_timeline
            from nod.experiments.common import sha256
            timeline = load_timeline(args.timeline)
            cfg['output']['port'] = args.port
            cfg['experiment'] = {'timeline_sha256':sha256(args.timeline),'timing_mode':timeline['timing_mode'],
                                 'audio_sha256':timeline.get('audio_sha256'),'models':timeline.get('models'),
                                 'prominence':timeline.get('prominence')}
            path = args.log or str(Path('sessions')/(datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-experiment.jsonl'))
            runtime = Runtime(cfg,args.semantic,path)
            result = asyncio.run(runtime.run(lambda:feed_timeline(runtime,timeline)))
            result.update(log=str(Path(path).resolve()),inputs='prepared_audio_timeline',semantic=args.semantic)
        elif args.command == 'analyze':
            from nod.experiments.report import analyze_session
            from nod.experiments.common import save_json
            result = analyze_session(args.log)
            if args.out: save_json(args.out,result)
        elif args.command == 'research-review':
            from nod.experiments.research import review_session
            result=review_session(args.log,args.out)
        elif args.command == 'research-score':
            from nod.experiments.research import score_annotations
            result=score_annotations(args.annotations,args.predictions)
        elif args.command == 'listener-eval':
            from nod.experiments.listener import evaluate
            from nod.responsive import configure
            result=asyncio.run(evaluate(cfg if cfg.get('listener') else configure(cfg),args.cases,args.out,args.semantic,args.limit))
        elif args.command in ('listener-score','listener-fit'):
            from nod.experiments.listener import score,fit,save
            result=(fit if args.command=='listener-fit' else score)(args.annotations,args.predictions)
            save(args.out,result)
        elif args.command == 'semantic-eval':
            from nod.experiments.interaction import evaluate
            from nod.production import configure
            result=asyncio.run(evaluate(cfg if cfg.get('interaction') else configure(cfg),args.cases,args.out,args.semantic,args.limit))
        elif args.command in ('semantic-score','semantic-calibrate'):
            from nod.experiments.interaction import score,save
            result=score(args.annotations,args.predictions,args.command=='semantic-calibrate')
            if args.out:save(args.out,result)
        elif args.command == 'jev-eval':
            from nod.experiments.evaluate import evaluate_cases
            result = asyncio.run(evaluate_cases(cfg,args.cases,args.out,args.semantic,args.limit))
        elif args.command == 'replay':
            result = replay(args.path)
        elif args.command == 'doctor':
            result = {'api_key_configured': bool(os.environ.get(cfg['semantic']['api_key_env'])),
                      'jev_model': cfg['semantic']['model'],
                      'acn_manifest_exists': Path(cfg['prominence']['model_manifest']).is_file(),
                      'local_audio_manifest_exists':Path('models/local-audio/manifest.json').is_file(),
                      'local_asr_adapter':'SenseVoice CPU if installed; Whisper CPU fallback / ja or en',
                      'sensevoice_manifest_exists':Path('models/sensevoice/manifest.json').is_file(),
                      'timing_adapter':'MaAI Japanese or English VAP-BC / live chunks',
                      'live_microphone_inference':True,
                      'milestone':'live bilingual experimental agent'}
        elif args.command == 'jev-check':
            async def check():
                backend = JevBackend(cfg['semantic'])
                try:
                    return await backend.evaluate({'stable_transcript': '昨日は駅まで歩いていきました。',
                                                   'current_partial': '', 'recent_context': []})
                finally:
                    await backend.close()
            result = asyncio.run(check())
        else:
            asyncio.run(SimulatedController().run(args.url))
            result = {'stopped': True}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except SemanticError as exc:
        hint = 'Use --prompt-key or set TYPESAFE_API_KEY.' if exc.reason == 'missing_api_key' else 'Check model access, connection and the configured deadline.'
        parser.exit(2, f'Jev: {exc.reason}. {hint}\n')
    except ImportError:
        parser.exit(2, 'Missing optional audio dependencies. Install the audio extra or requirements-audio.lock.\n')
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Stopped.\n')
