import asyncio
import hashlib
from importlib.resources import files
from pathlib import Path
from nod.audio.models import verify_assets,LocalASR,LocalTiming
from nod.audio.live import LiveInput
from nod.dashboard import Dashboard
from nod.runtime import Runtime
from nod.config import validate_config
from nod.experiments.common import sha256


def live_config(cfg,language,port,custom=False,response=None):
    if response not in (None,'production','legacy-production','responsive','conservative','bayes','bayes-vap','bayes-no-history','direct','presentation'):
        raise ValueError('Unknown response profile')
    if custom and response is not None:
        raise ValueError('Use either --config or --response, not both')
    cfg['status']='live_audio_experiment'
    cfg['asr'].update(backend='faster_whisper_cpu_int8',language=language)
    cfg['timing'].update(backend='maai_vap_bc_cpu',language='jp' if language=='ja' else 'en')
    if not cfg['semantic'].get('schema'):
        cfg['semantic']['question_version']='listener-function-ja-en-v2'
    if not custom:
        response=response or 'production'
        cfg['profile']='live_cpu_jev'
        cfg['semantic'].update(total_deadline_ms=1500,source_max_age_ms=3000)
        if response == 'production':
            from nod.grounded import configure
            cfg=configure(cfg)
            from nod.provenance import source_digest
            cfg['research']['implementation_sha256']=source_digest()
        elif response == 'legacy-production':
            from nod.production import configure
            cfg=configure(cfg)
        elif response != 'conservative':
            cfg['profile']='live_responsive_v2'
            cfg['timing'].update(open_threshold=.4,close_threshold=.25)
            cfg['policy']['opportunity_weighting']='gate_only'
            cfg['belief']['observation_rule']='posterior_blend'
            cfg['policy']['understanding_action']='STRONG_NOD'
            cfg['policy']['endpoint_specific_ack']={
                'enabled':True,'minimum_reliability':.65,
                'minimum_probability':.65,'minimum_margin':.20,
                'minimum_belief':.50}
            cfg['motor']['cooldown_after_completion_ms']=1800
            if response in ('bayes','bayes-vap','bayes-no-history','direct','presentation'):
                cfg['profile']='research_'+response.replace('-','_')+'_v2'
                cfg['belief']['observation_rule']='direct' if response=='direct' else 'likelihood_product'
                cfg['belief']['history_mode']='none' if response in ('direct','bayes-no-history') else 'carry'
                cfg['belief']['initial']=[.25]*4
                cfg['belief']['transition']['stationary']=[.25]*4
                cfg['semantic']['refine_exact_final']=True
                cfg['logging']['research_diagnostics']=True
                cfg['policy']['endpoint_specific_ack']['enabled']=response!='bayes-vap'
                cfg['policy']['endpoint_specific_ack']['allowed_final_reasons']=['silence']
                cfg['policy']['stable_clause_ack']=response=='presentation'
                cfg['research']={'method':'reliability_weighted_direct' if response=='direct' else 'generalized_bayes_pseudo_likelihood',
                    'calibration':'identity_unfitted','prior':'symmetric_assumption_not_estimated',
                    'parameters_status':'development_not_validated','protocol_version':'research-v2',
                    'filter_history':cfg['belief']['history_mode'],
                    'stable_clause_extension':response=='presentation',
                    'semantic_context':'shared_recent_20s','endpoint_extension':response!='bayes-vap'}
                from nod.provenance import source_digest
                cfg['research']['implementation_sha256']=source_digest()
    cfg['output']['port']=port
    validate_config(cfg)
    return cfg


def run_live(args,cfg,path):
    if not 1<=args.seconds<=1800: raise ValueError('Choose 1–1800 seconds')
    if not 0<=args.dashboard_port<=65535: raise ValueError('Invalid dashboard port')
    manifest=verify_assets(args.models)
    if args.language=='en' and 'timing/en.pt' not in manifest['sha256']:
        raise ValueError('English timing asset missing; run models-prepare --language en')
    print('Loading local ASR and VAP-BC...',flush=True)
    backend=getattr(args,'asr_backend','auto')
    sense_root=Path(getattr(args,'sensevoice_models','models/sensevoice'))
    if backend=='auto':backend='sensevoice' if (sense_root/'manifest.json').is_file() else 'whisper'
    if backend=='sensevoice':
        from nod.audio.sensevoice import SenseVoiceASR
        asr=SenseVoiceASR(sense_root,args.language)
        asr_manifest=asr.metadata
    else:
        asr=LocalASR(args.models,args.language,word_timestamps=False)
        asr_manifest={'backend':'faster_whisper_cpu_int8','word_timestamps':False,'threads':4}
    timing=LocalTiming(args.models,args.language)
    partial_ms=args.partial_ms or (400 if backend=='sensevoice' else 800)
    # Warm CPU kernels before starting capture, without advancing the live model's audio history.
    import numpy as np
    timing.infer(np.zeros(1600,dtype=np.float32)); timing.reset()
    asr.infer(np.zeros(16000,dtype=np.float32))
    cfg=live_config(cfg,args.language,args.port,bool(args.config),args.response)
    cfg['asr'].update(backend=asr_manifest['backend'],partial_target_interval_ms=partial_ms)
    if getattr(args,'condition',None):
        if 'listener' not in cfg: raise ValueError('--condition requires the listener model')
        cfg['listener']['condition']=args.condition
        cfg['research']['condition']=args.condition
    if getattr(args,'observation_model',None):
        from nod.experiments.listener import load_model
        load_model(cfg,args.observation_model)
    if getattr(args,'calibration',None):
        from nod.config import strict_json
        from nod.semantic.frames import SCHEMA
        artifact=strict_json(Path(args.calibration).read_text(encoding='utf-8-sig'))
        if not cfg.get('interaction') or artifact.get('semantic_schema')!=SCHEMA or artifact.get('calibration')!='temperature_fitted':
            raise ValueError('Calibration artifact does not match the production semantic schema')
        cfg['interaction'].update(temperature=artifact['temperature'],calibration='temperature_fitted')
        cfg['research'].update(calibration='temperature_fitted',calibration_artifact_sha256=sha256(args.calibration))
        validate_config(cfg)
    cfg['experiment']={'input':'paced_wav' if args.wav else 'microphone','language':args.language,
        'models':manifest,'prominence':'unavailable_fixed_low_feature','controller':'external' if args.external_controller else 'timer_simulator',
        'duration_limit_seconds':args.seconds,'partial_ms':partial_ms,'rms_threshold':args.rms_threshold,
        'asr_runtime':asr_manifest,
        'audio_sha256':sha256(args.wav) if args.wav else None}
    cfg['experiment']['semantic_prompt_sha256']=hashlib.sha256(
        files('nod').joinpath('resources/'+cfg['semantic'].get('question_resource','jev-request.json')).read_bytes()).hexdigest()

    async def session():
        runtime=Runtime(cfg,args.semantic,path)
        live=LiveInput(runtime,asr,timing,device=args.device,wav=args.wav,duration=args.seconds,
                       partial_ms=partial_ms,threshold=args.rms_threshold)
        loop=asyncio.get_running_loop()
        def stop():
            live.stop.set(); runtime.emit('input_health',{'healthy':False})
        dashboard=None
        try:
            dashboard=Dashboard(args.language,args.semantic,args.dashboard_port,
                stop=lambda:loop.call_soon_threadsafe(stop),
                rendered=(lambda data:loop.call_soon_threadsafe(runtime.emit,'presentation',data))
                    if cfg.get('listener',{}).get('version')==4 else None)
            def observe(event,records,engine):
                dashboard.observe(event,records,engine)
                dashboard.publish(rms=live.level,asr_ms=live.asr_ms,timing_ms=live.timing_ms,
                                  asr_superseded=live.overwritten_asr)
            runtime.observer=observe
            labels={'listener_grounded_v4_2':'逐次ベイズ聞き手 / 内容に応じた頷き・表情',
                    'listener_responsive_v4':'逐次ベイズ聞き手 v4 / 高速認識・意味に応じた表情',
                    'listener_incremental_v3':'逐次ベイズ聞き手 / 内容の保持・表情',
                    'listener_bayes_v2':'ベイズ聞き手モデル / 知覚・理解・態度',
                    'listener_production_v1':'意味・音声・履歴の統合モデル',
                    'research_bayes_v2':'Bayes研究版 v2 / 履歴あり',
                    'research_presentation_v2':'プレゼン練習 / Bayes + 文の区切りで反応',
                    'research_bayes_no_history_v2':'比較条件 / Bayes履歴なし',
                    'research_direct_v2':'比較条件 / Jev直接対応',
                    'research_bayes_vap_v2':'Bayes研究版 v2 / VAPのみ',
                    'research_bayes_v1':'Bayes研究版 / 終了後反応あり',
                    'research_bayes_vap_v1':'Bayes研究版 / VAPのみ',
                    'live_responsive_v2':'混合モデル v2（比較用）'}
            dashboard.start(); dashboard.publish(status='listening',
                render_telemetry=cfg.get('listener',{}).get('version')==4,
                response=labels.get(cfg['profile'],'従来 / カスタム')+' / '+cfg.get('listener',{}).get('condition',''))
            url=f'http://127.0.0.1:{dashboard.port}'
            print(f'Dashboard: {url}\nController: {"external" if args.external_controller else "PC simulator"}\nLog: {Path(path).resolve()}',flush=True)
            print(f'Response profile: {cfg["profile"]}',flush=True)
            if args.open_browser:
                import webbrowser
                webbrowser.open(url)
            result=await runtime.run(live.run,args.external_controller)
            dashboard.publish(status='finished')
            return {**result,'log':str(Path(path).resolve()),'language':args.language,
                    'input':'paced_wav' if args.wav else 'microphone','asr_ms_last':live.asr_ms,
                    'timing_ms_last':live.timing_ms,'asr_superseded':live.overwritten_asr,
                    'controller':cfg['experiment']['controller']}
        finally:
            if dashboard:
                await asyncio.to_thread(dashboard.close)
            # Also covers an occupied dashboard port before runtime.run starts.
            await runtime.semantic.close()
            runtime.log.close()
    return asyncio.run(session())
