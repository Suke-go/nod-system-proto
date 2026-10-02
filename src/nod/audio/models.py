import json
import os
import contextlib
import io
from pathlib import Path
from nod.config import probability, strict_json
from nod.experiments.common import sha256, save_json


def verify_assets(root):
    root = Path(root).resolve()
    path = root/'manifest.json'
    if not path.is_file():
        raise ValueError('Local audio assets missing; run models-prepare first')
    manifest = strict_json(path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('asr_model') != 'base'
            or manifest.get('mode') != 'bc' or manifest.get('language') != 'jp'
            or manifest.get('frame_rate') != 10 or manifest.get('context_seconds') != 20):
        raise ValueError('Unsupported local model manifest')
    required = {'asr/model.bin','asr/config.json','asr/tokenizer.json','timing/model.pt','timing/cpc.pt'}
    if not required <= set(manifest['sha256']):
        raise ValueError('Incomplete model manifest')
    for relative, expected in manifest['sha256'].items():
        target = (root/relative).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha256(target) != expected:
            raise ValueError(f'Model asset missing or changed: {relative}')
    return manifest


class LocalASR:
    def __init__(self, root, language='ja', *, word_timestamps=True):
        if language not in ('ja','en'): raise ValueError('Choose ja or en')
        self.language = language
        self.word_timestamps=word_timestamps
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(Path(root)/'asr'), device='cpu',compute_type='int8',
                                  cpu_threads=4,num_workers=1,local_files_only=True)

    def infer(self, samples):
        segments, _ = self.model.transcribe(samples, language=self.language,beam_size=1,
            condition_on_previous_text=False,word_timestamps=self.word_timestamps,vad_filter=True)
        segments = list(segments)
        return {'text':''.join(s.text for s in segments).strip(),
                'words':[{'text':w.word,'start_sample':round(w.start*16000),
                          'end_sample':round(w.end*16000),'probability':w.probability}
                         for s in segments for w in (s.words or [])]}


class LocalTiming:
    def __init__(self, root, language='ja'):
        if language not in ('ja','en'): raise ValueError('Choose ja or en')
        import torch
        from maai import Maai, MaaiInput
        torch.set_num_threads(2)
        root = Path(root)
        checkpoint = root / ('timing/en.pt' if language == 'en' else 'timing/model.pt')
        if not (root/'timing/cpc.pt').is_file() or not checkpoint.is_file():
            raise ValueError('Local timing assets are missing')
        self.model = Maai(mode='bc',lang='en' if language == 'en' else 'jp',frame_rate=10,context_len_sec=20,
            audio_ch1=MaaiInput.Chunk(),audio_ch2=MaaiInput.Zero(),device='cpu',
            model_type='normal',local_model=str(checkpoint),cpc_model=str(root/'timing/cpc.pt'))

    def reset(self):
        self.model.reset_runtime_state()

    def infer(self, samples):
        import numpy as np
        if len(samples) != 1600:
            raise ValueError('MaAI 10Hz expects exactly 1600 samples per call')
        with contextlib.redirect_stdout(io.StringIO()):
            self.model.process(np.asarray(samples,dtype=np.float32),np.zeros(1600,dtype=np.float32))
        # A bounded read turns an upstream API mismatch into an explicit error.
        result = self.model.result_dict_queue.get(timeout=5)
        score = np.asarray(result['p_bc']).reshape(-1)
        if len(score) != 1:
            raise ValueError('Expected one system-channel backchannel score')
        return probability(float(score[0]))


def prepare_models(root):
    """Explicit setup command. Downloads weights; inference uses only these local files."""
    from importlib.metadata import version
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    import shutil
    root = Path(root).resolve()
    if (root/'manifest.json').exists():
        return verify_assets(root)
    root.mkdir(parents=True,exist_ok=True)
    api = HfApi()
    asr_repo, timing_repo = 'Systran/faster-whisper-base','maai-kyoto/vap_bc_jp'
    asr_rev = api.model_info(asr_repo).sha
    timing_rev = api.model_info(timing_repo).sha
    snapshot_download(asr_repo,revision=asr_rev,local_dir=root/'asr',
        allow_patterns=['model.bin','config.json','tokenizer.json','vocabulary.*','preprocessor_config.json','README.md'])
    timing_name = 'vap-bc_state_dict_jp_10hz_20000msec.pt'
    timing_path = hf_hub_download(timing_repo,filename=timing_name,revision=timing_rev,cache_dir=root/'download-cache')
    (root/'timing').mkdir(exist_ok=True)
    shutil.copyfile(timing_path,root/'timing/model.pt')
    # Download CPC explicitly to the project; do not populate the user's global torch cache.
    cpc_url = 'https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt'
    if not (root/'timing/cpc.pt').exists():
        import httpx
        temporary = root/'timing/cpc.pt.partial'
        with httpx.stream('GET',cpc_url,timeout=60,follow_redirects=True) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for block in response.iter_bytes(): stream.write(block)
        if not sha256(temporary).startswith('d0f474de'):
            raise ValueError('CPC download checksum does not match its published filename')
        temporary.replace(root/'timing/cpc.pt')
    from maai import Maai, MaaiInput
    import torch
    torch.set_num_threads(2)
    Maai(mode='bc',lang='jp',frame_rate=10,context_len_sec=20,
         audio_ch1=MaaiInput.Chunk(),audio_ch2=MaaiInput.Zero(),device='cpu',
         model_type='normal',local_model=str(root/'timing/model.pt'),cpc_model=str(root/'timing/cpc.pt'))
    asset_files = [p for folder in ('asr','timing') for p in (root/folder).glob('*') if p.is_file()]
    manifest = {'schema_version':1,'asr_model':'base','asr_repo':asr_repo,'asr_revision':asr_rev,
        'timing_repo':timing_repo,'timing_revision':timing_rev,'timing_file':timing_name,
        'mode':'bc','language':'jp','frame_rate':10,'context_seconds':20,'model_type':'normal',
        'runtime_device':'cpu','asr_compute_type':'int8',
        'cpc_url':cpc_url,
        'versions':{name:version(name) for name in ('maai','torch','faster-whisper','ctranslate2')},
        'sha256':{p.relative_to(root).as_posix():sha256(p) for p in asset_files}}
    save_json(root/'manifest.json',manifest)
    return verify_assets(root)


def prepare_english(root):
    from huggingface_hub import HfApi,hf_hub_download
    import shutil
    root=Path(root).resolve(); manifest=verify_assets(root)
    if 'timing/en.pt' in manifest['sha256']: return manifest
    repo='maai-kyoto/vap_bc_en'; revision=HfApi().model_info(repo).sha
    name='vap-bc_state_dict_en_10hz_20000msec.pt'
    path=hf_hub_download(repo,filename=name,revision=revision,cache_dir=root/'download-cache')
    shutil.copyfile(path,root/'timing/en.pt')
    manifest['sha256']['timing/en.pt']=sha256(root/'timing/en.pt')
    manifest['english_timing']={'repo':repo,'revision':revision,'file':name,'language':'en','context_seconds':20}
    temporary=root/'manifest.en.json'
    temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(root/'manifest.json')
    return verify_assets(root)
