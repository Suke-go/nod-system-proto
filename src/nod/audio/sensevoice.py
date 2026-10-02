"""Local CPU SenseVoice adapter. No model downloads during inference."""
from pathlib import Path
import re
from nod.config import strict_json
from nod.experiments.common import sha256

MODEL_SHA='c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51'
TOKENS_SHA='f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc'
SOURCE_URL='https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2'
ARCHIVE_SHA='7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e'


def verify(root):
    root=Path(root)
    manifest=strict_json((root/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in [('model.int8.onnx',MODEL_SHA),('tokens.txt',TOKENS_SHA)]:
        if manifest.get('sha256',{}).get(name)!=digest or sha256(root/name)!=digest:
            raise ValueError('SenseVoice asset mismatch: '+name)
    return manifest


def normalize_text(text,language):
    if language=='ja':
        # The tokenizer emits spaces between Japanese subwords; preserve spaces
        # between Latin words and do not rewrite lexical content or negation.
        text=re.sub(r'(?<=[\u3040-\u30ff\u3400-\u9fff])\s+(?=[\u3040-\u30ff\u3400-\u9fff])','',text)
    return text.strip()


def prepare(root):
    """Explicit setup only; pinned archive, regular files, no archive scripts."""
    import json,tarfile,shutil,tempfile
    import httpx
    root=Path(root).resolve()
    if (root/'manifest.json').exists():return verify(root)
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sensevoice-',dir=root) as work:
        archive=Path(work)/'model.tar.bz2'
        with httpx.stream('GET',SOURCE_URL,follow_redirects=True,timeout=120) as response:
            response.raise_for_status()
            with archive.open('wb') as stream:
                for block in response.iter_bytes():stream.write(block)
        if sha256(archive)!=ARCHIVE_SHA:raise ValueError('SenseVoice archive hash mismatch')
        hashes={'model.int8.onnx':MODEL_SHA,'tokens.txt':TOKENS_SHA}
        with tarfile.open(archive,'r:bz2') as bundle:
            for name,digest in hashes.items():
                members=[m for m in bundle.getmembers() if m.isfile() and Path(m.name).name==name]
                if len(members)!=1:raise ValueError('Unexpected model archive')
                staged=Path(work)/name
                with bundle.extractfile(members[0]) as source,staged.open('wb') as target:shutil.copyfileobj(source,target)
                if sha256(staged)!=digest:raise ValueError('SenseVoice model hash mismatch')
            for name in hashes:(Path(work)/name).replace(root/name)
        (root/'manifest.json').write_text(json.dumps({'source_url':SOURCE_URL,'archive_sha256':ARCHIVE_SHA,'sha256':hashes},indent=2),encoding='utf-8')
    return verify(root)


class SenseVoiceASR:
    def __init__(self,root,language='ja',threads=4):
        if language not in ('ja','en'):raise ValueError('Choose ja or en')
        self.manifest=verify(root);self.language=language
        import sherpa_onnx
        self.model=sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(Path(root)/'model.int8.onnx'),tokens=str(Path(root)/'tokens.txt'),
            num_threads=threads,provider='cpu',language=language,use_itn=True)
        self.metadata={'backend':'sensevoice_cpu_int8','threads':threads,'language':language,
            'sherpa_onnx_version':sherpa_onnx.__version__,'assets':self.manifest,
            'confidence':'unavailable_stability_proxy_only','streaming':'bounded_redecoded_causal_prefix'}

    def infer(self,samples):
        stream=self.model.create_stream();stream.accept_waveform(16000,samples)
        self.model.decode_stream(stream)
        return {'text':normalize_text(stream.result.text,self.language),'words':[]}
