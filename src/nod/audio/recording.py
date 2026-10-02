from pathlib import Path
import wave


def devices():
    import sounddevice as sd
    return [dict(index=i, name=d['name'], channels=d['max_input_channels'],
                 default_sample_rate=d['default_samplerate'])
            for i,d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]


def record(path, seconds=12, device=None):
    """Explicit CLI action only: never called by doctor, tests or model setup."""
    import numpy as np
    import sounddevice as sd
    if not 1 <= seconds <= 30:
        raise ValueError('Pilot recordings must be between 1 and 30 seconds')
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    sd.check_input_settings(device=device,channels=1,dtype='int16',samplerate=16000)
    with path.open('xb') as raw:
        print(f'Recording {seconds}s at 16kHz mono into {path}.')
        samples = sd.rec(int(seconds*16000),samplerate=16000,channels=1,dtype='int16',device=device)
        try:
            status = sd.wait()
            if status:
                raise RuntimeError(f'Audio device reported data loss: {status}')
        except BaseException:
            sd.stop()
            raise
        with wave.open(raw,'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(samples.astype('<i2',copy=False).tobytes())
    return {'path':str(path.resolve()),'duration_seconds':seconds,'sample_rate':16000,
            'peak':float(np.max(np.abs(samples.astype(np.int32))))/32768}


def load_wave(path):
    import numpy as np
    with wave.open(str(path),'rb') as stream:
        if stream.getframerate() != 16000 or stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
            raise ValueError('Expected uncompressed 16kHz mono PCM16 WAV; convert explicitly before use')
        if not 16000 <= stream.getnframes() <= 30*16000:
            raise ValueError('Pilot audio must be between 1 and 30 seconds')
        raw = stream.readframes(stream.getnframes())
    return np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768
