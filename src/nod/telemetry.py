import json
from pathlib import Path
from nod.config import strict_json, validate_config
from nod.core.engine import create_engine
from nod.core.events import Event


class SessionLog:
    def __init__(self, path, cfg, backend):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental loss of earlier recordings.
        self.file = path.open('x', encoding='utf-8', buffering=1)
        self.sequence = 0
        self.write({'record': 'header', 'log_version': 1, 'config': cfg, 'backend': backend,
                    'source_time_basis': 'session_relative_ms', 'audio_recorded': False})

    def write(self, record):
        self.file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def event(self, event, records):
        self.sequence += 1
        self.write({'record': 'event', 'ingest_seq': self.sequence, 'event': event.to_dict(), 'derived': records})

    def close(self):
        self.file.close()


def replay(path):
    with Path(path).open(encoding='utf-8') as stream:
        header = strict_json(next(stream))
        if header.get('record') != 'header' or header.get('log_version') != 1:
            raise ValueError('Unsupported replay header')
        validate_config(header['config'])
        engine = create_engine(header['config'])
        actions = []
        count = 0
        for line in stream:
            record = strict_json(line)
            count += 1
            if record['record'] != 'event' or record['ingest_seq'] != count:
                raise ValueError('Broken ingest sequence')
            actual = engine.process(Event(**record['event']))
            # JSON normalization makes tuple/list representation irrelevant.
            if json.loads(json.dumps(actual)) != record['derived']:
                raise ValueError(f'Replay mismatch at ingest_seq={count}')
            actions.extend(item['command'] for item in actual if item['kind'] == 'command')
        return {'events': count, 'actions': actions, 'matched': True}
