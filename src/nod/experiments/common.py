import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def quantiles(values):
    if not values:
        return {'count': 0, 'p50': None, 'p95': None, 'max': None}
    values = sorted(values)
    return {'count': len(values), 'p50': values[math.ceil(.5*len(values))-1],
            'p95': values[math.ceil(.95*len(values))-1], 'max': values[-1]}
