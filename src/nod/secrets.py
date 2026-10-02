import os
from pathlib import Path


def load_api_key_file(path='.env'):
    """Read only the expected secret, without shell evaluation or logging."""
    path = Path(path)
    if os.environ.get('TYPESAFE_API_KEY') or not path.is_file():
        return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        key, sep, value = line.strip().partition('=')
        if sep and key == 'TYPESAFE_API_KEY':
            value = value.strip().strip('\"\'')
            if value and not any(c.isspace() for c in value):
                os.environ[key] = value
                return
    raise ValueError('The API key file does not contain a valid TYPESAFE_API_KEY entry')
