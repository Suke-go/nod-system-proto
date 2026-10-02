import asyncio
import os
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from importlib.resources import files
import httpx
from nod.config import CLASSES, distribution, probability, strict_json


class SemanticError(Exception):
    def __init__(self, reason, *, permanent=False, retry_after_ms=0):
        super().__init__(reason)
        self.reason, self.permanent, self.retry_after_ms = reason, permanent, retry_after_ms


def retry_after(value):
    if not value:
        return 0
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = date.timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0
    # Non-finite headers are invalid; a finite server delay is honored in full.
    import math
    return max(0, int(seconds * 1000)) if math.isfinite(seconds) else 0


class JevBackend:
    """One HTTP attempt per evaluation. Credentials and response bodies are never logged."""
    def __init__(self, cfg, *, client=None, api_key=None):
        self.cfg = cfg
        self.key = api_key if api_key is not None else os.environ.get(cfg['api_key_env'])
        if not self.key:
            raise SemanticError('missing_api_key', permanent=True)
        if cfg['base_url'] != 'https://api.typesafe.ai':
            raise ValueError('This adapter only sends credentials to the official HTTPS endpoint')
        self.client = client or httpx.AsyncClient(follow_redirects=False)
        self.owns_client = client is None
        resource = cfg.get('question_resource','jev-request.json')
        if resource not in ('jev-request.json','jev-semantics.json','jev-observation.json','jev-observation-v3.json'):
            raise ValueError('Unsupported semantic question resource')
        self.questions = strict_json(files('nod').joinpath('resources/'+resource).read_text(encoding='utf-8'))['questions']

    async def evaluate(self, state):
        try:
            async with asyncio.timeout(self.cfg['total_deadline_ms'] / 1000):
                response = await self.client.post(
                    self.cfg['base_url'] + '/v1/systemone',
                    headers={'Authorization': 'Bearer ' + self.key},
                    json={'model': self.cfg['model'], 'state': state, 'questions': self.questions},
                    timeout=self.cfg['total_deadline_ms'] / 1000)
        except (TimeoutError, httpx.TimeoutException):
            raise SemanticError('timeout') from None
        except httpx.RequestError:
            raise SemanticError('network_error') from None
        if response.status_code != 200:
            status = response.status_code
            transient = status in (408, 429) or status >= 500
            raise SemanticError(f'http_{status}', permanent=not transient,
                                retry_after_ms=retry_after(response.headers.get('Retry-After')))
        try:
            body = strict_json(response.text)
            if self.cfg.get('schema') == 'listener-observation-v2':
                from nod.semantic.sensor import parse
                return parse(body)
            if self.cfg.get('schema') == 'listener-semantics-v1':
                from nod.semantic.frames import SCHEMA,FRAME_ORDER,frame_distribution,function_distribution
                answer = body['answers']['semantic_frame']
                p = frame_distribution(answer['probabilities'])
                if (answer['type']!='choice' or answer['choice'] not in FRAME_ORDER
                        or p[FRAME_ORDER.index(answer['choice'])]<max(p)-1e-6
                        or not isinstance(body['model'],str)):
                    raise ValueError('Invalid joint semantic response')
                return {'schema':SCHEMA,'frames':p,'probabilities':function_distribution(p),
                        'model':body['model'],'confidence':probability(answer['confidence'])}
            answer = body['answers']['listener_function']
            p = distribution(answer['probabilities'])
            if answer['type'] != 'choice' or answer['choice'] not in CLASSES:
                raise ValueError('Unexpected choice response')
            if p[CLASSES.index(answer['choice'])] < max(p) - 1e-6:
                raise ValueError('Choice contradicts probabilities')
            confidence = probability(answer['confidence'])
            if not isinstance(body['model'], str):
                raise ValueError('Missing model')
            return {'probabilities': p, 'model': body['model'], 'confidence': confidence}
        except (ValueError, KeyError, TypeError, IndexError):
            raise SemanticError('invalid_response') from None

    async def close(self):
        if self.owns_client:
            await self.client.aclose()


class MockBackend:
    """A fixed synthetic distribution; this is not a language understanding model."""
    def __init__(self,cfg=None):
        self.cfg=cfg or {}

    async def evaluate(self, state):
        await asyncio.sleep(0.025)
        if self.cfg.get('schema')=='listener-observation-v2':
            from nod.semantic.sensor import SCHEMA,fixture,compatibility
            obs=fixture(completion=.1)
            return {'schema':SCHEMA,'observation':obs,'probabilities':compatibility(obs),
                    'model':'synthetic-observation','confidence':{}}
        if self.cfg.get('schema')=='listener-semantics-v1':
            from nod.semantic.frames import SCHEMA,fixture_frame,function_distribution
            p=fixture_frame('explanation_open')
            return {'schema':SCHEMA,'frames':p,'probabilities':function_distribution(p),
                    'model':'synthetic-explanation-open','confidence':.98}
        return {'probabilities': (0.001, 0.995, 0.003, 0.001),
                'model': 'synthetic-continuer', 'confidence': 0.98}

    async def close(self):
        pass
