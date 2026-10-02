import asyncio
import json
import httpx
import pytest
from nod.config import load_config
from nod.core.clock import VirtualClock
from nod.asr.stability import TranscriptTracker
from nod.semantic.backend import JevBackend, SemanticError, retry_after
from nod.semantic.coordinator import Coordinator


def payload():
    return {'model': 'jev-test', 'answers': {'listener_function': {
        'type': 'choice', 'choice': 'continuer', 'confidence': .8,
        'probabilities': {'no_bc': .1,'continuer':.7,'understanding':.1,'empathic':.1}}}}


def test_http_request_contract_and_probability_order():
    async def run():
        def handler(request):
            assert request.url == 'https://api.typesafe.ai/v1/systemone'
            body = json.loads(request.content)
            assert set(body) == {'model','state','questions'}
            assert set(body['state']) == {'stable_transcript','current_partial','recent_context'}
            assert request.headers['Authorization'] == 'Bearer test-only'
            return httpx.Response(200, json=payload())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = JevBackend(load_config()['semantic'], client=client, api_key='test-only')
            result = await backend.evaluate({'stable_transcript':'話','current_partial':'します','recent_context':[]})
            assert result['probabilities'] == pytest.approx((.1,.7,.1,.1))
    asyncio.run(run())


@pytest.mark.parametrize('status,permanent', [(401,True),(403,True),(400,True),(429,False),(529,False),(503,False)])
def test_http_errors_are_typed_without_body_or_retry(status, permanent):
    async def run():
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(status, text='private-response', headers={'Retry-After':'3'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = JevBackend(load_config()['semantic'], client=client, api_key='test-only')
            with pytest.raises(SemanticError) as error:
                await backend.evaluate({})
            assert error.value.permanent == permanent
            assert error.value.retry_after_ms == 3000
            assert 'private-response' not in str(error.value) and len(calls) == 1
    asyncio.run(run())


@pytest.mark.parametrize('bad', ['nan','missing','choice'])
def test_invalid_api_distributions_rejected(bad):
    async def run():
        body = payload()
        answer = body['answers']['listener_function']
        if bad == 'nan': answer['probabilities']['continuer'] = float('nan')
        elif bad == 'missing': del answer['probabilities']['empathic']
        else: answer['choice'] = 'empathic'
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=json.dumps(body)))) as client:
            backend = JevBackend(load_config()['semantic'], client=client, api_key='test-only')
            with pytest.raises(SemanticError, match='invalid_response'):
                await backend.evaluate({})
    asyncio.run(run())


def test_whole_request_deadline():
    async def run():
        async def handler(request):
            await asyncio.sleep(.1)
            return httpx.Response(200, json=payload())
        cfg = load_config()['semantic']; cfg['total_deadline_ms'] = 10
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(SemanticError, match='timeout'):
                await JevBackend(cfg, client=client, api_key='test').evaluate({})
    asyncio.run(run())


class ControlledBackend:
    def __init__(self):
        self.wait = asyncio.Event(); self.states = []
    async def evaluate(self, state):
        self.states.append(state)
        await self.wait.wait()
        return {'probabilities':(.1,.7,.1,.1)}
    async def close(self): pass


def test_latest_pending_single_flight_no_starvation():
    async def run():
        cfg = load_config(); clock = VirtualClock(); events = []
        backend = ControlledBackend()
        co = Coordinator(cfg['semantic'], backend, clock, events.append)
        tracker = TranscriptTracker(cfg)
        for now, text in [(0,'a'), (100,'ab'), (200,'abc'), (300,'abcd')]:
            clock.advance(now)
            co.offer(tracker.update('u',text,now,now),[]); co.step()
        await asyncio.sleep(0)
        assert len(backend.states) == 1 and backend.states[0]['current_partial'].endswith('bcd')
        for now, text in [(400,'abcde'),(500,'abcdef')]:
            clock.advance(now)
            co.offer(tracker.update('u',text,now,now),[]); co.step()
        assert len(backend.states) == 1
        backend.wait.set(); await co.task
        clock.advance(700); co.step(); await co.task
        assert len(backend.states) == 2
        assert backend.states[-1]['stable_transcript'] + backend.states[-1]['current_partial'] == 'abcdef'
        assert [e.data['request']['seq'] for e in events] == [1,2]
        await co.close()
    asyncio.run(run())


def test_backoff_and_permanent_failure_no_fake_observation():
    async def run():
        class Failure(ControlledBackend):
            async def evaluate(self, state):
                raise SemanticError('http_401', permanent=True)
        cfg = load_config(); clock = VirtualClock(); events = []
        co = Coordinator(cfg['semantic'], Failure(), clock, events.append)
        tracker = TranscriptTracker(cfg)
        co.offer(tracker.update('u','hello',0,0),[])
        clock.advance(150); co.step(); await co.task
        assert co.disabled and events[0].kind == 'semantic_error'
        assert not any(e.kind == 'semantic_result' for e in events)
        await co.close()
    asyncio.run(run())


def test_retry_after_invalid_and_long_duration():
    assert retry_after('NaN') == 0
    assert retry_after('120') == 120000
