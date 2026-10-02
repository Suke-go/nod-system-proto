import asyncio
import pytest
from nod.config import load_config
from nod.asr.stability import TranscriptTracker
from nod.core.clock import VirtualClock
from nod.core.engine import Engine
from nod.core.events import Event
from nod.semantic.backend import SemanticError
from nod.semantic.coordinator import Coordinator


def test_repair_invalidates_semantics_even_at_epoch_rollover():
    cfg = load_config(); cfg['belief']['max_semantic_epoch_ms'] = 1000
    engine = Engine(cfg)
    engine.process(Event('asr',0,{'utterance_id':'u','text':'合格','source_ms':0,'final':True}))
    snapshot = engine.transcripts.current.to_dict()
    engine.process(Event('semantic_result',10,{'request':{'seq':1,'snapshot':snapshot,'dispatched_ms':1},
        'result':{'probabilities':[.001,.001,.001,.997]}}))
    assert engine.belief.at(10)[3] > .9
    engine.process(Event('asr',1000,{'utterance_id':'u','text':'不合格','source_ms':1000,'final':True}))
    assert engine.semantic is None
    assert engine.belief.at(1000) == pytest.approx(cfg['belief']['initial'])


def test_future_source_rejected_before_action():
    engine = Engine(load_config())
    with pytest.raises(ValueError, match='future'):
        engine.process(Event('opportunity',100,{'score':1,'source_ms':200}))


def test_delayed_previous_utterance_cannot_replace_current_transcript():
    tracker = TranscriptTracker(load_config())
    tracker.update('u2', 'new', 500, 500)
    with pytest.raises(ValueError, match='backwards'):
        tracker.update('u1', 'old', 100, 600)
    assert tracker.current.utterance_id == 'u2'


def test_experiment_diagnostics_explain_suppressed_actions():
    cfg = load_config(); cfg['logging']['decision_diagnostics'] = True
    engine = Engine(cfg)
    diagnostics = engine.process(Event('tick',0))[-1]
    assert diagnostics['kind'] == 'decision_status'
    assert 'opportunity_unavailable' in diagnostics['reasons']
    assert 'semantic_missing' in diagnostics['reasons']


def test_append_lag_allows_only_generic_continuer():
    cfg = load_config(); engine = Engine(cfg)
    engine.process(Event('controller_ready',0,{'neutral_expression':True}))
    engine.process(Event('asr',0,{'utterance_id':'u','text':'合格','source_ms':0,'final':True}))
    snapshot = engine.transcripts.current.to_dict()
    engine.process(Event('asr',400,{'utterance_id':'u','text':'合格したと思ったら','source_ms':400,'final':False}))
    engine.process(Event('semantic_result',450,{'request':{'seq':1,'snapshot':snapshot,'dispatched_ms':100},
        'result':{'probabilities':[.001,.001,.001,.997]}}))
    engine.process(Event('opportunity',500,{'score':1,'source_ms':500}))
    result = engine.process(Event('opportunity',600,{'score':1,'source_ms':600}))
    assert not any(r.get('function') in ('understanding','empathic') for r in result)


def test_transient_backoff_honors_retry_after_and_dispatches_latest():
    async def run():
        class Backend:
            def __init__(self): self.calls = []
            async def evaluate(self,state):
                self.calls.append(state)
                if len(self.calls) == 1:
                    raise SemanticError('http_429',retry_after_ms=5000)
                return {'probabilities':[.1,.7,.1,.1]}
            async def close(self): pass
        cfg = load_config(); clock = VirtualClock(); tracker = TranscriptTracker(cfg)
        backend = Backend(); events = []
        co = Coordinator(cfg['semantic'], backend, clock, events.append)
        co.offer(tracker.update('u','a',0,0),[])
        clock.advance(150); co.step(); await co.task
        clock.advance(4900); co.offer(tracker.update('u','ab',4900,4900),[])
        clock.advance(5050); co.step(); assert co.task is None
        clock.advance(5150); co.step(); await co.task
        assert len(backend.calls) == 2 and backend.calls[-1]['current_partial'] == 'b'
        assert events[0].kind == 'semantic_error' and events[1].kind == 'semantic_result'
        await co.close()
    asyncio.run(run())


def test_pending_input_replaced_when_current_reverts_to_last_dispatched():
    async def run():
        class Backend:
            async def evaluate(self, state): return {'probabilities':[.1,.7,.1,.1]}
            async def close(self): pass
        cfg = load_config(); clock = VirtualClock(); tracker = TranscriptTracker(cfg)
        co = Coordinator(cfg['semantic'],Backend(),clock,lambda event:None)
        first = tracker.update('u','a',0,0)
        co.offer(first,[]); clock.advance(150); co.step(); await co.task
        from dataclasses import replace
        co.offer(replace(first,text='ab'),[])
        assert co.pending is not None
        co.offer(first,[])
        assert co.pending is None
        await co.close()
    asyncio.run(run())
