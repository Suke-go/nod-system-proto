import json
import math
import pytest
from nod.config import load_config, distribution
from nod.core.events import Event, EventBus
from nod.core.engine import Engine
from nod.asr.stability import TranscriptTracker
from nod.belief.filter import BeliefFilter, Transition
from nod.policy.motor import Motor
from nod.policy.selector import ExpectedUtilityPolicy
from nod.timing.opportunity import OpportunityGate
from nod.telemetry import SessionLog, replay


@pytest.fixture
def cfg():
    return load_config()


@pytest.mark.parametrize('value', [[0, 0, 0, 0], [1, 1, 0, 0], [float('nan'), 0, 0, 1], [True, 0, 0, 0]])
def test_distribution_rejects_invalid(value):
    with pytest.raises(ValueError):
        distribution(value)


def test_transition_is_independent_of_tick_count(cfg):
    transition = Transition(cfg['belief']['transition']['stationary'], 4000)
    b = (0.01, 0.02, 0.07, 0.9)
    predicted = b
    for _ in range(100):
        predicted = transition.predict(predicted, 40)
    assert predicted == pytest.approx(transition.predict(b, 4000))


def test_zero_quality_and_replacement_not_double_counted(cfg):
    f = BeliefFilter(cfg['belief'])
    f.open_epoch(0)
    p = (0.01, 0.02, 0.07, 0.9)
    assert f.observe(p, 0, 100) == pytest.approx(cfg['belief']['initial'])
    f.observe(p, 1, 200)
    repeated = f.observe(p, 1, 400)
    fresh = BeliefFilter(cfg['belief']).observe(p, 1, 400)
    assert repeated == pytest.approx(fresh)
    f.invalidate(450)
    assert f.at(450) == pytest.approx(cfg['belief']['initial'])


def test_asr_stability_append_repair_and_missing_confidence(cfg):
    tracker = TranscriptTracker(cfg)
    first = tracker.update('u1', '試験に', 0, 0)
    assert first.stable == '' and 0 <= first.reliability <= 1
    second = tracker.update('u1', '試験に合格', 350, 350)
    assert second.stable == '試験に' and second.repair_epoch == 0
    repair = tracker.update('u1', '試験に不合格', 400, 400)
    assert repair.repair_epoch == 1 and repair.stable == '試験に'
    tracker.update('u2', '次の話', 500, 500)
    assert tracker.recent_context() == [{'speaker': 'user', 'text': '試験に不合格'}]


def test_opportunity_distinct_sources_and_rearm(cfg):
    gate = OpportunityGate(cfg['timing'])
    gate.update(.9, 0, 0)
    gate.update(.9, 0, 100)
    assert not gate.eligible(100)
    gate.update(.9, 100, 100)
    assert gate.eligible(100)
    gate.consume()
    for now in range(200, 3000, 100):
        gate.update(.9, now, now)
        assert not gate.eligible(now)
    for now in (3000, 3100, 3200):
        gate.update(.1, now, now)
    gate.update(.9, 3300, 3300)
    gate.update(.9, 3400, 3400)
    assert gate.eligible(3400) and gate.window_id == 2


def test_stale_opportunity_cannot_reopen_with_high_only(cfg):
    gate = OpportunityGate(cfg['timing'])
    gate.update(.9, 0, 0); gate.update(.9, 100, 100)
    assert not gate.eligible(351)
    gate.update(.9, 400, 400); gate.update(.9, 500, 500)
    assert not gate.eligible(500)


def test_policy_conservative_gates_and_strength_separation(cfg):
    policy = ExpectedUtilityPolicy(cfg['policy'])
    assert policy.choose(0, (0, 0, 0, 1)).function == 'none'
    assert policy.choose(1, (0, 0, 0, 1)).function == 'empathic'
    assert policy.choose(1, (0, 0, 0, 1), specific=False).function != 'empathic'
    assert policy.embody('continuer', 1)[0] == 'SMALL_NOD'
    assert policy.embody('empathic', .5, False)[0] == 'STRONG_NOD'
    for f, (lo, hi) in cfg['policy']['intensity_ranges'].items():
        assert policy.embody(f, 0)[1] == lo
        assert policy.embody(f, 1)[1] == pytest.approx(hi)


def test_motor_out_of_order_completion_and_repeated_nod(cfg):
    motor = Motor(cfg['motor'])
    first = motor.start('SMALL_NOD', .2, 0, 1)
    motor.feedback(first['action_id'], 'started', 20)
    motor.feedback(first['action_id'], 'accepted', 30)
    assert motor.state == 'ACTIVE'
    motor.feedback(first['action_id'], 'completed', 400)
    assert not motor.feedback(first['action_id'], 'started', 450)
    motor.tick(1299); assert motor.state == 'COOLDOWN'
    motor.tick(1300)
    second = motor.start('SMALL_NOD', .2, 1300, 2)
    assert first['action_id'] != second['action_id']


def test_motor_unknown_delivery_no_retry(cfg):
    motor = Motor(cfg['motor'])
    command = motor.start('SMALL_NOD', .2, 0, 1)
    motor.tick(301)
    assert motor.state == 'FAULT'
    motor.feedback(command['action_id'], 'unknown', 350)
    with pytest.raises(RuntimeError):
        motor.start('SMALL_NOD', .2, 5000, 2)
    motor.feedback(command['action_id'], 'completed', 5100)
    assert motor.state == 'COOLDOWN'


def transcript(engine, now=0, text='続けて話します', u='u1'):
    return engine.process(Event('asr', now, {'utterance_id': u, 'text': text, 'source_ms': now, 'confidence': .99, 'final': True}))


def response(engine, now=50, seq=1, snapshot=None, p=(.001,.995,.003,.001)):
    return Event('semantic_result', now, {'request': {'seq': seq, 'dispatched_ms': now-10,
        'snapshot': snapshot or engine.transcripts.current.to_dict()},
        'result': {'probabilities': p, 'model': 'test', 'confidence': .9}})


def test_engine_rejects_old_repaired_duplicate_and_expired(cfg):
    engine = Engine(cfg)
    transcript(engine)
    original = engine.transcripts.current.to_dict()
    transcript(engine, 100, '話すのはやめます')
    assert engine.process(response(engine, 110, snapshot=original))[0]['reason'] == 'repaired_input'
    assert engine.process(response(engine, 120))[0]['reason'] == 'accepted'
    assert engine.process(response(engine, 130))[0]['reason'] == 'duplicate_or_superseded'
    old = engine.transcripts.current.to_dict()
    transcript(engine, 200, u='u2')
    assert engine.process(response(engine, 220, 2, old))[0]['reason'] == 'prior_utterance'
    assert engine.process(response(engine, 1800, 3))[0]['reason'] == 'expired'


def test_engine_action_requires_fresh_semantics_and_controller(cfg):
    engine = Engine(cfg)
    transcript(engine)
    engine.process(response(engine))
    engine.process(Event('opportunity', 100, {'score': .97, 'source_ms':100}))
    records = engine.process(Event('opportunity', 200, {'score': .97, 'source_ms':200}))
    assert not any(r['kind'] == 'command' for r in records)
    records = engine.process(Event('controller_ready', 210))
    command = next(r for r in records if r['kind'] == 'command')
    assert command['prominence_degraded'] and command['command']['action'] == 'SMALL_NOD'
    assert not any(r['kind'] == 'command' for r in engine.process(Event('tick', 220)))


def test_replay_compares_all_derived_records_and_detects_tampering(cfg, tmp_path):
    engine = Engine(cfg)
    path = tmp_path / 'session.jsonl'
    logger = SessionLog(path, cfg, 'test')
    events = [Event('controller_ready', 0), Event('asr', 0, {'utterance_id':'u1','text':'話します','source_ms':0,'confidence':.99,'final':True})]
    for event in events:
        logger.event(event, engine.process(event))
    event = response(engine, 50)
    logger.event(event, engine.process(event))
    for now in [100,200]:
        event = Event('opportunity', now, {'score': .99, 'source_ms': now})
        logger.event(event, engine.process(event))
    logger.close()
    assert replay(path)['matched']
    lines = path.read_text(encoding='utf-8').splitlines()
    item = json.loads(lines[-1]); item['derived'] = []
    lines[-1] = json.dumps(item)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Replay mismatch'):
        replay(path)


def test_queue_overflow_latches_and_does_not_silently_drop():
    bus = EventBus(1)
    bus.publish(Event('tick', 0))
    with pytest.raises(RuntimeError):
        bus.publish(Event('tick', 1))
    assert bus.overflowed
