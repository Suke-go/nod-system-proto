import asyncio
import json
import time
import pytest
from nod.config import load_config, strict_json
from nod.core.clock import Clock
from nod.output.transport import ControllerServer, SimulatedController, clock_estimate, command_expired
from nod.policy.motor import Motor
from nod.runtime import Runtime
from nod.telemetry import replay


def test_clock_offset_and_ttl_include_network_transit():
    offset, uncertainty = clock_estimate(100, 1110, 1120, 130)
    assert offset == 1000 and uncertainty == 10
    command = {'created_monotonic_ns':100_000_000, 'ttl_ms':250}
    assert not command_expired(command, 300_000_000, 0, 1_000_000)
    assert command_expired(command, 351_000_000, 0, 1_000_000)
    assert command_expired(command, 100, 0, 1)
    with pytest.raises(ValueError):
        clock_estimate(100, 1110, 1200, 130)


def test_simulator_deduplicates_and_rejects_expired_command():
    class Socket:
        def __init__(self): self.messages = []
        async def send(self, raw): self.messages.append(strict_json(raw))
    async def run():
        simulator = SimulatedController(); ws = Socket()
        session = {'session_id':'s','connection_epoch':1,'offset_ns':0,'uncertainty_ns':0}
        cmd = {'action_id':'a1','command_id':'c1','command':'execute','action':'SMALL_NOD',
               'onset_delay_ms':0,'created_monotonic_ns':time.monotonic_ns(), 'ttl_ms':250,'duration_ms':10}
        await simulator._execute(ws, cmd, session)
        await simulator._execute(ws, cmd, session)
        await asyncio.gather(*simulator.tasks)
        assert len(simulator.executions) == 1
        expired = dict(cmd, action_id='a2', command_id='c2', created_monotonic_ns=time.monotonic_ns()-1_000_000_000)
        await simulator._execute(ws, expired, session)
        assert ws.messages[-1]['status'] == 'rejected'
        assert len(simulator.executions) == 1
    asyncio.run(run())


def test_real_websocket_handshake_command_feedback_and_status_query():
    async def run():
        cfg = load_config(); cfg['output']['port'] = 0
        clock = Clock(); events = []
        server = ControllerServer(cfg['output'], clock, events.append)
        simulator = SimulatedController()
        await server.start()
        task = asyncio.create_task(simulator.run(f'ws://127.0.0.1:{server.port}'))
        try:
            async with asyncio.timeout(3): await server.ready.wait()
            cmd = Motor(cfg['motor']).start('SMALL_NOD', .2, clock.now_ms(), 1)
            await server.send(cmd)
            async with asyncio.timeout(2):
                while not any(e.kind == 'feedback' and e.data['status'] == 'completed' for e in events):
                    await asyncio.sleep(.01)
            await server.send(cmd)
            await server.query(cmd['action_id'])
            await asyncio.sleep(.02)
            assert len(simulator.executions) == 1
            assert any(e.kind == 'controller_ready' for e in events)
            assert [e.data['status'] for e in events if e.kind == 'feedback'][:3] == ['accepted','started','completed']
        finally:
            await server.close()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_end_to_end_demo_and_offline_replay(tmp_path):
    async def run():
        cfg = load_config(); cfg['output']['port'] = 0
        path = tmp_path/'demo.jsonl'
        runtime = Runtime(cfg, 'mock', path)
        result = await runtime.demo()
        assert result['actions'] == 2 and result['functions'] == ['continuer','continuer']
        replayed = replay(path)
        assert replayed['matched'] and len(replayed['actions']) == 2
        assert replayed['actions'][0]['action_id'] != replayed['actions'][1]['action_id']
    asyncio.run(run())
