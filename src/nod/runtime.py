import asyncio
from pathlib import Path
from nod.core.clock import Clock
from nod.core.engine import create_engine
from nod.core.events import Event, EventBus
from nod.output.transport import ControllerServer, SimulatedController
from nod.semantic.backend import JevBackend, MockBackend
from nod.semantic.coordinator import Coordinator
from nod.telemetry import SessionLog


class Runtime:
    def __init__(self, cfg, backend, log_path):
        self.cfg = cfg
        self.clock = Clock()
        self.bus = EventBus(cfg['runtime']['event_queue_capacity'])
        self.engine = create_engine(cfg)
        self.log = SessionLog(log_path, cfg, backend)
        self.semantic_enabled = cfg.get('listener',{}).get('condition') != 'acoustic_only'
        try:
            model = JevBackend(cfg['semantic']) if backend == 'jev' and self.semantic_enabled else MockBackend(cfg['semantic'])
        except Exception:
            self.log.close()
            raise
        self.semantic = Coordinator(cfg['semantic'], model, self.clock, self.bus.publish)
        self.server = ControllerServer(cfg['output'], self.clock, self.bus.publish)
        self.io_tasks = set()
        self.commands = []
        self.observer = None

    def emit(self, kind, data=None):
        self.bus.publish(Event(kind, self.clock.now_ms(), data or {}))

    def _send(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.io_tasks.add(task)
        # Keep finished tasks until close, so failures remain observable.

    async def actor(self):
        while True:
            event = await self.bus.get()
            if self.bus.overflowed:
                raise RuntimeError('Queue overflow; output stopped')
            records = self.engine.process(event)
            self.log.event(event, records)
            if self.observer:
                self.observer(event, records, self.engine)
            if event.kind == 'asr' and self.semantic_enabled:
                snapshot,context = (self.engine.semantic_input() if hasattr(self.engine,'semantic_input')
                                    else (self.engine.transcripts.current,self.engine.transcripts.recent_context()))
                self.semantic.offer(snapshot,context)
            self.semantic.step()
            for record in records:
                if record['kind'] == 'command':
                    self.commands.append(record)
                    self._send(self.server.send(record['command']))
                elif record['kind'] == 'query_status':
                    self._send(self.server.query(record['action_id']))
            finished = [task for task in self.io_tasks if task.done()]
            for task in finished:
                self.io_tasks.remove(task)
                task.result()

    async def ticker(self):
        while True:
            self.emit('tick')
            await asyncio.sleep(self.cfg['runtime']['control_tick_ms'] / 1000)

    async def synthetic_input(self):
        # Two distinct windows exercise repeated SMALL_NOD with different IDs.
        # Every score and transcript here is a fixture, not a microphone inference.
        for index, text in enumerate(['昨日の出来事を順番に話します。', 'それから駅まで歩いていきました。']):
            for step in range(28):
                now = self.clock.now_ms()
                if step in (0, 3, 6, 12):
                    self.emit('asr', {'utterance_id': f'u{index+1}', 'text': text,
                                     'confidence': 0.95, 'source_ms': now, 'final': step >= 3})
                self.emit('opportunity', {'score': 0.97 if 8 <= step <= 17 else 0.1, 'source_ms': now})
                await asyncio.sleep(0.1)

    async def demo(self, external_controller=False):
        return await self.run(self.synthetic_input,external_controller)

    async def run(self, input_feeder, external_controller=False):
        tasks = []
        try:
            await self.server.start()
            if not external_controller:
                simulator = SimulatedController()
                tasks.append(asyncio.create_task(simulator.run(f'ws://127.0.0.1:{self.server.port}')))
            async with asyncio.timeout(15):
                await self.server.ready.wait()
            actor = asyncio.create_task(self.actor())
            ticker = asyncio.create_task(self.ticker())
            feeder = asyncio.create_task(input_feeder())
            tasks.extend([actor, ticker, feeder])
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if feeder not in done:
                raise RuntimeError('Runtime component stopped unexpectedly')
            # Close the input gate, then drain semantic/control work and acknowledgements.
            self.emit('input_health',{'healthy':False})
            end = self.clock.now_ms() + self.cfg['motor']['execution_timeout_ms'] + 200
            await asyncio.sleep(.1)
            while self.engine.motor.state in ('SENT','ACTIVE') and self.clock.now_ms() < end:
                if actor.done(): actor.result()
                await asyncio.sleep(.05)
            if actor.done():
                actor.result()
            return {'actions': len(self.commands), 'functions': [r['function'] for r in self.commands],
                    'motor_state': self.engine.motor.state, 'semantic_disabled': self.semantic.disabled}
        finally:
            await self.semantic.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.server.close()
            if self.io_tasks:
                await asyncio.gather(*self.io_tasks)
            self.log.close()
