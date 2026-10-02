import asyncio
import json
import math
import time
import uuid
from importlib.resources import files
from jsonschema import Draft202012Validator, ValidationError
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from nod.config import strict_json
from nod.core.events import Event

SCHEMA = strict_json(files('nod').joinpath('resources/contracts.json').read_text(encoding='utf-8'))


def validate_message(message, kind):
    schema = {'$ref': '#/$defs/' + kind, '$defs': SCHEMA['$defs']}
    Draft202012Validator(schema).validate(message)


def clock_estimate(t0, t1, t2, t3):
    if t3 < t0 or t2 < t1 or (t3-t0) < (t2-t1):
        raise ValueError('Invalid clock exchange')
    offset = ((t1-t0)+(t2-t3)) / 2
    uncertainty = ((t3-t0)-(t2-t1)) / 2
    return offset, uncertainty


def command_expired(command, controller_now_ns, offset_ns, uncertainty_ns):
    # Offset is controller minus server. Use the worst-case age, including transit.
    created = command['created_monotonic_ns'] + offset_ns
    age = controller_now_ns - created
    return age < -uncertainty_ns or age + uncertainty_ns >= command['ttl_ms'] * 1_000_000


class ControllerServer:
    def __init__(self, cfg, clock, publish):
        self.cfg, self.clock, self.publish = cfg, clock, publish
        self.session_id = str(uuid.uuid4())
        self.epoch = 0
        self.connection = None
        self.ready = asyncio.Event()
        self.server = None
        self.claimed = False
        self.clock_id = None
        self.last_command = None

    async def start(self):
        self.server = await serve(self._handler, self.cfg['bind_host'], self.cfg['port'],
                                  max_size=65536, max_queue=16, compression=None)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handler(self, ws):
        if self.claimed:
            await ws.close(code=1008, reason='One controller per session')
            return
        self.claimed = True
        is_ready = False
        try:
            async with asyncio.timeout(3):
                hello = strict_json(await ws.recv())
                validate_message(hello, 'controller_hello')
                if not {'SMALL_NOD', 'STRONG_NOD'} <= set(hello['supported_actions']) or not hello['supports_status_query'] or hello['deduplication_window_ms'] < 60000:
                    raise ValueError('Required controller capabilities missing')
                self.epoch += 1
                self.clock_id = hello['controller_clock_id']
                t0 = time.monotonic_ns()
                await ws.send(json.dumps({'kind': 'clock_ping', 't0': t0}))
                pong = strict_json(await ws.recv())
                t3 = time.monotonic_ns()
                if pong['kind'] != 'clock_pong' or pong['t0'] != t0:
                    raise ValueError('Clock exchange mismatch')
                offset, uncertainty = clock_estimate(t0, pong['t1'], pong['t2'], t3)
                if uncertainty > 50_000_000:
                    raise ValueError('Clock uncertainty exceeds 50ms')
                await ws.send(json.dumps({'kind': 'session_ready', 'session_id': self.session_id,
                                          'connection_epoch': self.epoch, 'offset_ns': offset,
                                          'uncertainty_ns': uncertainty}))
            self.connection, is_ready = ws, True
            self.ready.set()
            self.publish(Event('controller_ready', self.clock.now_ms(), {
                'neutral_expression': hello['empathic_expression_style'] == 'neutral_acknowledgement'
                and 'EMPATHIC_EXPRESSION' in hello['supported_actions'],
                'supported_expressions':hello.get('supported_expressions',['neutral'])}))
            async for raw in ws:
                message = strict_json(raw)
                validate_message(message, 'controller_event')
                if message['session_id'] != self.session_id or message['connection_epoch'] != self.epoch or message['controller_clock_id'] != self.clock_id:
                    raise ValueError('Controller session mismatch')
                if self.last_command and message['command_id'] == self.last_command['command_id'] and message['action_id'] == self.last_command['action_id']:
                    self.publish(Event('feedback', self.clock.now_ms(), {
                        'action_id': message['action_id'], 'status': message['status']}))
        except (ConnectionClosed, TimeoutError, ValueError, KeyError, TypeError, ValidationError):
            await ws.close(code=1008, reason='Invalid or disconnected controller')
        finally:
            self.connection = None
            self.ready.clear()
            self.claimed = False
            if is_ready:
                self.publish(Event('controller_disconnected', self.clock.now_ms()))

    async def send(self, command):
        if not self.connection:
            self.publish(Event('controller_disconnected', self.clock.now_ms()))
            return
        wire = {key: value for key, value in command.items() if key != 'at_ms'}
        wire.update(kind='action_command', schema_version=1, session_id=self.session_id,
                    connection_epoch=self.epoch,
                    created_monotonic_ns=self.clock.origin_ns + command['at_ms'] * 1_000_000)
        validate_message(wire, 'action_command')
        self.last_command = wire
        try:
            async with asyncio.timeout(0.2):
                await self.connection.send(json.dumps(wire))
        except (ConnectionClosed, TimeoutError):
            self.publish(Event('controller_disconnected', self.clock.now_ms()))

    async def query(self, action_id):
        if self.connection and self.last_command:
            try:
                async with asyncio.timeout(0.2):
                    await self.connection.send(json.dumps({'kind': 'status_query', 'action_id': action_id,
                                                           'command_id': self.last_command['command_id'],
                                                           'session_id': self.session_id,
                                                           'connection_epoch': self.epoch}))
            except (ConnectionClosed, TimeoutError):
                pass

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()


class SimulatedController:
    """Wire-compatible timer simulator. No animation or physical actuator is driven."""
    def __init__(self):
        self.clock_id = str(uuid.uuid4())
        self.history = {}
        self.executions = []
        self.tasks = set()

    async def run(self, url):
        async with connect(url, max_size=65536, compression=None) as ws:
            await ws.send(json.dumps({'schema_version': 1, 'kind': 'controller_hello',
                'controller_id': 'nod-simulator', 'connection_epoch': 0,
                'controller_clock_id': self.clock_id,
                'supported_actions': ['SMALL_NOD', 'STRONG_NOD', 'EMPATHIC_EXPRESSION'],
                'supports_prepare': False, 'supports_stop': False, 'supports_status_query': True,
                'deduplication_window_ms': 60000, 'empathic_expression_style': 'neutral_acknowledgement',
                'supported_expressions':['neutral','attentive','warm','concerned']}))
            ping = strict_json(await ws.recv())
            t1 = time.monotonic_ns()
            if ping['kind'] != 'clock_ping':
                raise ValueError('Expected clock ping')
            await ws.send(json.dumps({'kind': 'clock_pong', 't0': ping['t0'], 't1': t1, 't2': time.monotonic_ns()}))
            session = strict_json(await ws.recv())
            if session['kind'] != 'session_ready' or not all(math.isfinite(session[k]) for k in ('offset_ns', 'uncertainty_ns')):
                raise ValueError('Invalid clock synchronization')
            try:
                async for raw in ws:
                    message = strict_json(raw)
                    if message.get('session_id') != session['session_id'] or message.get('connection_epoch') != session['connection_epoch']:
                        raise ValueError('Wrong session')
                    if message['kind'] == 'status_query':
                        entry = self.history.get((session['session_id'], message['action_id']))
                        if entry:
                            await self._feedback(ws, entry[0], entry[1], session)
                        else:
                            await self._feedback(ws, {'action_id': message['action_id'], 'command_id': message.get('command_id', 'unknown')}, 'unknown', session)
                    elif message['kind'] == 'action_command':
                        validate_message(message, 'action_command')
                        await self._execute(ws, message, session)
            finally:
                # A simulator task must not survive a lost connection as an unobserved action.
                pending = list(self.tasks)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                self.tasks.clear()

    async def _feedback(self, ws, command, status, session):
        await ws.send(json.dumps({'schema_version': 1, 'kind': 'controller_event',
            'session_id': session['session_id'], 'connection_epoch': session['connection_epoch'],
            'command_id': command['command_id'], 'action_id': command['action_id'], 'status': status,
            'controller_clock_id': self.clock_id, 'controller_monotonic_ns': time.monotonic_ns(), 'reason': None}))

    async def _execute(self, ws, command, session):
        now = time.monotonic_ns()
        self.history = {k:v for k,v in self.history.items() if v[1] in ('accepted', 'started') or now-v[2] < 60_000_000_000}
        key = (session['session_id'], command['action_id'])
        if key in self.history:
            await self._feedback(ws, command, self.history[key][1], session)
            return
        invalid = command['command'] != 'execute' or command['action'] == 'CLAP' or command['onset_delay_ms'] != 0
        busy = any(v[1] in ('accepted', 'started') for v in self.history.values())
        if invalid or busy or command_expired(command, now, session['offset_ns'], session['uncertainty_ns']):
            self.history[key] = (command, 'rejected', now)
            await self._feedback(ws, command, 'rejected', session)
            return
        self.history[key] = (command, 'accepted', now)
        await self._feedback(ws, command, 'accepted', session)
        task = asyncio.create_task(self._animate(ws, command, session, key))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _animate(self, ws, command, session, key):
        try:
            self.history[key] = (command, 'started', time.monotonic_ns())
            self.executions.append(command)
            await self._feedback(ws, command, 'started', session)
            await asyncio.sleep(command['duration_ms'] / 1000)
            self.history[key] = (command, 'completed', time.monotonic_ns())
            await self._feedback(ws, command, 'completed', session)
        except asyncio.CancelledError:
            self.history[key] = (command, 'cancelled', time.monotonic_ns())
            raise
        except ConnectionClosed:
            self.history[key] = (command, 'failed', time.monotonic_ns())
