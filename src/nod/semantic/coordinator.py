import asyncio
import hashlib
import json
from nod.core.events import Event
from nod.semantic.backend import SemanticError


class Coordinator:
    """Single flight, replaceable pending input; never retries an old request."""
    def __init__(self, cfg, backend, clock, publish):
        self.cfg, self.backend, self.clock, self.publish = cfg, backend, clock, publish
        self.pending = None
        self.first_pending_ms = None
        self.last_change_ms = 0
        self.last_dispatch_ms = -10**12
        self.last_key = None
        self.task = None
        self.sequence = 0
        self.disabled = False
        self.backoff_until = 0
        self.failures = 0

    def offer(self, snapshot, context):
        if not snapshot.text.strip():
            self.pending = None
            self.first_pending_ms = None
            return
        state = {'stable_transcript': snapshot.stable,
                 'current_partial': snapshot.text[len(snapshot.stable):],
                 'recent_context': context}
        key = hashlib.sha256(json.dumps([snapshot.utterance_id, snapshot.repair_epoch, state],
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if key == self.last_key:
            self.pending = None
            self.first_pending_ms = None
            return
        if self.disabled:
            return
        if self.pending and self.pending['key'] == key:
            # Refresh source metadata without postponing debounce for identical text.
            self.pending['snapshot'] = snapshot.to_dict()
            return
        if self.pending is None:
            self.first_pending_ms = self.clock.now_ms()
        self.pending = {'key': key, 'snapshot': snapshot.to_dict(), 'state': state}
        self.last_change_ms = self.clock.now_ms()

    def step(self):
        now = self.clock.now_ms()
        if self.disabled or self.pending is None or self.task is not None:
            return
        if now < self.backoff_until or now - self.last_dispatch_ms < self.cfg['min_dispatch_interval_ms']:
            return
        if now - self.last_change_ms < self.cfg['debounce_ms'] and now - self.first_pending_ms < self.cfg['max_coalesce_wait_ms']:
            return
        request, self.pending = self.pending, None
        self.first_pending_ms = None
        if now - request['snapshot']['source_ms'] > self.cfg['source_max_age_ms']:
            return
        self.sequence += 1
        request.update(seq=self.sequence, dispatched_ms=now)
        self.last_dispatch_ms, self.last_key = now, request['key']
        self.task = asyncio.create_task(self._run(request))

    async def _run(self, request):
        try:
            async with asyncio.timeout(self.cfg['total_deadline_ms'] / 1000):
                result = await self.backend.evaluate(request['state'])
            self.failures = 0
            self.publish(Event('semantic_result', self.clock.now_ms(), {'request': request, 'result': result}))
        except (SemanticError, TimeoutError) as exc:
            error = exc if isinstance(exc, SemanticError) else SemanticError('timeout')
            self.disabled = error.permanent
            self.failures += 1
            delay = min(self.cfg['max_backoff_ms'], self.cfg['initial_backoff_ms'] * 2 ** min(16, self.failures - 1))
            self.backoff_until = self.clock.now_ms() + max(delay, error.retry_after_ms)
            self.publish(Event('semantic_error', self.clock.now_ms(),
                               {'seq': request['seq'], 'reason': error.reason, 'permanent': error.permanent,
                                'backoff_until_ms': self.backoff_until}))
        finally:
            self.task = None

    async def close(self):
        if self.task:
            task = self.task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.backend.close()
