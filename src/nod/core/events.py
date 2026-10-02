from dataclasses import asdict, dataclass, field
import asyncio
import math


@dataclass(frozen=True)
class Event:
    kind: str
    at_ms: int
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.at_ms, bool) or not isinstance(self.at_ms, int) or self.at_ms < 0:
            raise ValueError("at_ms must be a nonnegative session time")

    def to_dict(self):
        return asdict(self)


class EventBus:
    """Overflow is latched: the runtime stops outputs, rather than silently dropping."""
    def __init__(self, capacity):
        self.queue = asyncio.Queue(capacity)
        self.overflowed = False

    def publish(self, event):
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.overflowed = True
            raise RuntimeError("Control event queue overflow") from None

    async def get(self):
        return await self.queue.get()


def source_time(data, now_ms):
    value = data["source_ms"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= now_ms:
        raise ValueError("Invalid or future audio source time")
    return value
