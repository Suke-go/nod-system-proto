import time


class Clock:
    def __init__(self):
        self.origin_ns = time.monotonic_ns()

    def now_ms(self):
        return (time.monotonic_ns() - self.origin_ns) // 1_000_000


class VirtualClock:
    def __init__(self):
        self.value = 0

    def advance(self, at_ms):
        if at_ms < self.value:
            raise ValueError("Clock cannot move backwards")
        self.value = at_ms

    def now_ms(self):
        return self.value

