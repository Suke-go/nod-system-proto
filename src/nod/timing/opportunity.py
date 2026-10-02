from nod.config import probability


class OpportunityGate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.score = 0.0
        self.source_ms = None
        self.candidate_ms = None
        self.low_ms = None
        self.opened_ms = None
        self.window_id = 0
        self.consumed = False
        self.blocked = False

    def update(self, score, source_ms, now_ms):
        score = probability(score)
        if self.source_ms is not None and source_ms <= self.source_ms:
            return False
        if self.source_ms is not None and source_ms-self.source_ms > self.cfg["opportunity_max_age_ms"]:
            self.candidate_ms = self.low_ms = None
            if self.opened_ms is not None:
                self.blocked = True
                self.opened_ms = None
        self.source_ms, self.score = source_ms, score
        if score <= self.cfg["close_threshold"]:
            self.candidate_ms = None
            if self.low_ms is None:
                self.low_ms = source_ms
            elif source_ms-self.low_ms >= self.cfg["rearm_hold_ms"]:
                self.opened_ms = None
                self.blocked = self.consumed = False
        else:
            self.low_ms = None
        if score >= self.cfg["open_threshold"] and not self.blocked and self.opened_ms is None:
            if self.candidate_ms is None:
                self.candidate_ms = source_ms
            elif source_ms-self.candidate_ms >= self.cfg["open_hold_ms"]:
                self.window_id += 1
                self.opened_ms = now_ms
                self.consumed = False
                self.candidate_ms = None
        elif score < self.cfg["open_threshold"]:
            self.candidate_ms = None
        self.expire(now_ms)
        return True

    def expire(self, now_ms):
        if self.opened_ms is not None and (now_ms-self.opened_ms >= self.cfg["expire_open_window_ms"] or
                                           now_ms-self.source_ms > self.cfg["opportunity_max_age_ms"]):
            self.opened_ms = None
            self.blocked = True

    def eligible(self, now_ms):
        self.expire(now_ms)
        return self.opened_ms is not None and not self.consumed and self.score > self.cfg["close_threshold"]

    def consume(self):
        self.consumed = True

