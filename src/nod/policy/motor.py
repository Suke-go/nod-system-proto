class Motor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "IDLE"
        self.command = None
        self.sent_ms = None
        self.completed_ms = None
        self.acknowledged = False
        self.fault_reason = None
        self.sequence = 0

    def tick(self, now_ms):
        if self.state == "COOLDOWN" and now_ms-self.completed_ms >= self.cfg["cooldown_after_completion_ms"]:
            self.state = "IDLE"
        if self.state in ("SENT", "ACTIVE"):
            if not self.acknowledged and now_ms-self.sent_ms > self.cfg["ack_timeout_ms"]:
                self.fail("ack_timeout")
            elif now_ms-self.sent_ms > self.cfg["execution_timeout_ms"]:
                self.fail("execution_timeout")

    def start(self, action, intensity, now_ms, opportunity_id):
        if self.state != "IDLE":
            raise RuntimeError("Motor is not idle")
        self.sequence += 1
        self.command = {"command_id": f"c{self.sequence}", "action_id": f"a{self.sequence}",
                        "command": "execute", "action": action, "intensity": intensity,
                        "duration_ms": self.cfg["default_durations_ms"][action],
                        "ttl_ms": self.cfg["command_ttl_ms"], "onset_delay_ms": 0,
                        "opportunity_id": f"o{opportunity_id}", "decision_id": f"d{self.sequence}",
                        "at_ms": now_ms}
        self.sent_ms, self.acknowledged = now_ms, False
        self.state, self.fault_reason = "SENT", None
        return dict(self.command)

    def feedback(self, action_id, status, now_ms):
        if not self.command or action_id != self.command["action_id"]:
            return False
        if self.state in ("IDLE", "COOLDOWN"):
            return False
        if status == "accepted":
            self.acknowledged = True
        elif status == "started":
            self.acknowledged = True
            self.state = "ACTIVE"
        elif status in ("completed", "cancelled"):
            self.acknowledged = True
            self.completed_ms = now_ms
            self.state = "COOLDOWN"
            self.fault_reason = None
        elif status in ("rejected", "failed", "unknown"):
            self.fail(status)
        else:
            return False
        return True

    def fail(self, reason):
        self.state, self.fault_reason = "FAULT", reason

