from collections import deque
from dataclasses import asdict, dataclass
from os.path import commonprefix
from nod.config import probability


@dataclass(frozen=True)
class Snapshot:
    utterance_id: str
    revision: int
    repair_epoch: int
    text: str
    stable: str
    source_ms: float
    created_ms: int
    reliability: float
    final: bool = False
    final_reason: str | None = None

    def to_dict(self):
        result=asdict(self)
        if self.final_reason is None:
            result.pop('final_reason')  # Preserve legacy log replay.
        return result


class TranscriptTracker:
    """Tracks raw partials without treating append-only updates as ASR repairs."""
    def __init__(self, config):
        self.cfg = config
        self.current = None
        self.first_seen = []
        self.repairs = deque(maxlen=config["belief"]["revision_window_updates"])
        self.context = deque()

    def update(self, utterance_id, text, source_ms, now_ms, confidence=None, final=False, final_reason=None):
        if final_reason is not None and (not final or final_reason not in ('silence','max_duration','input_end')):
            raise ValueError('Invalid ASR final reason')
        if not isinstance(text, str) or not isinstance(utterance_id, str) or not utterance_id:
            raise ValueError("Invalid transcript or utterance ID")
        if len(text) > self.cfg["semantic"]["max_current_characters"]:
            raise ValueError("Transcript exceeds epoch limit; upstream adapter must split it")
        if confidence is not None:
            confidence = probability(confidence)
        previous = self.current
        same = previous is not None and previous.utterance_id == utterance_id
        if previous is not None and source_ms < previous.source_ms:
            raise ValueError("ASR source time moved backwards")
        if previous is not None and not same:
            self.context.append((previous.source_ms, previous.text))
        if not same:
            self.first_seen = [now_ms] * len(text)
            self.repairs.clear()
            common = ""
            repair = False
        else:
            common = commonprefix([previous.text, text])
            repair = not text.startswith(previous.text)
            self.first_seen = self.first_seen[:len(common)] + [now_ms] * (len(text) - len(common))
        self.repairs.append(int(repair))
        stability = 1.0 if final else (len(common) / max(1, min(len(previous.text), len(text))) if same else 0.0)
        stable_len = 0
        for seen in self.first_seen[:len(common)]:
            if now_ms - seen < self.cfg["asr"]["stable_prefix_min_age_ms"]:
                break
            stable_len += 1
        stable = text if final else text[:stable_len]
        components = {"asr_confidence": confidence, "stability": stability,
                      "revision_quality": 1 - sum(self.repairs) / len(self.repairs)}
        weights = self.cfg["belief"]["reliability_weights"]
        denom = sum(weights[k] for k,v in components.items() if v is not None)
        q = sum(weights[k]*v for k,v in components.items() if v is not None) / denom if denom else 0
        snapshot = Snapshot(utterance_id, previous.revision + 1 if same else 1,
                            previous.repair_epoch + int(repair) if same else 0,
                            text, stable, source_ms, now_ms, q, final, final_reason)
        self.current = snapshot
        cutoff = now_ms - self.cfg["semantic"]["context_window_seconds"] * 1000
        while self.context and self.context[0][0] < cutoff:
            self.context.popleft()
        return snapshot

    def recent_context(self):
        remaining = self.cfg["semantic"]["max_context_characters"]
        selected = []
        for _, text in reversed(self.context):
            if len(text) > remaining:
                break
            selected.append({"speaker": "user", "text": text})
            remaining -= len(text)
        return list(reversed(selected))
