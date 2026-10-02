"""ASR stability that distinguishes punctuation formatting from word revisions."""
from dataclasses import replace
from os.path import commonprefix
from nod.asr.stability import TranscriptTracker
from nod.semantic.evidence import words


class IncrementalTranscriptTracker(TranscriptTracker):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.word_seen = []

    def update(self, utterance_id, text, source_ms, now_ms, confidence=None, final=False, final_reason=None):
        previous = self.current
        same = previous is not None and previous.utterance_id == utterance_id
        old = words(previous.text) if same else ''
        current = words(text)
        common = commonprefix([old,current]) if same else ''
        revised = same and not current.startswith(old)
        snapshot = super().update(utterance_id,text,source_ms,now_ms,confidence,final,final_reason)
        # Parent validation/context handling remains shared; replace only the
        # stability computation and lexical-repair epoch for this profile.
        self.repairs[-1] = int(revised)
        self.word_seen = (self.word_seen[:len(common)] if same else []) + [now_ms]*(len(current)-len(common))
        stable_count = 0
        for seen in self.word_seen[:len(common)]:
            if now_ms-seen < self.cfg['asr']['stable_prefix_min_age_ms']:break
            stable_count += 1
        stable_end = 0
        # Text lengths are bounded to the ASR epoch, so offset mapping is bounded.
        for i in range(1,len(text)+1):
            if len(words(text[:i]))>stable_count:break
            stable_end=i
        stability=1. if final else len(common)/max(1,min(len(old),len(current))) if same else 0.
        values={'asr_confidence':confidence,'stability':stability,
                'revision_quality':1-sum(self.repairs)/len(self.repairs)}
        weights=self.cfg['belief']['reliability_weights']
        total=sum(weights[k] for k,v in values.items() if v is not None)
        reliability=sum(weights[k]*v for k,v in values.items() if v is not None)/total if total else 0.
        self.current=replace(snapshot,stable=text if final else text[:stable_end],reliability=reliability,
            repair_epoch=previous.repair_epoch+int(revised) if same else 0)
        return self.current
