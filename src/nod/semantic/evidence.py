"""Short-lived, source-anchored interpretations; never independent repeated evidence."""
from copy import deepcopy
import re
from nod.belief.listener import ListenerBelief


def words(text):
    # Ignore ASR formatting, not question/exclamation marks or lexical changes.
    text = re.sub(r'[,、。]|(?<!\d)\.|\.(?!\d)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # English word boundaries matter: "nowhere" is not "now here".
    return re.sub(r'(?<![A-Za-z0-9]) | (?![A-Za-z0-9])', '', text)


class EvidenceMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.items = []
        self.sequence = 0

    def reconcile(self, transcript, now):
        kept, revoked = [], []
        for item in self.items:
            if item['utterance_id'] != transcript.utterance_id:
                continue
            if not words(transcript.text).startswith(item['prefix']):
                revoked.append(item['id'])
            elif now-item['source_ms'] <= self.cfg['incremental']['retention_ms']:
                if words(transcript.stable).startswith(item['prefix']):
                    # New ASR evidence can stabilize this same observed span.
                    # Preserve semantic acquisition time and replace the packet.
                    item['belief'].observe(transcript.reliability,item['observation'],item['source_ms'])
                kept.append(item)
        self.items = kept
        return revoked

    def add(self, snapshot, observation, transcript, start, prior, now, seq):
        end = start+len(snapshot['text'])
        prefix = words(transcript.text[:start]+snapshot['text'])
        current = words(transcript.text)
        if not prefix or not current.startswith(prefix):
            return None
        q = observation
        # A newer observation of the SAME scope replaces it even when uncertain.
        same = next((x for x in self.items if x['prefix']==prefix), None)
        identity = same['id'] if same else None
        self.items = [x for x in self.items if x['prefix']!=prefix]
        # A coherent larger interpretation supersedes its overlapping prefixes.
        # Conflicting appraisal also retires a prefix, even if U remains uncertain.
        best = max(q['appraisal'], key=q['appraisal'].get)
        def superseded(item):
            overlaps = prefix.startswith(item['prefix'])
            old_best = max(item['observation']['appraisal'], key=item['observation']['appraisal'].get)
            return overlaps and (q['interpretability']['resolved'] >= self.cfg['incremental']['retain_interpretability']
                or (best != old_best and q['appraisal'][best] >= .7))
        self.items = [x for x in self.items if not superseded(x)]
        if (q['interpretability']['resolved'] < self.cfg['incremental']['retain_interpretability']
                or snapshot['reliability'] < self.cfg['incremental']['retain_reliability']):
            return None
        self.sequence += 1
        belief = ListenerBelief(self.cfg)
        belief.anchor = tuple(prior); belief.anchor_ms = snapshot['source_ms']
        belief.observe(snapshot['reliability'], deepcopy(q), snapshot['source_ms'])
        item = {'id': identity or f'e{self.sequence}', 'utterance_id': transcript.utterance_id,
                'unit_id': snapshot['utterance_id'], 'start': start, 'end': end,
                'text': snapshot['text'], 'prefix': prefix, 'source_ms': snapshot['source_ms'],
                'semantic_seq': seq, 'observation': deepcopy(q), 'belief': belief}
        self.items.append(item)
        self.items = self.items[-self.cfg['incremental']['max_items']:]
        return item['id']

    def candidates(self, transcript, now):
        self.reconcile(transcript, now)
        if self.cfg.get('commit_scope')=='asr_utterance':
            # no_history removes cross-utterance priors only, not revision memory.
            return [] if self.cfg['condition']=='acoustic_only' else list(self.items)
        return [x for x in self.items if self.cfg['condition'] not in ('acoustic_only', 'no_history')]

    def describe(self, now):
        return [{'id':x['id'], 'unit_id':x['unit_id'], 'start':x['start'], 'end':x['end'],
                 'text':x['text'], 'age_ms':now-x['source_ms'],
                 'understanding':x['belief'].summary(now)['understanding']}
                for x in self.items]
