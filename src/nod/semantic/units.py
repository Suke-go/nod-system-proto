"""Causal, revisable evidence units anchored in stable ASR text."""
from dataclasses import replace
import re

# Only sentence punctuation in the stable prefix may create a new unit. Decimal
# points, initials, common abbreviations and ellipses are not sentence boundaries.
_ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'e.g', 'i.e'}


def boundaries(text):
    for match in re.finditer(r'[。！？!?]|\.(?=\s|$)', text):
        end = match.end()
        if match.group() == '.':
            before = text[:match.start()]
            token = re.search(r'([\w.]+)$', before)
            word = token.group(1).lower() if token else ''
            if before.endswith('.') or word in _ABBREVIATIONS or len(word) == 1:
                continue
        yield end


class UnitTracker:
    def __init__(self, max_context):
        self.max_context = max_context
        self.current = None
        self.start = 0
        self.context = []

    def update(self, snapshot, context):
        start = 0
        for end in boundaries(snapshot.stable):
            if snapshot.text[end:].strip():
                start = end
        while start < len(snapshot.text) and snapshot.text[start].isspace():
            start += 1
        text = snapshot.text[start:]
        stable = snapshot.stable[start:]
        uid = f'{snapshot.utterance_id}@{start}'
        previous = self.current
        new = previous is None or previous.utterance_id != uid
        repaired = bool(previous and not new and
                        (snapshot.repair_epoch != previous.repair_epoch or not text.startswith(previous.text)))
        self.current = replace(snapshot, utterance_id=uid, text=text, stable=stable)
        self.start = start
        items = list(context)
        if start:
            items.append({'speaker': 'user', 'text': snapshot.text[:start]})
        self.context = []
        remaining = self.max_context
        for item in reversed(items):
            if not remaining:
                break
            part = item['text'][-remaining:]
            self.context.insert(0, {'speaker': 'user', 'text': part})
            remaining -= len(part)
        return self.current, new, repaired
