"""Versioned joint semantic observations, not commands or mental-state diagnoses.

One categorical distribution avoids multiplying correlated act/progress/valence
marginals. The reduced ontology is an operational choice; see production-design.md.
"""
from dataclasses import dataclass
from nod.config import probability

SCHEMA = 'listener-semantics-v1'


@dataclass(frozen=True)
class Frame:
    act: str
    progress: str
    stance: str
    function: str


FRAMES = {'unclear': Frame('unknown', 'unknown', 'unknown', 'none')}
for _act, _stance, _name in (
    ('inform', 'neutral', 'explanation'),
    ('assessment', 'neutral', 'evaluation'),
    ('experience', 'positive', 'positive_experience'),
    ('experience', 'negative', 'negative_experience'),
    ('experience', 'mixed', 'mixed_experience'),
    ('question', 'unknown', 'question'),
    ('repair', 'unknown', 'repair'),
):
    for _progress in ('open', 'complete'):
        _function = ('none' if _act == 'question' or (_act == 'repair' and _progress == 'open')
                     else 'continuer' if _progress == 'open'
                     else 'empathic' if _act == 'experience' else 'understanding')
        FRAMES[f'{_name}_{_progress}'] = Frame(_act, _progress, _stance, _function)
FRAME_ORDER = tuple(FRAMES)
FUNCTIONS = ('none', 'continuer', 'understanding', 'empathic')


def frame_distribution(values):
    if isinstance(values, dict):
        if set(values) != set(FRAMES):
            raise ValueError('Semantic frame keys do not match ' + SCHEMA)
        values = [values[name] for name in FRAME_ORDER]
    if not isinstance(values, (tuple, list)) or len(values) != len(FRAMES):
        raise ValueError('Expected the complete semantic frame distribution')
    result = tuple(probability(v) for v in values)
    total = sum(result)
    if abs(total - 1) > 1e-4:
        raise ValueError('Semantic probabilities must sum to one')
    return tuple(v / total for v in result)


def marginals(values):
    values = frame_distribution(values)
    result = {k: {} for k in ('act', 'progress', 'stance', 'function')}
    for name, p in zip(FRAME_ORDER, values):
        for key, out in result.items():
            label = getattr(FRAMES[name], key)
            out[label] = out.get(label, 0.) + p
    return result


def function_distribution(values):
    m = marginals(values)['function']
    return tuple(m.get(f, 0.) for f in FUNCTIONS)


def fixture_frame(name, certainty=.98):
    """Synthetic test helper; never used as a fallback for an unavailable model."""
    rest = (1-certainty)/(len(FRAMES)-1)
    return tuple(certainty if k == name else rest for k in FRAME_ORDER)
