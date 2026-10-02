"""Provider-independent observations. These are sensor scores, not robot states."""
import hashlib
import math
from importlib.resources import files
from nod.config import probability

SCHEMA = 'listener-observation-v2'
CHOICES = {'interpretability': ('unresolved', 'resolved'),
           'appraisal': ('neutral', 'positive', 'negative', 'mixed')}
SCALARS = ('completion', 'response_demand')


def distribution(values, order):
    if not isinstance(values, dict) or set(values) != set(order):
        raise ValueError('Observation categories do not match the sensor schema')
    p = [probability(values[k]) for k in order]
    total = sum(p)
    if abs(total-1) > 1e-4:
        raise ValueError('Observation probabilities must sum to one')
    return dict(zip(order, (v/total for v in p)))


def observation(values):
    if not isinstance(values, dict) or set(values) != set(CHOICES) | set(SCALARS):
        raise ValueError('Incomplete semantic observation')
    return {**{k: distribution(values[k], order) for k, order in CHOICES.items()},
            **{k: probability(values[k]) for k in SCALARS}}


def parse(body):
    answers = body['answers']; out = {}; confidence = {}
    if not isinstance(body['model'], str): raise ValueError('Missing model')
    for name, order in CHOICES.items():
        a = answers[name]; p = distribution(a['probabilities'], order)
        if a['type'] != 'choice' or a['choice'] not in p or p[a['choice']] < max(p.values())-1e-6:
            raise ValueError('Invalid observation choice')
        out[name] = p; confidence[name] = probability(a['confidence'])
    for name in SCALARS:
        a = answers[name]
        if a['type'] != 'noul': raise ValueError('Invalid observation type')
        out[name] = probability(a['noul'])
    return {'schema': SCHEMA, 'observation': observation(out), 'model': body['model'],
            'confidence': confidence, 'probabilities': compatibility(out)}


def compatibility(obs):
    """Legacy display/report only. Never used as a likelihood or controller state."""
    q = observation(obs); understood = q['interpretability']['resolved']
    c = q['completion']; demand = q['response_demand']
    receipt = understood*c*(1-demand)*q['appraisal']['neutral']
    empathy = understood*c*(1-demand)*(1-q['appraisal']['neutral'])
    continuer = understood*(1-c)*(1-demand)
    return (1-receipt-empathy-continuer, continuer, receipt, empathy)


def fixture(appraisal='neutral', resolved=.95, completion=.95, demand=.02):
    return {'interpretability': {'unresolved': 1-resolved, 'resolved': resolved},
            'appraisal': {k: .94 if k == appraisal else .02 for k in CHOICES['appraisal']},
            'completion': completion, 'response_demand': demand}


def prompt_digest(resource='jev-observation.json'):
    if resource not in ('jev-observation.json','jev-observation-v3.json'):raise ValueError('Unknown observation rubric')
    return hashlib.sha256(files('nod').joinpath('resources/'+resource).read_bytes()).hexdigest()


def scale_report(values, exponent):
    """Positive temperature/power map with the same floor used by report densities."""
    logs = [exponent * math.log(max(1e-6, value)) for value in values]
    weights = [math.exp(value-max(logs)) for value in logs]
    return [value/sum(weights) for value in weights]


def calibrate_scalars(obs, calibration):
    result = observation(obs)
    for head, temperature in calibration['temperature'].items():
        result[head] = scale_report([1-obs[head], obs[head]], 1/temperature)[1]
    return result


FEATURES = ('perception_logit', 'interpretability_logit', 'positive_vs_neutral',
            'negative_vs_neutral', 'mixed_vs_neutral')


def features(perception, obs=None):
    def logit(p):
        p = min(.999, max(.001, probability(p)))
        return math.log(p/(1-p))
    result = [logit(perception)]
    if obs is None: return result + [None]*4
    q = observation(obs); a = q['appraisal']
    return result + [logit(q['interpretability']['resolved'])] + [
        math.log(max(.001, a[k])/max(.001, a['neutral'])) for k in CHOICES['appraisal'][1:]]
