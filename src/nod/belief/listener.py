"""Finite Bayes filter with a proper joint observation-density model.

The latest observation packet replaces its predecessor within a revisable unit.
Only a new unit commits a posterior through the transition matrix. Controller
ticks and repeated ASR prefixes never masquerade as independent measurements.
"""
import math
from nod.semantic.sensor import features

STATES = ('unperceived', 'perceived_unresolved', 'interpreted_neutral',
          'interpreted_positive', 'interpreted_negative', 'interpreted_mixed')


def normalized(values):
    total = sum(values)
    if total <= 0 or not math.isfinite(total): raise ValueError('Invalid probability mass')
    return tuple(v/total for v in values)


def cholesky(matrix):
    n = len(matrix); L = [[0.]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            v = matrix[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                if v <= 0 or not math.isfinite(v): raise ValueError('Covariance must be positive definite')
                L[i][j] = math.sqrt(v)
            else: L[i][j] = v/L[j][j]
    return L


def gaussian_logs(vector, model):
    """Marginalize missing coordinates; do not impute observations or fake labels."""
    indices = [i for i, x in enumerate(vector) if x is not None]
    if not indices: return (0.,)*len(STATES)
    L = cholesky([[model['covariance'][i][j] for j in indices] for i in indices])
    base = -.5*len(indices)*math.log(2*math.pi)-sum(math.log(L[i][i]) for i in range(len(indices)))
    def density(mean):
        y = []
        for i, k in enumerate(indices):
            y.append((vector[k]-mean[k]-sum(L[i][j]*y[j] for j in range(i)))/L[i][i])
        return base-.5*sum(v*v for v in y)
    result = []
    for state, mean in enumerate(model['means']):
        if state < 2:
            # Without established understanding, appraisal is not a robot state.
            # Marginalize that nuisance variable rather than treating a neutral
            # classifier output as positive evidence for understanding.
            components=[density(mean[:2]+m[2:]) for m in model['means'][2:]]
            peak=max(components)
            result.append(peak+math.log(sum(math.exp(v-peak) for v in components)/4))
        else:
            result.append(density(mean))
    return tuple(result)


class ListenerBelief:
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchor = tuple(cfg['initial']); self.anchor_ms = 0
        self.packet = None; self.audio = None

    def predicted(self, now):
        if self.cfg['condition'] == 'no_history': return tuple(self.cfg['initial'])
        # Continuous-time transition: exp(-lambda dt) I + (1-exp(-lambda dt)) 1 pi.
        keep = 2**(-max(0, now-self.anchor_ms)/self.cfg['state_half_life_ms'])
        return tuple(keep*b+(1-keep)*p for b, p in zip(self.anchor, self.cfg['initial']))

    def vector(self):
        if self.packet is None: return [None]*5
        perception, obs, _ = self.packet
        if self.cfg['condition'] == 'acoustic_only': obs = None
        if self.cfg['condition'] == 'argmax' and obs is not None:
            from copy import deepcopy
            obs = deepcopy(obs)
            for name in ('interpretability', 'appraisal'):
                best = max(obs[name], key=obs[name].get)
                obs[name] = {k: float(k == best) for k in obs[name]}
            for name in ('completion', 'response_demand'): obs[name] = float(obs[name] >= .5)
        return features(perception, obs)

    def posterior(self, now):
        prior = self.predicted(now)
        if self.packet is None: return prior
        evidence_ms = max(self.anchor_ms, self.packet[2])
        prior = self.predicted(evidence_ms)
        likelihood = self.log_likelihood()
        logs = [math.log(max(p, 1e-15))+l for p, l in zip(prior, likelihood)]
        peak = max(logs)
        observed = normalized([math.exp(x-peak) for x in logs])
        # Propagate from evidence time, using an explicit stochastic reset kernel.
        half_life=(self.cfg['state_half_life_ms'] if self.cfg.get('temporal_model')=='single_reset_kernel'
                   else self.cfg['evidence_half_life_ms'])
        age = 2**(-max(0, now-evidence_ms)/half_life)
        return tuple(age*b+(1-age)*p for b, p in zip(observed, self.cfg['initial']))

    def probability_report(self):
        from nod.belief.reports import report
        if self.packet is None:return [None]*6
        r,obs,_=self.packet
        if self.cfg['condition']=='acoustic_only':obs=None
        return report(r,obs,self.cfg['condition']=='argmax')

    def log_likelihood(self):
        from nod.belief.reports import FAMILY,report_logs
        model=self.cfg['observation_model']
        if model['family']==FAMILY:return report_logs(self.probability_report(),model)
        return gaussian_logs(self.vector(),model)

    def open_unit(self, now, revoke=False):
        if revoke: self.packet = None
        b = self.posterior(now)
        T = self.cfg['transition']
        self.anchor = tuple(sum(b[i]*T[i][j] for i in range(len(STATES))) for j in range(len(STATES)))
        self.anchor_ms = now; self.packet = None

    def revoke(self): self.packet = None

    def observe(self, perception, obs, source_ms):
        self.packet = (perception, obs, source_ms)

    def acoustic(self, score, source, now):
        if self.audio is None or source > self.audio[1]: self.audio = (score, source)

    def at(self, now):
        b = self.posterior(now)
        return (b[0], b[1], b[2], sum(b[3:]))

    def summary(self, now):
        b = self.posterior(now)
        result = {'state_order': STATES, 'posterior': b, 'predicted': self.predicted(now),
                'perception': 1-b[0], 'understanding': sum(b[2:]),
                'appraisal': dict(zip(('neutral', 'positive', 'negative', 'mixed'), b[2:])),
                'features': self.vector(),
                'log_likelihood': self.log_likelihood(),
                'normalized_entropy': -sum(p*math.log(p) for p in b if p)/math.log(len(STATES)),
                'observation_model_status': self.cfg['observation_model']['status']}
        if self.cfg.get('version')==4:
            result.update(probability_report=self.probability_report(),observation_family=self.cfg['observation_model']['family'])
        return result
