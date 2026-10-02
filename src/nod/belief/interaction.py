"""Generalized Bayesian filtering with replaceable semantic and acoustic factors.

Jev outputs are conditional scores, so this is a specified compatibility model,
not a claim that P(frame|text) is a generative likelihood P(text|frame).
"""
import math
from nod.config import probability
from nod.semantic.frames import FRAME_ORDER, FRAMES, frame_distribution, function_distribution, marginals


def temper(values, temperature):
    logs = [math.log(max(v, 1e-9))/temperature for v in frame_distribution(values)]
    peak = max(logs)
    weights = [math.exp(v-peak) for v in logs]
    return tuple(v/sum(weights) for v in weights)


class InteractionBelief:
    def __init__(self, cfg):
        self.cfg = cfg
        self.uniform = (1/len(FRAMES),)*len(FRAMES)
        self.anchor = self.uniform
        self.anchor_ms = 0
        self.raw = None
        self.reliability = 0.
        self.observed_ms = 0
        self.audio = None
        self.complete_boundary = False
        self.silence_boundary = False

    def frames_at(self, now):
        if self.cfg['condition']=='acoustic_only':
            return tuple(float(k=='explanation_open') for k in FRAME_ORDER)
        alpha = 2**(-(now-self.anchor_ms)/self.cfg['semantic_half_life_ms'])
        prior = tuple(alpha*x+(1-alpha)*u for x,u in zip(self.anchor,self.uniform))
        if self.cfg['condition'] in ('no_history', 'direct'):
            prior = self.uniform
        if self.raw is None:
            return prior
        scores = temper(self.raw, self.cfg['temperature'])
        if self.cfg['condition']=='argmax':
            best=max(range(len(scores)),key=scores.__getitem__)
            scores=tuple(float(i==best) for i in range(len(scores)))
        age_weight = 2**(-max(0,now-self.observed_ms)/self.cfg['evidence_half_life_ms'])
        weight = self.reliability * age_weight
        if self.cfg['condition'] == 'direct':
            return tuple(weight*p+(1-weight)*u for p,u in zip(scores,self.uniform))
        logs = [math.log(max(p,1e-9))+weight*math.log(max(q,1e-9)) for p,q in zip(prior,scores)]
        peak = max(logs)
        weights = [math.exp(v-peak) for v in logs]
        return tuple(v/sum(weights) for v in weights)

    def open_unit(self, now, revoke=False):
        if revoke:
            # Remove this unit's evidence before predicting. The previous unit
            # remains the anchor; an ASR repair must not carry its own old score.
            self.raw = None
        prior = self.frames_at(now)
        carry = self.cfg['unit_carry']
        self.anchor = tuple(carry*p+(1-carry)*u for p,u in zip(prior,self.uniform))
        self.anchor_ms = now
        self.raw = None
        self.reliability = 0.
        self.complete_boundary = self.silence_boundary = False

    def revoke(self):
        self.raw = None
        self.reliability = 0.
        self.complete_boundary = self.silence_boundary = False

    def observe(self, values, reliability, now):
        self.raw = frame_distribution(values)
        self.reliability = probability(reliability)
        self.observed_ms = now

    def acoustic(self, score, source, now):
        if self.audio is None or source > self.audio[1]:
            self.audio = (probability(score), source)

    def joint_at(self, now):
        """P(frame, response_ready), frame-major [wait, ready] order.

        Readiness resets to a uniform prior at each control evaluation. Its
        observation is the LATEST audio factor conditioned on semantic progress;
        a 10 Hz repeated score is never compounded as independent evidence.
        """
        values = self.frames_at(now)
        fresh = self.audio is not None and now-self.audio[1] <= self.cfg['audio_max_age_ms']
        score = self.audio[0] if fresh else 0.
        acoustic = self.cfg['readiness']
        joint = []
        for name, p in zip(FRAME_ORDER,values):
            frame = FRAMES[name]
            logit = (acoustic['intercept'] + acoustic['vap']*score
                     + acoustic['complete']*(frame.progress == 'complete')
                     + acoustic['stable_boundary']*self.complete_boundary
                     + acoustic['silence_boundary']*self.silence_boundary
                     - acoustic['repair_open']*(frame.act == 'repair' and frame.progress == 'open'))
            ready = 1/(1+math.exp(-logit)) if fresh else 0.
            joint.extend((p*(1-ready),p*ready))
        return tuple(joint)

    def at(self, now):
        # Compatibility view for old dashboards; decisions use the joint state.
        return function_distribution(self.frames_at(now))

    def summary(self, now):
        frames = self.frames_at(now)
        joint = self.joint_at(now)
        entropy = -sum(p*math.log(p) for p in frames if p)/math.log(len(FRAMES))
        return {'frames':dict(zip(FRAME_ORDER,frames)), 'marginals':marginals(frames),
                'ready':sum(joint[1::2]), 'normalized_entropy':entropy,
                'joint_order':'frame-major: wait, ready', 'joint':joint,
                'temperature':self.cfg['temperature'], 'calibration':self.cfg['calibration']}
