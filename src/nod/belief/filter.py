import math
from nod.config import distribution, probability


class Transition:
    def __init__(self, stationary, half_life_ms):
        self.stationary = distribution(stationary)
        self.half_life_ms = half_life_ms

    def predict(self, belief, dt_ms):
        if dt_ms < 0:
            raise ValueError("Negative transition interval")
        alpha = 2 ** (-dt_ms / self.half_life_ms)
        return tuple(alpha*b+(1-alpha)*p for b,p in zip(belief, self.stationary))

    def matrix(self, dt_ms):
        """T(dt)=a I+(1-a) 1 pi; a continuous-time Markov transition."""
        if dt_ms<0: raise ValueError('Negative transition interval')
        alpha=2**(-dt_ms/self.half_life_ms)
        return tuple(tuple(alpha*(i==j)+(1-alpha)*self.stationary[j] for j in range(4)) for i in range(4))


class IdentityCalibration:
    calibrated = False

    def calibrate(self, values):
        return distribution(values)


class BeliefFilter:
    def __init__(self, cfg, calibrator=None):
        self.transition = Transition(cfg["transition"]["stationary"], cfg["transition"]["half_life_ms"])
        self.calibrator = calibrator or IdentityCalibration()
        self.floor = cfg["calibration"]["probability_floor"]
        self.anchor = distribution(cfg["initial"])
        self.anchor_ms = 0
        self.posterior = self.anchor
        self.posterior_ms = 0
        self.observation_rule = cfg.get('observation_rule', 'likelihood_product')
        self.history_mode = cfg.get('history_mode', 'carry')

    def observation_prior(self, now_ms):
        if self.history_mode == 'none':
            return self.transition.stationary
        return self.transition.predict(self.anchor, now_ms-self.anchor_ms)

    def at(self, now_ms):
        return self.transition.predict(self.posterior, now_ms-self.posterior_ms)

    def open_epoch(self, now_ms):
        self.anchor = self.transition.stationary if self.history_mode == 'none' else self.at(now_ms)
        self.anchor_ms = now_ms
        self.posterior, self.posterior_ms = self.anchor, now_ms

    def observe(self, probabilities, q, now_ms):
        q = probability(q)
        prior = self.observation_prior(now_ms)
        likelihood = self.calibrator.calibrate(probabilities)
        if q == 0:
            self.posterior = prior
        elif self.observation_rule in ('posterior_blend', 'direct'):
            # Jev supplies a class distribution conditioned on the transcript,
            # not P(transcript | class). Blend by ASR reliability, without
            # multiplying another class prior into an already conditioned result.
            self.posterior = tuple((1-q)*a+q*p for a,p in zip(prior,likelihood))
        else:
            log_b = [math.log(max(b,self.floor))+q*math.log(max(p,self.floor)) for b,p in zip(prior,likelihood)]
            peak = max(log_b)
            weights = [math.exp(x-peak) for x in log_b]
            self.posterior = tuple(x/sum(weights) for x in weights)
        self.posterior_ms = now_ms
        return self.posterior

    def invalidate(self, now_ms):
        self.posterior = self.observation_prior(now_ms)
        self.posterior_ms = now_ms
