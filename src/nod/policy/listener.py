"""One-step Bayes decision rule over the robot's listener state."""
import math
from nod.belief.listener import STATES

FUNCTIONS = ('none', 'continuer', 'understanding', 'empathic')


class ListenerPolicy:
    def __init__(self, cfg): self.cfg = cfg

    def choose(self, belief, history, now, unit_id, obs, vap, silence, allow_specific):
        c = self.cfg
        completion = obs['completion'] if obs else 0.
        demand = obs['response_demand'] if obs else 0.
        t = c['timing']
        ready = 1/(1+math.exp(-(t['intercept']+t['vap']*vap+t['completion']*completion+t['silence']*silence)))
        if silence and obs is not None:
            ready = max(ready, t['confirmed_silence_weight'])
        # Timing is a contextual utility modifier, not another independent
        # likelihood multiplied into the semantic posterior.
        utility = {'none': sum(b*u for b, u in zip(belief, c['utility']['none']))}
        recent = [h for h in history if now-h['at_ms'] < c['history_window_ms']
                  and h['status'] not in ('rejected', 'failed', 'cancelled')]
        for f in FUNCTIONS[1:]:
            utility[f] = (ready*sum(b*u for b, u in zip(belief, c['utility'][f]))
                          -(1-ready)*c['premature_cost'][f]-c['motion_cost'][f]
                          -c['repetition_cost']*sum(h['function'] == f for h in recent))
        acknowledged = any(h['unit_id'] == unit_id and h['function'] in ('understanding', 'empathic')
                           and h['status'] not in ('rejected', 'failed', 'cancelled') for h in history)
        allowed = ['none', 'continuer']; reason = 'utility_prefers_none'
        if allow_specific and obs and completion >= c['minimum_completion'] and sum(belief[2:]) >= c['minimum_understanding']:
            allowed = list(FUNCTIONS)
        if demand > c['maximum_response_demand']:
            allowed = ['none']; reason = 'substantive_response_required'
        if acknowledged:
            allowed = ['none']; reason = 'unit_already_acknowledged'
        if len(recent) >= c['max_actions_per_window']:
            allowed = ['none']; reason = 'action_budget'
        winner = max(allowed, key=utility.get)
        if utility[winner]-utility['none'] < c['minimum_gain']: winner = 'none'
        return winner, utility, {'readiness': ready, 'completion': completion, 'response_demand': demand,
                                 'allowed': allowed, 'reason': reason}

    def embody(self, function, summary, prominence, expressive):
        b = summary['posterior']; appraisal = summary['appraisal']
        confidence = (summary['perception'] if function == 'continuer' else b[2]
                      if function == 'understanding' else sum(b[3:]))
        lo, hi = self.cfg['intensity_ranges'][function]
        intensity = lo+(hi-lo)*(.7*confidence+.3*prominence)
        expression = 'neutral'
        if function == 'empathic':
            expression = 'attentive'
            if appraisal['positive'] >= self.cfg['expression_minimum_probability']: expression = 'warm'
            elif appraisal['negative'] >= self.cfg['expression_minimum_probability']: expression = 'concerned'
        action = ('SMALL_NOD' if function == 'continuer' else 'STRONG_NOD'
                  if function == 'understanding' or not expressive else 'EMPATHIC_EXPRESSION')
        return action, intensity, expression
