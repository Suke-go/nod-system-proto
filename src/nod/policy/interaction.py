"""Bayes decision rule over the joint state, with task-specific explicit losses."""
from nod.semantic.frames import FRAMES, FRAME_ORDER, FUNCTIONS


class InteractionPolicy:
    def __init__(self, cfg):
        self.cfg = cfg

    def choose(self, joint, history, now, unit_id, allow_specific):
        utilities = {f:0. for f in FUNCTIONS}
        for i,name in enumerate(FRAME_ORDER):
            target = FRAMES[name].function
            for ready in (0,1):
                p = joint[2*i+ready]
                for function in FUNCTIONS:
                    value = self.cfg['utility'][function][FUNCTIONS.index(target)]
                    if function != 'none' and not ready:
                        value = -self.cfg['premature_cost'][function]
                    utilities[function] += p*value
        recent = [h for h in history if now-h['at_ms'] < self.cfg['history_window_ms']
                  and h['status'] not in ('rejected','failed','cancelled')]
        for function in FUNCTIONS[1:]:
            utilities[function] -= self.cfg['motion_cost'][function]
            utilities[function] -= self.cfg['repetition_cost']*sum(h['function']==function for h in recent)
        acknowledged = any(h['unit_id']==unit_id and h['function'] in ('understanding','empathic')
                           and h['status'] not in ('rejected','failed','cancelled') for h in history)
        allowed = list(FUNCTIONS) if allow_specific and not acknowledged else ['none','continuer']
        if acknowledged or len(recent) >= self.cfg['max_actions_per_window']:
            allowed = ['none']
        winner = max(allowed,key=lambda f:utilities[f])
        if utilities[winner]-utilities['none'] < self.cfg['minimum_gain']:
            winner = 'none'
        return winner, utilities

    def embody(self, function, summary, prominence, expressive):
        low,high = self.cfg['intensity_ranges'][function]
        confidence = summary['marginals']['function'].get(function,0.)
        strength = .7*confidence+.3*prominence
        intensity = low+(high-low)*strength
        stance = summary['marginals']['stance']
        expression = 'neutral'
        if function == 'empathic':
            if stance.get('positive',0.) >= self.cfg['expression_minimum_probability']:
                expression = 'warm'
            elif stance.get('negative',0.) >= self.cfg['expression_minimum_probability']:
                expression = 'concerned'
            else:
                expression = 'attentive'
        action = ('SMALL_NOD' if function=='continuer' else 'STRONG_NOD'
                  if function=='understanding' or not expressive else 'EMPATHIC_EXPRESSION')
        return action,intensity,expression
