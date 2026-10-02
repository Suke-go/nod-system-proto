"""Function-specific realization costs over the same finite listener belief."""
import math
import re
from nod.policy.listener import ListenerPolicy, FUNCTIONS


def filler_only(text):
    """Conservative realization guard, not a semantic/psychological classifier.

    Do not include acknowledgements such as はい/yes or short feeling reports.
    Full-span matching prevents removing fillers from substantive sentences.
    """
    cleaned=re.sub(r'[\W_]+','',text.casefold())
    return bool(cleaned and re.fullmatch(
        r'(?:(?:あの[うー]?|え[えーっ]?と|え[えー]|う[うー]?ん?|その[うー]?|で|と)|'
        r'(?:u+m+|u+h+|e+r+m*|h+m+|well|so))+',cleaned))


class IncrementalPolicy(ListenerPolicy):
    def choose(self, belief, history, now, unit_id, obs, vap, silence, allow_specific,
               *, retained=False, evidence_id=None, expressive=True, allow_expression=True,
               evidence_text=None, expression_only=False, evidence_age_ms=0,
               receipt_upgrade=False):
        c = self.cfg
        completion = obs['completion'] if obs else 0.
        demand = obs['response_demand'] if obs else 0.
        t = c['timing']
        ready = 1/(1+math.exp(-(t['intercept']+t['vap']*vap+t['completion']*completion+t['silence']*silence)))
        if silence and obs: ready = max(ready, t['confirmed_silence_weight'])
        # A short silent facial expression does not claim receipt of a whole turn.
        # This is a designed utility weight, not an estimated timing probability.
        face_ready = c['expression_readiness']
        local=c.get('local_receipt')
        receipt_supported=bool(local and not retained and obs
            and obs['interpretability']['resolved']>=local['minimum_resolved']
            and completion>=c['minimum_completion']
            and sum(belief[2:])>=c['minimum_understanding']
            and demand<=c['maximum_response_demand'])
        receipt_ready=max(ready,local['readiness']) if receipt_supported else ready
        weights = {'continuer':ready, 'understanding':receipt_ready, 'empathic':face_ready}
        recent = [h for h in history if now-h['at_ms'] < c['history_window_ms']
                  and h['status'] not in ('rejected','failed','cancelled') and not h.get('invalidated')]
        utility = {'none':sum(b*u for b,u in zip(belief,c['utility']['none']))}
        for f in FUNCTIONS[1:]:
            w = weights[f]
            utility[f] = w*sum(b*u for b,u in zip(belief,c['utility'][f]))-(1-w)*c['premature_cost'][f]-c['motion_cost'][f]-c['repetition_cost']*sum(h['function']==f for h in recent)
        allowed = ['none'] if retained else ['none','continuer']
        if c.get('continuer_requires_progress') and evidence_text is not None:
            from nod.semantic.evidence import words
            current=words(evidence_text)
            if any(h['function']=='continuer' and h['unit_id']==unit_id and
                   words(h.get('evidence_text',''))==current for h in history
                   if h['status'] not in ('rejected','failed','cancelled') and not h.get('invalidated')):
                allowed=['none']
            if (evidence_age_ms>c['continuer_max_age_ms'] or
                any(h['function'] in ('understanding','empathic') and h['unit_id']==unit_id
                    for h in history if h['status'] not in ('rejected','failed','cancelled') and not h.get('invalidated'))):
                allowed=['none']
        if (allow_specific or receipt_supported) and obs:
            if not retained and completion>=c['minimum_completion'] and sum(belief[2:])>=c['minimum_understanding'] and demand<=c['maximum_response_demand']:
                allowed.append('understanding')
        if obs and allow_expression:
            sensor_support=(not c.get('expression_can_refine') or
                1-obs['appraisal']['neutral']>=c['expression_minimum_sensor_mass'])
            if expressive and sensor_support and 1-belief[0]>=c['expression_minimum_perception'] and sum(belief[3:])>=c['expression_minimum_mass']:
                allowed.append('empathic')
        if expression_only:
            upgraded=('none','empathic','understanding') if receipt_upgrade else ('none','empathic')
            allowed=[f for f in allowed if f in upgraded]
        # Questions suppress receipt/agreement, but not a nonverbal listening cue.
        used = [h for h in history if h['status'] not in ('rejected','failed','cancelled') and not h.get('invalidated')]
        for f in ('understanding','empathic'):
            matching=[h for h in used if h['function']==f and (h['unit_id']==unit_id or (evidence_id and h.get('evidence_id')==evidence_id))]
            if f=='empathic' and c.get('expression_can_refine') and obs:
                summary={'appraisal':dict(zip(('neutral','positive','negative','mixed'),belief[2:])),
                         'understanding':sum(belief[2:])}
                _,_,direction=self.embody('empathic',summary,0,True)
                key={'warm':'positive','concerned':'negative'}.get(direction)
                if key and obs['appraisal'][key]>=c['expression_minimum_probability']:
                    matching=[h for h in matching if h.get('expression')==direction]
            if matching:
                if f in allowed: allowed.remove(f)
        reason = 'utility_prefers_none'
        if local and local['suppress_filler_only'] and evidence_text is not None and filler_only(evidence_text):
            allowed=['none'];reason='filler_only_wait'
        if len(recent)>=c['max_actions_per_window']:
            allowed=['none'];reason='action_budget'
        winner=max(allowed,key=utility.get)
        if utility[winner]-utility['none']<c['minimum_gain']:winner='none'
        context={'readiness':ready,'expression_readiness':face_ready,
            'completion':completion,'response_demand':demand,'allowed':allowed,'reason':reason,
            'evidence_id':evidence_id,'retained':retained}
        if local:context.update(receipt_readiness=receipt_ready,local_receipt_supported=receipt_supported)
        return winner,utility,context

    def embody(self,function,summary,prominence,expressive):
        if function!='empathic': return super().embody(function,summary,prominence,expressive)
        a=summary['appraisal'];mass=sum(a[k] for k in ('positive','negative','mixed'))
        expression='attentive'
        for key,name in [('positive','warm'),('negative','concerned')]:
            if a[key]>=self.cfg['expression_minimum_mass'] and a[key]/max(1e-12,summary['understanding'])>=self.cfg['expression_minimum_probability']:
                expression=name
        lo,hi=self.cfg['intensity_ranges']['empathic']
        return 'EMPATHIC_EXPRESSION',lo+(hi-lo)*mass,expression
