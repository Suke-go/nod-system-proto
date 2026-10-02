"""Choose the realized cue by its expected loss, without probability floors.

The observation distribution is integrated once. Completion and response demand
enter as additive contextual costs; no joint independence of Jev heads is assumed.
All costs are elicitable preferences, currently declared designed/unfitted.
"""
from nod.policy.listener import ListenerPolicy,FUNCTIONS
from nod.semantic.evidence import words

ACTIONS=('none','continuer','receipt','attentive','warm','concerned')
FUNCTION={'none':'none','continuer':'continuer','receipt':'understanding',
          'attentive':'empathic','warm':'empathic','concerned':'empathic'}


class BayesRiskPolicy(ListenerPolicy):
    def choose(self,belief,history,now,unit_id,obs,vap,silence,allow_specific,
               *,retained=False,evidence_id=None,expressive=True,allow_expression=True,
               evidence_text=None,expression_only=False,evidence_age_ms=0,
               receipt_upgrade=False,supported_expressions=None):
        c=self.cfg;m=c['decision_model']
        used=[h for h in history if h['status'] not in ('rejected','failed','cancelled') and not h.get('invalidated')]
        # Revoking the interpretation cannot undo a movement already dispatched.
        recent=[h for h in history if h['status']!='rejected' and now-h['at_ms']<c['history_window_ms']]
        same=[h for h in used if h['unit_id']==unit_id or (evidence_id and h.get('evidence_id')==evidence_id)]
        completion=obs['completion'] if obs else 0.
        demand=obs['response_demand'] if obs else 0.
        overlap=0. if silence else 1-vap
        risks={};terms={}
        for action in ACTIONS:
            loss=list(m['loss'][action])
            if action=='none':
                # Missing a *neutral content receipt* matters as the local content
                # becomes complete. Waiting during an unfinished thought is not
                # charged as though a receipt opportunity had already occurred.
                loss[2]*=completion
            terms[action]={
                'state':sum(b*l for b,l in zip(belief,loss)),
                'motion':m['motion'][action],
                'acoustic_overlap':m['acoustic_overlap'][action]*overlap,
                'incomplete':m['incomplete_receipt_cost']*(1-completion) if action=='receipt' else 0.,
                'response_demand':m['substantive_demand_cost']*demand if action=='receipt' else 0.,
                'repetition':m['repetition_cost']*sum(h['function']==FUNCTION[action] for h in recent) if action!='none' else 0.}
            risks[action]=sum(terms[action].values())
        # Feasibility concerns availability and execution, never a second semantic threshold.
        allowed=['none'];excluded={}
        for action in ACTIONS[1:]:
            why=[]
            if action=='continuer':
                if retained:why.append('retained_scope')
                if evidence_age_ms>c['continuer_max_age_ms']:why.append('evidence_stale')
                if any(h['function'] in ('understanding','empathic') for h in same):why.append('specific_feedback_done')
                if evidence_text is not None and any(h['function']=='continuer' and
                    words(h.get('evidence_text',''))==words(evidence_text) for h in same):why.append('no_content_progress')
            elif action=='receipt':
                if obs is None:why.append('semantics_unavailable')
                if retained:why.append('retained_scope')
                if any(h['function']=='understanding' for h in same):why.append('receipt_done')
            else:
                if obs is None:why.append('semantics_unavailable')
                if not expressive or (supported_expressions is not None and action not in supported_expressions):
                    why.append('expression_unavailable')
                if any(h['function']=='empathic' and h.get('expression')==action for h in same):why.append('expression_done')
            if expression_only and not (action in ('attentive','warm','concerned') or (receipt_upgrade and action=='receipt')):
                why.append('motor_cooldown')
            if len(recent)>=c['max_actions_per_window']:why.append('action_budget')
            if why:excluded[action]=why
            else:allowed.append(action)
        best=min(allowed,key=risks.get)
        gain=risks['none']-risks[best]
        if gain<m['minimum_gain']:best='none'
        # Public function utilities remain compatible with the existing UI/logs.
        # Per-realization risks in context show exactly what was selected and sent.
        utilities={f:max(-risks[a] for a in ACTIONS if FUNCTION[a]==f) for f in FUNCTIONS}
        function=FUNCTION[best]
        utilities[function]=-risks[best]
        reason='action_selected' if best!='none' else 'action_budget' if len(recent)>=c['max_actions_per_window'] else 'risk_prefers_wait'
        context={'decision_model':m['family'],'parameter_status':m['status'],
            'selected_realization':best,'risks':risks,'risk_terms':terms,
            'allowed_realizations':allowed,'excluded_realizations':excluded,
            'allowed':list(dict.fromkeys(FUNCTION[a] for a in allowed)),
            'completion':completion,'response_demand':demand,'acoustic_overlap':overlap,
            'reason':reason,'evidence_id':evidence_id,'retained':retained}
        return function,utilities,context

    def realize(self,function,summary,prominence,expressive,context):
        chosen=context['selected_realization']
        if FUNCTION[chosen]!=function or chosen=='none':raise ValueError('Decision/realization mismatch')
        if function!='empathic':return super().embody(function,summary,prominence,expressive)
        a=summary['appraisal']
        confidence=a['positive'] if chosen=='warm' else a['negative'] if chosen=='concerned' else sum(a.values())
        lo,hi=self.cfg['intensity_ranges']['empathic']
        return 'EMPATHIC_EXPRESSION',lo+(hi-lo)*confidence,chosen


def validate_model(model):
    import math
    if model.get('family')!='additive_bayes_risk_v1' or model.get('status') not in ('designed_unfitted','fitted_on_calibration'):
        raise ValueError('Unsupported decision model')
    def finite(value):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Costs must be finite and nonnegative')
    for group in ('loss','motion','acoustic_overlap'):
        if set(model[group])!=set(ACTIONS):raise ValueError('Invalid concrete action costs')
        for value in model[group].values():
            if group=='loss':
                if len(value)!=6:raise ValueError('Invalid state dimension')
                for item in value:finite(item)
            else:finite(value)
    for key in ('incomplete_receipt_cost','substantive_demand_cost','repetition_cost','minimum_gain'):finite(model[key])
    if model['motion']['none'] or model['acoustic_overlap']['none']:raise ValueError('Waiting must not incur movement/overlap cost')
