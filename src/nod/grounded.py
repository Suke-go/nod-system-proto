"""Auditable Bayes-risk control; v4 observation schema, revised decision model."""
from nod.responsive import configure as previous


def configure(cfg, condition='full'):
    cfg=previous(cfg,condition)
    cfg['profile']='listener_grounded_v4_2'
    listener=cfg['listener']
    listener['temporal_model']='single_reset_kernel'
    listener['commit_scope']='asr_utterance'
    listener.pop('evidence_half_life_ms',None)
    listener.pop('specific_minimum_reliability',None)
    common=('history_window_ms','max_actions_per_window','intensity_ranges',
            'expression_upgrade_gap_ms','continuer_max_age_ms')
    listener['policy']={key:listener['policy'][key] for key in common}
    listener['policy']['decision_model']={
        'family':'additive_bayes_risk_v1','status':'designed_unfitted',
        # State order: unperceived, unresolved, neutral, positive, negative, mixed.
        # These are explicit relative costs, not inferred psychological constants.
        'loss':{
            'none':[0.,.6,2.,2.,2.,2.],
            'continuer':[1.2,0.,.45,.65,.65,.65],
            'receipt':[2.,1.,0.,.35,.35,.35],
            'attentive':[2.,1.2,.6,.35,.35,0.],
            'warm':[4.,3.,2.,0.,5.,2.],
            'concerned':[4.,3.,2.,5.,0.,2.]},
        'motion':{'none':0.,'continuer':.05,'receipt':.08,'attentive':.1,'warm':.1,'concerned':.1},
        'acoustic_overlap':{'none':0.,'continuer':.65,'receipt':.05,'attentive':.05,'warm':.05,'concerned':.05},
        'incomplete_receipt_cost':.4,'substantive_demand_cost':2.,
        'repetition_cost':.1,'minimum_gain':0.,
    }
    cfg['research'].update(policy_revision='additive-risk-1',
        method='scoped_report_bayes_filter_concrete_action_risk',
        temporal_assumption='one_continuous_reset_kernel; commit_only_at_asr_utterance_boundary',
        parameters_status='observation_transition_and_loss_designed_unfitted')
    return cfg
