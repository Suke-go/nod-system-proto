"""v4: probability-report observations and semantic upgrades after a continuer."""
from nod.incremental import configure as previous
from nod.belief.reports import FAMILY, REPORTS


def configure(cfg,condition='full'):
    cfg=previous(cfg,condition)
    cfg['profile']='listener_responsive_v4'
    cfg['semantic'].update(question_resource='jev-observation-v3.json',question_version='listener-observation-rubric-v3')
    c=cfg['listener'];c['version']=4
    c['observation_model']={'family':FAMILY,'reports':list(REPORTS),
        'concentration':{'perception':1.,'interpretability':1.,'appraisal':1.},
        'probability_floor':1e-6,'status':'designed_unfitted'}
    c['policy'].update(continuer_requires_progress=True,expression_upgrade_gap_ms=150,
        expression_can_refine=True,expression_minimum_sensor_mass=.65)
    c['policy']['intensity_ranges']['empathic']=[.3,.55]
    c['policy']['utility']['continuer']=[-1,1,.5,0,0,0]
    c['policy']['continuer_max_age_ms']=1200
    # Local receipt is a silent listener cue, not a claim to take the floor.
    # Keep the belief and completion requirements; do not gate ASR twice.
    c['policy']['local_receipt']={'readiness':.85,'minimum_resolved':.65,
        'suppress_filler_only':True}
    cfg['asr'].update(preserve_final_jobs=True,final_queue_capacity=2)
    cfg['motor']['default_durations_ms']['EMPATHIC_EXPRESSION']=1000
    cfg['research'].update(protocol_version='listener-v4',
        policy_revision='local-receipt-1',
        method='finite_bayes_filter_hierarchical_report_density_scoped_evidence',
        observation_assumption='conditional_independence_of_reports_given_state; uniform_undefined_axes',
        appraisal_scope='speaker_evaluation_of_experience_not_direct_measurement_of_empathy_desire')
    return cfg
