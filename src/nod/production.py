"""Versioned, inspectable operational parameters for the complete listener model."""
from copy import deepcopy
import math
from nod.semantic.frames import SCHEMA, FUNCTIONS


def configure(cfg, condition='full'):
    cfg = deepcopy(cfg)
    cfg['profile'] = 'listener_production_v1' if condition=='full' else 'listener_'+condition+'_v1'
    cfg['semantic'].update(schema=SCHEMA,question_version=SCHEMA,question_resource='jev-semantics.json',
        total_deadline_ms=1800,source_max_age_ms=3500,min_dispatch_interval_ms=300)
    cfg['motor']['cooldown_after_completion_ms'] = 1800
    cfg['logging']['research_diagnostics'] = True
    cfg['interaction'] = {
        'enabled':True, 'version':1, 'condition':condition, 'task':'presentation_listening',
        'semantic_half_life_ms':2500, 'evidence_half_life_ms':4000, 'unit_carry':.25,
        'temperature':1., 'calibration':'identity_unfitted', 'audio_max_age_ms':350,
        'specific_minimum_reliability':.65,
        'readiness':{'intercept':-2.8,'vap':4.,'complete':1.3,'stable_boundary':.8,
                     'silence_boundary':1.8,'repair_open':2.},
        'policy':{
            'utility':{'none':[0.,-.15,-.3,-.3], 'continuer':[-1.,1.,.1,-.1],
                       'understanding':[-1.5,-.4,1.4,-.3], 'empathic':[-2.,-.8,-.5,1.6]},
            'premature_cost':{'continuer':.45,'understanding':1.2,'empathic':1.5},
            'motion_cost':{'continuer':.05,'understanding':.08,'empathic':.1},
            'intensity_ranges':{'continuer':[.15,.35],'understanding':[.35,.7],'empathic':[.3,.7]},
            'history_window_ms':15000,'max_actions_per_window':4,'repetition_cost':.1,
            'minimum_gain':.08,'expression_minimum_probability':.65},
    }
    cfg['research'] = {'method':'joint_semantic_readiness_generalized_bayes',
        'semantic_schema':SCHEMA,'parameters_status':'explicit_design_values_not_human_fitted',
        'calibration':'identity_unfitted','protocol_version':'listener-production-v1',
        'condition':condition,'psychological_scope':'operational_feedback_state_not_measured_mental_state'}
    return cfg


def validate_interaction(cfg):
    c = cfg['interaction']
    if c.get('enabled') is not True or c.get('version')!=1:
        raise ValueError('Unsupported interaction configuration')
    if cfg['semantic'].get('schema')!=SCHEMA or cfg['semantic'].get('question_resource')!='jev-semantics.json':
        raise ValueError('Interaction model requires its typed semantic schema')
    if c['condition'] not in ('full','no_history','direct','argmax','acoustic_only'):
        raise ValueError('Unsupported interaction condition')
    def finite(value,low=None,high=None):
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError('Interaction parameters must be finite numbers')
        if low is not None and value<low or high is not None and value>high:
            raise ValueError('Interaction parameter out of range')
    for key in ('semantic_half_life_ms','evidence_half_life_ms','temperature','audio_max_age_ms'):
        finite(c[key],.001)
    for key in ('unit_carry','specific_minimum_reliability'):
        finite(c[key],0,1)
    if not isinstance(c['calibration'],str): raise ValueError('Calibration metadata required')
    for v in c['readiness'].values(): finite(v,-20,20)
    p = c['policy']
    if set(p['utility'])!=set(FUNCTIONS): raise ValueError('Invalid interaction utility')
    for row in p['utility'].values():
        if len(row)!=4: raise ValueError('Utility must cover four feedback functions')
        for v in row: finite(v)
    for group in ('premature_cost','motion_cost'):
        if set(p[group])!=set(FUNCTIONS[1:]): raise ValueError('Invalid action costs')
        for v in p[group].values(): finite(v,0)
    for f in FUNCTIONS[1:]:
        low,high=p['intensity_ranges'][f];finite(low,0,1);finite(high,low,1)
    finite(p['history_window_ms'],1);finite(p['max_actions_per_window'],1)
    if not isinstance(p['max_actions_per_window'],int): raise ValueError('Action budget must be integer')
    finite(p['repetition_cost'],0);finite(p['minimum_gain'],0)
    finite(p['expression_minimum_probability'],0,1)
