"""Minimal P/U/E listener model: theory structure + explicit operational assumptions."""
from copy import deepcopy
import math
from nod.semantic.sensor import SCHEMA, FEATURES
from nod.belief.listener import STATES, cholesky

CONDITIONS = ('full', 'no_history', 'argmax', 'acoustic_only')


def configure(cfg, condition='full'):
    cfg = deepcopy(cfg); cfg.pop('interaction', None)
    cfg['profile'] = 'listener_bayes_v2'
    cfg['semantic'].update(schema=SCHEMA, question_version=SCHEMA, question_resource='jev-observation.json',
        total_deadline_ms=1800, source_max_age_ms=3500, min_dispatch_interval_ms=300)
    cfg['motor']['cooldown_after_completion_ms'] = 1800
    cfg['logging'].update(research_diagnostics=True, decision_diagnostics=True)
    initial = [.4, .35, .1, .05, .05, .05]
    means = [[-1.73, 0, 0, 0, 0], [1.73, -1.73, 0, 0, 0],
             [1.73, 1.73, -2.83, -2.83, -2.83], [1.73, 1.73, 2.83, 0, 0],
             [1.73, 1.73, 0, 2.83, 0], [1.73, 1.73, 0, 0, 2.83]]
    covariance = [[4, .8, 0, 0, 0], [.8, 2, .15, .15, .15],
                  [0, .15, 4, 1, 1], [0, .15, 1, 4, 1], [0, .15, 1, 1, 4]]
    cfg['listener'] = {'enabled': True, 'version': 2, 'condition': condition,
        'state_order': list(STATES), 'initial': initial,
        'transition': [[.25*(i == j)+.75*initial[j] for j in range(6)] for i in range(6)],
        'transition_action_effect': 'identity_assumption_no_grounding_from_own_nod',
        'state_half_life_ms': 5000, 'evidence_half_life_ms': 4000, 'audio_max_age_ms': 350,
        'specific_minimum_reliability': .65,
        'observation_model': {'family': 'joint_logistic_normal_appraisal_marginalized',
            'features': list(FEATURES), 'means': means, 'covariance': covariance,
            'status': 'designed_unfitted'},
        'policy': {'minimum_gain': .08, 'minimum_completion': .65, 'maximum_response_demand': .65,
            'minimum_understanding': .65, 'expression_minimum_probability': .65,
            'history_window_ms': 15000, 'max_actions_per_window': 4, 'repetition_cost': .1,
            'timing': {'intercept': -2.8, 'vap': 4., 'completion': 1.3, 'silence': 1.8, 'confirmed_silence_weight': .9},
            'utility': {'none': [0, -.05, -.2, -.2, -.2, -.2],
                        'continuer': [-1, 1, .5, .4, .4, .4],
                        'understanding': [-2, -1, 1.5, .1, .1, .1],
                        'empathic': [-2, -1.5, -.8, 1.8, 1.8, 1.8]},
            'premature_cost': {'continuer': .45, 'understanding': 1.2, 'empathic': 1.5},
            'motion_cost': {'continuer': .05, 'understanding': .08, 'empathic': .1},
            'intensity_ranges': {'continuer': [.15, .35], 'understanding': [.35, .7], 'empathic': [.3, .7]}}}
    cfg['research'] = {'method': 'finite_bayes_filter_joint_observation_density', 'condition': condition,
        'protocol_version': 'listener-v2', 'parameters_status': 'designed_unfitted',
        'state_scope': 'robot_operational_listener_state_not_human_mental_truth',
        'structural_basis': ['Kopp et al. 2007', 'Meguro et al. 2010'],
        'semantic_schema': SCHEMA}
    return cfg


def validate(cfg):
    c = cfg['listener']
    if c.get('enabled') is not True or c.get('version') not in (2,3,4) or c['condition'] not in CONDITIONS:
        raise ValueError('Unsupported listener configuration')
    resource='jev-observation-v3.json' if c['version']==4 else 'jev-observation.json'
    if cfg['semantic'].get('schema') != SCHEMA or cfg['semantic'].get('question_resource') != resource:
        raise ValueError('Listener requires its observation schema')
    if c['state_order'] != list(STATES): raise ValueError('State order mismatch')
    def finite(v, low=None, high=None):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v): raise ValueError('Nonfinite parameter')
        if low is not None and v < low or high is not None and v > high: raise ValueError('Parameter outside bounds')
    def stochastic(row):
        if len(row) != 6: raise ValueError('State dimension mismatch')
        for v in row: finite(v, 0, 1)
        if abs(sum(row)-1) > 1e-8: raise ValueError('Transition/prior must sum to one')
    stochastic(c['initial'])
    if len(c['transition']) != 6: raise ValueError('Transition dimension mismatch')
    for row in c['transition']: stochastic(row)
    for k in ('state_half_life_ms', 'audio_max_age_ms'): finite(c[k], 1)
    if c.get('temporal_model')!='single_reset_kernel':finite(c['evidence_half_life_ms'],1)
    if not c['policy'].get('decision_model'):finite(c['specific_minimum_reliability'], 0, 1)
    m = c['observation_model']
    if m['status'] not in ('designed_unfitted', 'fitted_on_calibration', 'partially_fitted_on_calibration'): raise ValueError('Invalid observation model status')
    if m['status']=='partially_fitted_on_calibration':
        if c['version']!=4 or m.get('fitted_axes')!=['interpretability','appraisal']:
            raise ValueError('Partial fit must identify the two fitted semantic axes')
    if 'scalar_calibration' in c:
        from nod.semantic.sensor import prompt_digest
        calibration=c['scalar_calibration']
        if c['version']!=4 or set(calibration['temperature'])!={'completion','response_demand'}:
            raise ValueError('Invalid scalar calibration axes')
        for value in calibration['temperature'].values():finite(value,.05,20)
        checksum=calibration.get('artifact_sha256','')
        if not isinstance(checksum,str) or len(checksum)!=64 or any(ch not in '0123456789abcdef' for ch in checksum):
            raise ValueError('Scalar calibration artifact digest required')
        if calibration.get('model')!=cfg['semantic']['model'] or calibration.get('prompt_sha256')!=prompt_digest(resource):
            raise ValueError('Scalar calibration model or question mismatch')
    if c['version']==4:
        from nod.belief.reports import FAMILY,REPORTS
        if m['family']!=FAMILY or m.get('reports')!=list(REPORTS) or m.get('probability_floor')!=1e-6:
            raise ValueError('Report observation model mismatch')
        if set(m['concentration'])!={'perception','interpretability','appraisal'}:raise ValueError('Invalid report axes')
        for v in m['concentration'].values():finite(v,.05,20)
    else:
        if m['family'] != 'joint_logistic_normal_appraisal_marginalized' or m['features'] != list(FEATURES): raise ValueError('Observation model mismatch')
        for matrix, rows in ((m['means'], 6), (m['covariance'], 5)):
            if len(matrix) != rows or any(len(row) != 5 for row in matrix): raise ValueError('Observation dimensions mismatch')
            for row in matrix:
                for v in row: finite(v, -1000, 1000)
        if any(abs(m['covariance'][i][j]-m['covariance'][j][i]) > 1e-10 for i in range(5) for j in range(5)):
            raise ValueError('Covariance must be symmetric')
        cholesky(m['covariance'])
    p = c['policy']; functions = ('none', 'continuer', 'understanding', 'empathic')
    if 'decision_model' in p:
        from nod.policy.risk import validate_model
        validate_model(p['decision_model'])
        if c['version']!=4 or 'local_receipt' in p:raise ValueError('Risk policy requires v4 without readiness override')
        if c.get('temporal_model')!='single_reset_kernel' or c.get('commit_scope')!='asr_utterance':
            raise ValueError('Risk listener requires explicit temporal and evidence scope models')
        for name in ('history_window_ms','max_actions_per_window'):finite(p[name],1)
        if isinstance(p['max_actions_per_window'],bool) or not isinstance(p['max_actions_per_window'],int):
            raise ValueError('Action count must be integer')
        for f in functions[1:]:
            lo,hi=p['intensity_ranges'][f];finite(lo,0,1);finite(hi,lo,1)
        finite(p['expression_upgrade_gap_ms'],0,cfg['motor']['cooldown_after_completion_ms'])
        finite(p['continuer_max_age_ms'],1,cfg['semantic']['source_max_age_ms'])
        inc=c['incremental'];finite(inc['retention_ms'],1,cfg['semantic']['source_max_age_ms'])
        if isinstance(inc['max_items'],bool) or not isinstance(inc['max_items'],int):raise ValueError('Invalid memory size')
        finite(inc['max_items'],1,32)
        for name in ('retain_interpretability','retain_reliability'):finite(inc[name],0,1)
        if cfg['asr'].get('preserve_final_jobs'):
            capacity=cfg['asr']['final_queue_capacity']
            if isinstance(capacity,bool) or not isinstance(capacity,int) or not 1<=capacity<=4:raise ValueError('Invalid ASR final queue')
        return
    if set(p['utility']) != set(functions): raise ValueError('Invalid utility actions')
    for row in p['utility'].values():
        if len(row) != 6: raise ValueError('Invalid utility state count')
        for v in row: finite(v)
    for name in ('minimum_completion', 'maximum_response_demand', 'minimum_understanding', 'expression_minimum_probability'):
        finite(p[name], 0, 1)
    for group in ('motion_cost', 'premature_cost'):
        if set(p[group]) != set(functions[1:]): raise ValueError('Invalid action costs')
        for v in p[group].values(): finite(v, 0)
    for f in functions[1:]:
        lo, hi = p['intensity_ranges'][f]; finite(lo, 0, 1); finite(hi, lo, 1)
    for v in p['timing'].values(): finite(v, -20, 20)
    finite(p['timing']['confirmed_silence_weight'], 0, 1)
    for name in ('history_window_ms', 'max_actions_per_window'): finite(p[name], 1)
    if not isinstance(p['max_actions_per_window'], int): raise ValueError('Action count must be integer')
    finite(p['minimum_gain'], 0); finite(p['repetition_cost'], 0)

    if c['version']>=3:
        inc=c['incremental']
        finite(inc['retention_ms'],1,cfg['semantic']['source_max_age_ms'])
        if not isinstance(inc['max_items'],int) or isinstance(inc['max_items'],bool):raise ValueError('Invalid memory size')
        finite(inc['max_items'],1,32)
        for name in ('retain_interpretability','retain_reliability'):finite(inc[name],0,1)
        for name in ('expression_readiness','expression_minimum_perception','expression_minimum_mass'):finite(p[name],0,1)
    if c['version']==4:
        if p.get('continuer_requires_progress') is not True:raise ValueError('v4 requires progress-gated continuers')
        finite(p['expression_upgrade_gap_ms'],0,1800)
        finite(p['expression_minimum_sensor_mass'],0,1)
        finite(p['continuer_max_age_ms'],1,cfg['semantic']['source_max_age_ms'])
        if p.get('expression_can_refine') is not True:raise ValueError('v4 requires expression refinement')
    if 'local_receipt' in p:
        if c['version']!=4:raise ValueError('Local receipt requires the report observation model')
        local=p['local_receipt']
        for name in ('readiness','minimum_resolved'):finite(local[name],0,1)
        if not isinstance(local['suppress_filler_only'],bool):raise ValueError('Invalid filler guard')
    if 'temporal_model' in c and c['temporal_model']!='single_reset_kernel':raise ValueError('Unsupported temporal model')
    if 'commit_scope' in c and c['commit_scope']!='asr_utterance':raise ValueError('Unsupported evidence commit scope')
    if cfg['asr'].get('preserve_final_jobs'):
        if c['version']!=4:raise ValueError('Final job telemetry requires v4')
        capacity=cfg['asr']['final_queue_capacity']
        if isinstance(capacity,bool) or not isinstance(capacity,int) or not 1<=capacity<=4:
            raise ValueError('Final ASR queue capacity must be 1–4')
