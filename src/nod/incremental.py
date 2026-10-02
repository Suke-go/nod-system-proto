"""Versioned incremental listener extension; v2 replay remains unchanged."""
from nod.listener import configure as base_configure


def configure(cfg,condition='full'):
    cfg=base_configure(cfg,condition)
    cfg['profile']='listener_incremental_v3'
    c=cfg['listener'];c['version']=3
    c['incremental']={'retention_ms':3500,'max_items':8,
                      'retain_interpretability':.55,'retain_reliability':.55}
    p=c['policy']
    p.update(expression_readiness=.85,expression_minimum_perception=.7,expression_minimum_mass=.25)
    p['utility']['empathic']=[-1.5,-.1,-.8,1.8,1.8,1.8]
    p['premature_cost']['empathic']=.2
    p['intensity_ranges']['empathic']=[.15,.4]
    cfg['research'].update(protocol_version='listener-v3',
        method='finite_bayes_filter_scoped_evidence_function_specific_policy',
        structural_basis=['Kopp et al. 2007','Schlangen & Skantze 2009','Meguro et al. 2010'])
    return cfg
