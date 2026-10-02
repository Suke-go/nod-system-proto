from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json
import math

CLASSES = ("no_bc", "continuer", "understanding", "empathic")


def strict_json(text):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(text, parse_constant=invalid)


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Probability must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability must be finite and in [0,1]")
    return float(value)


def distribution(values):
    if isinstance(values, dict):
        if set(values) != set(CLASSES):
            raise ValueError("Expected exactly four listener function keys")
        values = [values[key] for key in CLASSES]
    if len(values) != 4:
        raise ValueError("Expected four probabilities")
    result = tuple(probability(x) for x in values)
    total = sum(result)
    if abs(total - 1) > 1e-4:
        raise ValueError("Probability sum is not one")
    return tuple(x / total for x in result)


def load_config(path=None):
    text = Path(path).read_text(encoding="utf-8-sig") if path else files("nod").joinpath("resources/default.json").read_text(encoding="utf-8")
    cfg = strict_json(text)
    validate_config(cfg)
    return deepcopy(cfg)


def validate_config(c):
    if 'listener' in c:
        from nod.listener import validate
        validate(c)
    if 'interaction' in c:
        from nod.production import validate_interaction
        validate_interaction(c)
    if c["schema_version"] != 1 or c["belief"]["class_order"] != list(CLASSES):
        raise ValueError("Unsupported schema version or class order")
    distribution(c["belief"]["initial"])
    distribution(c["belief"]["transition"]["stationary"])
    for group, keys in {
        "runtime": ["control_tick_ms", "event_queue_capacity"],
        "semantic": ["total_deadline_ms", "source_max_age_ms", "min_dispatch_interval_ms", "max_coalesce_wait_ms"],
        "timing": ["opportunity_max_age_ms", "open_hold_ms", "rearm_hold_ms", "expire_open_window_ms"],
        "motor": ["ack_timeout_ms", "execution_timeout_ms", "command_ttl_ms"],
    }.items():
        for key in keys:
            value = c[group][key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{group}.{key} must be positive")
    t = c["timing"]
    for group,key in [('semantic','refine_exact_final'),('logging','research_diagnostics')]:
        if key in c[group] and not isinstance(c[group][key],bool):
            raise ValueError(f'{group}.{key} must be boolean')
    if c['belief'].get('observation_rule','likelihood_product') not in ('likelihood_product','posterior_blend','direct'):
        raise ValueError('Unsupported belief observation rule')
    if c['belief'].get('history_mode','carry') not in ('carry','none'):
        raise ValueError('Unsupported belief history mode')
    if c['belief'].get('observation_rule')=='direct' and c['belief'].get('history_mode')!='none':
        raise ValueError('Direct observation requires history_mode=none')
    if c['policy'].get('understanding_action') not in (None,'STRONG_NOD'):
        raise ValueError('Unsupported understanding action')
    if 'stable_clause_ack' in c['policy'] and not isinstance(c['policy']['stable_clause_ack'],bool):
        raise ValueError('stable_clause_ack must be boolean')
    endpoint=c['policy'].get('endpoint_specific_ack')
    if endpoint is not None:
        if not isinstance(endpoint,dict) or not isinstance(endpoint.get('enabled'),bool):
            raise ValueError('Invalid endpoint acknowledgement configuration')
        for key in ('minimum_reliability','minimum_probability','minimum_margin','minimum_belief'):
            probability(endpoint[key])
        if 'allowed_final_reasons' in endpoint:
            reasons=endpoint['allowed_final_reasons']
            if not isinstance(reasons,list) or not reasons or any(x not in ('silence','max_duration','input_end') for x in reasons):
                raise ValueError('Invalid allowed final reasons')
    if c['policy'].get('opportunity_weighting', 'probability_mass') not in ('probability_mass', 'gate_only'):
        raise ValueError('Unsupported opportunity weighting')
    if not probability(t["close_threshold"]) < probability(t["open_threshold"]):
        raise ValueError("Opportunity thresholds must have hysteresis")
    if c["belief"]["transition"]["half_life_ms"] <= 0:
        raise ValueError("Transition half-life must be positive")
    if c['asr']['stable_prefix_agreement_updates'] != 2:
        raise ValueError('This implementation requires two consecutive ASR updates for stability')
    if c['policy']['max_actions_per_opportunity'] != 1 or c['motor']['automatic_execute_retry']:
        raise ValueError('Only one action per opportunity and no automatic execute retry are supported')
    if not c['logging']['record_transcript'] or c['logging']['record_audio'] or c['logging']['include_secret_headers']:
        raise ValueError('This milestone logs transcripts, never audio or secret headers')
    if not isinstance(c['output']['port'], int) or not 0 <= c['output']['port'] <= 65535:
        raise ValueError('Invalid controller port')
    for key in ('asr_confidence', 'stability', 'revision_quality'):
        probability(c['belief']['reliability_weights'][key])
    if sum(c['belief']['reliability_weights'].values()) <= 0:
        raise ValueError('Reliability weights cannot all be zero')
    if set(c['policy']['utility_by_function']) != {'none', 'continuer', 'understanding', 'empathic'}:
        raise ValueError('Invalid utility function keys')
    if c["semantic"]["max_inflight"] != 1 or c["semantic"]["pending_slots"] != 1:
        raise ValueError("This version supports one active and one pending request")
    if c["policy"]["allow_clap"] or c["timing"]["scheduled_onset_enabled"]:
        raise ValueError("CLAP and scheduled onset are not enabled in this milestone")
    if c["output"]["bind_host"] not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Controller transport must bind to loopback")
    if c["policy"]["tie_break_order"] != ["none", "continuer", "understanding", "empathic"]:
        raise ValueError("Unsupported policy order")
    for row in c["policy"]["utility_by_function"].values():
        if len(row) != 4 or not all(math.isfinite(x) for x in row):
            raise ValueError("Invalid utility matrix")
    for low, high in c["policy"]["intensity_ranges"].values():
        if not probability(low) <= probability(high):
            raise ValueError("Invalid intensity range")
    probability(c["policy"]["minimum_asr_reliability"])
