from nod.asr.stability import TranscriptTracker
from nod.belief.filter import BeliefFilter
from nod.config import distribution, probability
from nod.core.events import source_time
from nod.policy.motor import Motor
from nod.policy.selector import ExpectedUtilityPolicy
from nod.timing.opportunity import OpportunityGate


def create_engine(cfg):
    if cfg.get('listener',{}).get('version') in (3,4):
        from nod.core.incremental import IncrementalListenerEngine
        return IncrementalListenerEngine(cfg)
    if cfg.get('listener',{}).get('enabled'):
        from nod.core.listener import ListenerEngine
        return ListenerEngine(cfg)
    if cfg.get('interaction',{}).get('enabled'):
        from nod.core.interaction import InteractionEngine
        return InteractionEngine(cfg)
    return Engine(cfg)


class Engine:
    """Deterministic reducer. Only this object mutates decision state."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.transcripts = TranscriptTracker(cfg)
        self.belief = BeliefFilter(cfg['belief'])
        self.opportunity = OpportunityGate(cfg['timing'])
        self.policy = ExpectedUtilityPolicy(cfg['policy'])
        self.motor = Motor(cfg['motor'])
        self.now = 0
        self.epoch_ms = 0
        self.semantic = None
        self.last_seq = 0
        self.prominence = None
        self.connected = False
        self.neutral_expression = False
        self.input_healthy = True
        self.specific_ack_utterance = None
        self.semantic_trace = None

    def process(self, event):
        if event.at_ms < self.now:
            raise ValueError('Ingest time moved backwards')
        self.now = event.at_ms
        data, now = event.data, self.now
        records = []
        old_motor_state = self.motor.state
        self.motor.tick(now)
        if event.kind == 'asr':
            source = source_time(data, now)
            previous = self.transcripts.current
            snapshot = self.transcripts.update(data['utterance_id'], data['text'], source, now,
                                               data.get('confidence'), data.get('final', False), data.get('final_reason'))
            new_epoch = previous is None or previous.utterance_id != snapshot.utterance_id or now - self.epoch_ms >= self.cfg['belief']['max_semantic_epoch_ms']
            repaired = previous is not None and previous.utterance_id == snapshot.utterance_id and previous.repair_epoch != snapshot.repair_epoch
            if repaired:
                self.belief.invalidate(now)
                self.semantic = None
            if new_epoch:
                self.belief.open_epoch(now)
                self.epoch_ms = now
                self.semantic = None
            records.append({'kind': 'transcript', 'snapshot': snapshot.to_dict()})
            if self._refine_final(now):
                records.append({'kind':'semantic_refinement','reason':'identical_text_finalized',
                                'trace':self.semantic_trace})
        elif event.kind == 'opportunity':
            self.opportunity.update(data['score'], source_time(data, now), now)
        elif event.kind == 'prominence':
            p, source = probability(data['strength']), source_time(data, now)
            if self.prominence is None or source >= self.prominence[1]:
                self.prominence = (p, source)
        elif event.kind == 'semantic_result':
            disposition = self._semantic(data, now)
            records.append({'kind': 'semantic_disposition', 'seq': data['request']['seq'], 'reason': disposition})
            if disposition=='accepted' and self.cfg['logging'].get('research_diagnostics'):
                records.append({'kind':'semantic_trace',**self.semantic_trace})
        elif event.kind == 'controller_ready':
            self.connected = True
            self.neutral_expression = bool(data.get('neutral_expression', False))
            if self.motor.state == 'FAULT' and self.motor.command:
                records.append({'kind': 'query_status', 'action_id': self.motor.command['action_id'],
                                'reason': 'reconnected'})
        elif event.kind == 'controller_disconnected':
            self.connected = False
            if self.motor.state in ('SENT', 'ACTIVE'):
                self.motor.fail('connection_lost')
        elif event.kind == 'feedback':
            self.motor.feedback(data['action_id'], data['status'], now)
        elif event.kind == 'input_health':
            self.input_healthy = bool(data['healthy'])
        elif event.kind not in ('tick', 'semantic_error'):
            raise ValueError('Unsupported event kind')
        if self.motor.state == 'FAULT' and old_motor_state != 'FAULT':
            records.append({'kind': 'query_status', 'action_id': self.motor.command['action_id'],
                            'reason': self.motor.fault_reason})
        command = self._decide(now)
        if command:
            records.append(command)
        elif self.cfg['logging'].get('decision_diagnostics',False):
            records.append(self._diagnostics(now))
        return records

    def _diagnostics(self, now):
        current, semantic = self.transcripts.current, self.semantic
        reasons = []
        if not self.opportunity.eligible(now): reasons.append('opportunity_unavailable')
        if not self.connected: reasons.append('controller_unavailable')
        if not self.input_healthy: reasons.append('input_unhealthy')
        if self.motor.state != 'IDLE': reasons.append('motor_'+self.motor.state.lower())
        if current is None: reasons.append('asr_missing')
        elif current.reliability < self.cfg['policy']['minimum_asr_reliability']: reasons.append('asr_unreliable')
        if semantic is None: reasons.append('semantic_missing')
        elif now-semantic['source_ms'] > self.cfg['semantic']['source_max_age_ms']: reasons.append('semantic_stale')
        if current and self.specific_ack_utterance == current.utterance_id:
            reasons.append('utterance_already_acknowledged')
        if not reasons: reasons.append('utility_prefers_none')
        record = {'kind':'decision_status','reasons':reasons,'motor_state':self.motor.state,
                'opportunity_score':self.opportunity.score,
                'semantic_age_ms':None if semantic is None else now-semantic['source_ms']}
        if self.cfg['logging'].get('research_diagnostics'):
            specific=bool(current and semantic and current.source_ms-semantic['source_ms']<=self.cfg['semantic']['append_lag_for_specific_response_ms'])
            record.update(belief=self.belief.at(now),current_final=bool(current and current.final),
                semantic_final=bool(semantic and semantic.get('final')),
                semantic_reliability=semantic['reliability'] if semantic else None,
                specific_allowed=specific,
                endpoint_eligible=bool(current and semantic and self._endpoint_eligible(now,current,semantic,specific)),
                choice_without_timing=self.policy.choose(1.,self.belief.at(now),specific).function)
            if self.cfg['policy'].get('stable_clause_ack'):
                record['stable_clause_eligible']=bool(current and semantic and self._clause_eligible(now,current,semantic,specific))
        return record

    def _semantic(self, data, now):
        request, result = data['request'], data['result']
        snapshot, current = request['snapshot'], self.transcripts.current
        seq = request['seq']
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= self.last_seq:
            return 'duplicate_or_superseded'
        if current is None or snapshot['utterance_id'] != current.utterance_id:
            return 'prior_utterance'
        if snapshot['repair_epoch'] != current.repair_epoch or not current.text.startswith(snapshot['text']):
            return 'repaired_input'
        if snapshot['created_ms'] < self.epoch_ms:
            return 'prior_epoch'
        cfg = self.cfg['semantic']
        source = source_time(snapshot, now)
        if request['dispatched_ms'] > now or now - request['dispatched_ms'] > cfg['total_deadline_ms'] or now - source > cfg['source_max_age_ms']:
            return 'expired'
        p = distribution(result['probabilities'])
        self.semantic = dict(snapshot)
        if self.cfg['policy'].get('endpoint_specific_ack',{}).get('enabled') or cfg.get('refine_exact_final'):
            self.semantic['probabilities'] = p
        self.last_seq = seq
        if cfg.get('refine_exact_final') and current.final and not snapshot.get('final') and current.text==snapshot['text']:
            self.semantic.update(current.to_dict(),probabilities=p,original_source_ms=source)
        q=probability(self.semantic['reliability'])
        self._observe(p,q,now)
        return 'accepted'

    def _observe(self,p,q,now):
        prior=self.belief.observation_prior(now)
        posterior=self.belief.observe(p,q,now)
        self.semantic_trace={'seq':self.last_seq,'raw':p,'predicted':prior,'reliability':q,
            'posterior':posterior,'source_final':self.semantic.get('final',False),
            'final_refined':'original_source_ms' in self.semantic}

    def _refine_final(self,now):
        c,s=self.transcripts.current,self.semantic
        if not (self.cfg['semantic'].get('refine_exact_final') and c and s and c.final and not s.get('final')):
            return False
        if (c.utterance_id!=s['utterance_id'] or c.repair_epoch!=s['repair_epoch'] or c.text!=s['text']
                or now-s['source_ms']>self.cfg['semantic']['source_max_age_ms']):
            return False
        p=s['probabilities']; original=s['source_ms']
        self.semantic={**c.to_dict(),'probabilities':p,'original_source_ms':original}
        # Replace against the epoch anchor: the same observation is not multiplied twice.
        self._observe(p,c.reliability,now)
        return True

    def _decide(self, now):
        eligible = self.opportunity.eligible(now)
        current, semantic = self.transcripts.current, self.semantic
        if not (self.input_healthy and self.connected and self.motor.state == 'IDLE' and current and semantic):
            return None
        if now - semantic['source_ms'] > self.cfg['semantic']['source_max_age_ms'] or current.reliability < self.cfg['policy']['minimum_asr_reliability']:
            return None
        specific = current.source_ms - semantic['source_ms'] <= self.cfg['semantic']['append_lag_for_specific_response_ms']
        endpoint = self._endpoint_eligible(now, current, semantic, specific)
        clause = self._clause_eligible(now, current, semantic, specific)
        if not eligible and not endpoint and not clause:
            return None
        selection = self.policy.choose(1. if endpoint or clause else self.opportunity.score, self.belief.at(now), specific)
        if selection.function == 'none':
            return None
        if not eligible and selection.function not in ('understanding','empathic'):
            return None
        trigger = 'vap' if eligible else 'utterance_end' if endpoint else 'stable_clause'
        if self.specific_ack_utterance == current.utterance_id:
            return None
        degraded = self.prominence is None or now - self.prominence[1] > self.cfg['prominence']['max_audio_age_ms']
        strength = self.cfg['prominence']['default_strength'] if degraded else self.prominence[0]
        action, intensity = self.policy.embody(selection.function, strength, self.neutral_expression)
        opportunity_id = self.opportunity.window_id if eligible else ('clause-' if trigger=='stable_clause' else 'end-')+current.utterance_id
        command = self.motor.start(action, intensity, now, opportunity_id)
        if eligible: self.opportunity.consume()
        record = {'kind': 'command', 'command': command, 'function': selection.function,
                'belief': self.belief.at(now), 'mass': selection.mass, 'utilities': selection.utilities,
                'prominence_degraded': degraded}
        if self.cfg['policy'].get('endpoint_specific_ack',{}).get('enabled'):
            record['trigger'] = trigger
            if selection.function in ('understanding','empathic'):
                self.specific_ack_utterance = current.utterance_id
        return record

    def _endpoint_eligible(self, now, current, semantic, specific):
        cfg = self.cfg['policy'].get('endpoint_specific_ack',{})
        if not (cfg.get('enabled') and specific and current.final and semantic.get('final')
                and self.specific_ack_utterance != current.utterance_id):
            return False
        reasons=cfg.get('allowed_final_reasons')
        if reasons is not None and (current.final_reason not in reasons or semantic.get('final_reason') not in reasons):
            return False
        return self._specific_evidence(now,current,semantic,cfg)

    def _clause_eligible(self,now,current,semantic,specific):
        # Presentation-only heuristic. ASR punctuation is an imperfect clause cue,
        # not a turn-end label or another semantic observation.
        if not (self.cfg['policy'].get('stable_clause_ack') and specific
                and self.specific_ack_utterance!=current.utterance_id and not current.final):
            return False
        text=current.text.rstrip()
        if (not text or text[-1] not in '.!?。！？' or text.endswith('..')
                or current.stable.rstrip()!=text or semantic['text']!=current.text):
            return False
        return self._specific_evidence(now,current,semantic,self.cfg['policy']['endpoint_specific_ack'])

    def _specific_evidence(self,now,current,semantic,cfg):
        if min(current.reliability,semantic['reliability']) < cfg['minimum_reliability']:
            return False
        # Live audio must still be arriving; this is a final-utterance trigger,
        # never a timer-only fallback for a stopped microphone or old text.
        source = self.opportunity.source_ms
        if source is None or now-source > self.cfg['timing']['opportunity_max_age_ms']:
            return False
        p=semantic['probabilities']; index=2 if p[2]>=p[3] else 3
        return (p[index]>=cfg['minimum_probability']
                and p[index]-max(p[j] for j in range(4) if j!=index)>=cfg['minimum_margin']
                and self.belief.at(now)[index]>=cfg['minimum_belief'])
