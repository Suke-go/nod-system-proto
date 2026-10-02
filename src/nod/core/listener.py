"""Finite Bayesian listener: joint observations -> state belief -> feedback ledger."""
from collections import deque
import hashlib
from nod.asr.stability import TranscriptTracker
from nod.belief.listener import ListenerBelief
from nod.config import probability
from nod.core.events import source_time
from nod.policy.listener import ListenerPolicy
from nod.policy.motor import Motor
from nod.semantic.sensor import SCHEMA, observation, compatibility
from nod.semantic.units import UnitTracker
from nod.timing.opportunity import OpportunityGate


class ListenerEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transcripts = TranscriptTracker(cfg)
        self.units = UnitTracker(cfg['semantic']['max_context_characters'])
        self.belief = ListenerBelief(cfg['listener'])
        self.policy = ListenerPolicy(cfg['listener']['policy'])
        self.opportunity = OpportunityGate(cfg['timing'])
        self.motor = Motor(cfg['motor'])
        self.now = 0
        self.connected = False
        self.neutral_expression = False
        self.expressions = ['neutral']
        self.input_healthy = True
        self.semantic = None
        self.last_seq = 0
        self.prominence = None
        self.history = deque(maxlen=128)
        self.semantic_trace = None
        self.last_utilities = {}
        self.decision_context = {}
        self.reasons = ['semantic_missing']

    def semantic_input(self):
        return self.units.current, self.units.context

    def process(self, event):
        if event.at_ms < self.now:
            raise ValueError('Ingest time moved backwards')
        self.now = now = event.at_ms
        data = event.data
        records = []
        old_motor = self.motor.state
        self.motor.tick(now)
        if event.kind == 'asr':
            source = source_time(data,now)
            old = self.transcripts.current
            snapshot = self.transcripts.update(data['utterance_id'],data['text'],source,now,
                data.get('confidence'),data.get('final',False),data.get('final_reason'))
            repaired = bool(old and old.utterance_id==snapshot.utterance_id and
                            old.repair_epoch!=snapshot.repair_epoch)
            unit,new,local_repair = self.units.update(snapshot,self.transcripts.recent_context())
            if new:
                self.belief.open_unit(now,revoke=repaired)
                self.semantic = None
            elif local_repair:
                self.belief.revoke()
                self.semantic = None
            if repaired:
                for entry in self.history:
                    if entry['utterance_id'] == snapshot.utterance_id and not entry['invalidated']:
                        start = entry['start']
                        if not snapshot.text[start:].startswith(entry['evidence_text']):
                            entry['invalidated'] = True
                            records.append({'kind':'feedback_evidence_revoked','action_id':entry['action_id'],
                                            'reason':'asr_repair_after_dispatch'})
            records.append({'kind':'transcript','snapshot':snapshot.to_dict(),
                            'semantic_unit':unit.to_dict(),'unit_start':self.units.start,
                            'operation':'add' if new else 'revoke_replace' if local_repair else 'replace'})
            if self.semantic and self.semantic['text']==unit.text:
                # Same evidence, changed ASR reliability. Retain its acquisition
                # time: silence must not make an old interpretation fresh again.
                self.semantic['reliability'] = unit.reliability
        elif event.kind == 'opportunity':
            source = source_time(data,now)
            self.opportunity.update(data['score'],source,now)
            self.belief.acoustic(data['score'],source,now)
        elif event.kind == 'semantic_result':
            reason = self._semantic(data,now)
            records.append({'kind':'semantic_disposition','seq':data['request']['seq'],'reason':reason})
        elif event.kind == 'prominence':
            strength,source = probability(data['strength']),source_time(data,now)
            if self.prominence is None or source >= self.prominence[1]:
                self.prominence = (strength,source)
        elif event.kind == 'controller_ready':
            self.connected = True
            self.neutral_expression = bool(data.get('neutral_expression'))
            self.expressions = data.get('supported_expressions',['neutral'])
            if self.motor.state=='FAULT' and self.motor.command:
                records.append({'kind':'query_status','action_id':self.motor.command['action_id'],'reason':'reconnected'})
        elif event.kind == 'controller_disconnected':
            self.connected = False
            if self.motor.state in ('SENT','ACTIVE'):
                self.motor.fail('connection_lost')
        elif event.kind == 'feedback':
            changed = self.motor.feedback(data['action_id'],data['status'],now)
            if changed:
                for entry in self.history:
                    if entry['action_id']==data['action_id']:
                        entry['status']=data['status']
                        entry['feedback_ms']=now
        elif event.kind == 'input_health':
            self.input_healthy = bool(data['healthy'])
        elif event.kind not in ('tick','semantic_error'):
            raise ValueError('Unsupported event kind')
        if old_motor!='FAULT' and self.motor.state=='FAULT':
            records.append({'kind':'query_status','action_id':self.motor.command['action_id'],
                            'reason':self.motor.fault_reason})
        command = self._decide(now)
        if event.kind == 'semantic_result' and reason == 'accepted':
            self.semantic_trace = {'seq':self.last_seq, 'schema':SCHEMA,
                'unit_id':self.units.current.utterance_id, 'observation':self.semantic['observation'],
                'model':data['result'].get('model'), 'state':self.belief.summary(now),
                'evidence_rule':'replace_joint_packet_from_unit_checkpoint'}
            records.append({'kind':'semantic_trace', **self.semantic_trace})
        if command:
            records.append(command)
        if self.cfg['logging'].get('decision_diagnostics'):
            records.append({'kind':'decision_status','reasons':self.reasons,'motor_state':self.motor.state,
                'semantic_age_ms':None if not self.semantic else now-self.semantic['source_ms'],
                'opportunity_score':self.opportunity.score,'belief':self.belief.at(now),
                'listener':self.belief.summary(now),'utilities':self.last_utilities,
                'decision_context':self.decision_context,
                'unit_id':self.units.current.utterance_id if self.units.current else None,
                'feedback_count':len(self.history)})
        return records

    def _semantic(self,data,now):
        request,result = data['request'],data['result']
        snapshot = request['snapshot']
        unit = self.units.current
        seq = request['seq']
        if isinstance(seq,bool) or not isinstance(seq,int) or seq <= self.last_seq:
            return 'duplicate_or_superseded'
        if unit is None or snapshot['utterance_id']!=unit.utterance_id:
            return 'prior_unit'
        if snapshot['repair_epoch']!=unit.repair_epoch or not unit.text.startswith(snapshot['text']):
            return 'repaired_input'
        source = source_time(snapshot,now)
        if (request['dispatched_ms']>now or now-request['dispatched_ms']>self.cfg['semantic']['total_deadline_ms']
                or now-source>self.cfg['semantic']['source_max_age_ms']):
            return 'expired'
        try:
            if result.get('schema')!=SCHEMA:
                return 'incompatible_semantic_schema'
            values = observation(result['observation'])
        except (ValueError,KeyError,TypeError):
            return 'invalid_semantic_observation'
        self.last_seq = seq
        self.semantic = {**snapshot, 'observation':values, 'probabilities':compatibility(values)}
        return 'accepted'

    def _decide(self, now):
        unit = self.units.current
        acoustic_only = self.cfg['listener']['condition']=='acoustic_only'
        self.decision_context = {}; self.last_utilities = {}
        exact = bool(unit and self.semantic and unit.text==self.semantic['text'])
        fresh_semantic = bool(self.semantic and now-self.semantic['source_ms']<=self.cfg['semantic']['source_max_age_ms'])
        obs = self.semantic['observation'] if exact and fresh_semantic and not acoustic_only else None
        if unit:
            # Appended text retracts the old semantic factor until refreshed.
            # Perception and semantics form ONE replaceable correlated packet.
            stamp = min(unit.source_ms, self.semantic['source_ms']) if obs else unit.source_ms
            self.belief.observe(unit.reliability, obs, stamp)
        reasons = []
        if not self.connected: reasons.append('controller_unavailable')
        if not self.input_healthy: reasons.append('input_unhealthy')
        if self.motor.state!='IDLE': reasons.append('motor_'+self.motor.state.lower())
        if not unit or not unit.text: reasons.append('asr_missing')
        elif now-unit.source_ms>self.cfg['semantic']['source_max_age_ms']: reasons.append('asr_stale')
        elif unit.reliability<self.cfg['policy']['minimum_asr_reliability']: reasons.append('asr_unreliable')
        audio = self.belief.audio
        if audio is None or now-audio[1]>self.cfg['listener']['audio_max_age_ms']: reasons.append('audio_stale')
        self.reasons = reasons
        if reasons: return None
        specific = bool(obs and unit.reliability>=self.cfg['listener']['specific_minimum_reliability'])
        # Argmax is applied to decision-context scores as well as state observations.
        if obs and self.cfg['listener']['condition']=='argmax':
            obs = {**obs, 'completion':float(obs['completion']>=.5),
                   'response_demand':float(obs['response_demand']>=.5)}
        function, self.last_utilities, self.decision_context = self.policy.choose(
            self.belief.posterior(now), self.history, now, unit.utterance_id, obs,
            audio[0], bool(unit.final and unit.final_reason=='silence'), specific)
        if function=='none':
            self.reasons = [self.decision_context['reason']]
            return None
        summary = self.belief.summary(now)
        degraded = self.prominence is None or now-self.prominence[1]>self.cfg['prominence']['max_audio_age_ms']
        strength = self.cfg['prominence']['default_strength'] if degraded else self.prominence[0]
        action, intensity, expression = self.policy.embody(function, summary, strength, self.neutral_expression)
        if expression not in self.expressions: expression='neutral'
        command = self.motor.start(action, intensity, now, unit.utterance_id)
        command['expression'] = expression
        evidence = self.semantic if obs else unit.to_dict()
        self.history.append({'unit_id':unit.utterance_id, 'utterance_id':self.transcripts.current.utterance_id,
            'start':self.units.start, 'evidence_text':evidence['text'],
            'text_sha256':hashlib.sha256(evidence['text'].encode()).hexdigest(),
            'semantic_seq':self.last_seq if obs else None, 'action_id':command['action_id'], 'function':function,
            'at_ms':now, 'status':'dispatched', 'invalidated':False})
        self.reasons = ['action_selected']
        return {'kind':'command', 'command':command, 'function':function, 'belief':self.belief.at(now),
            'mass':self.belief.at(now), 'utilities':self.last_utilities, 'listener':summary,
            'unit_id':unit.utterance_id, 'semantic_seq':self.last_seq if obs else None,
            'trigger':'listener_belief', 'decision_context':self.decision_context,
            'evidence_source_ms':evidence['source_ms'], 'prominence_degraded':degraded}
