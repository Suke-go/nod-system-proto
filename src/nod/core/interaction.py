"""Production reducer: causal evidence -> joint belief -> feedback -> execution ledger."""
from collections import deque
import hashlib
from nod.asr.stability import TranscriptTracker
from nod.belief.interaction import InteractionBelief
from nod.config import probability
from nod.core.events import source_time
from nod.policy.interaction import InteractionPolicy
from nod.policy.motor import Motor
from nod.semantic.frames import SCHEMA, frame_distribution, FRAME_ORDER, function_distribution
from nod.semantic.units import UnitTracker
from nod.timing.opportunity import OpportunityGate


class InteractionEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transcripts = TranscriptTracker(cfg)
        self.units = UnitTracker(cfg['semantic']['max_context_characters'])
        self.belief = InteractionBelief(cfg['interaction'])
        self.policy = InteractionPolicy(cfg['interaction']['policy'])
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
                self.belief.reliability = unit.reliability
        elif event.kind == 'opportunity':
            source = source_time(data,now)
            self.opportunity.update(data['score'],source,now)
            self.belief.acoustic(data['score'],source,now)
        elif event.kind == 'semantic_result':
            reason = self._semantic(data,now)
            records.append({'kind':'semantic_disposition','seq':data['request']['seq'],'reason':reason})
            if reason=='accepted':
                records.append({'kind':'semantic_trace',**self.semantic_trace})
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
        if command:
            records.append(command)
        if self.cfg['logging'].get('decision_diagnostics'):
            records.append({'kind':'decision_status','reasons':self.reasons,'motor_state':self.motor.state,
                'semantic_age_ms':None if not self.semantic else now-self.semantic['source_ms'],
                'opportunity_score':self.opportunity.score,'belief':self.belief.at(now),
                'interaction':self.belief.summary(now),'utilities':self.last_utilities,
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
            values = frame_distribution(result['frames'])
        except (ValueError,KeyError,TypeError):
            return 'invalid_semantic_observation'
        self.last_seq = seq
        self.semantic = {**snapshot,'frames':values,'probabilities':function_distribution(values)}
        q = unit.reliability if unit.text==snapshot['text'] else snapshot['reliability']
        prior = self.belief.frames_at(now)
        self.belief.observe(values,q,now)
        self.semantic_trace = {'seq':seq,'schema':SCHEMA,'unit_id':unit.utterance_id,
            'raw_frames':dict(zip(FRAME_ORDER,values)),'predicted_frames':prior,
            'reliability':q,'posterior_frames':self.belief.frames_at(now),
            'model':result.get('model'),'model_confidence':result.get('confidence'),
            'evidence_rule':'replace_current_unit_factor','calibration':self.belief.cfg['calibration']}
        return 'accepted'

    def _decide(self,now):
        unit = self.units.current
        acoustic_only = self.cfg['interaction']['condition']=='acoustic_only'
        reasons = []
        if not self.connected: reasons.append('controller_unavailable')
        if not self.input_healthy: reasons.append('input_unhealthy')
        if self.motor.state!='IDLE': reasons.append('motor_'+self.motor.state.lower())
        if not unit or not unit.text: reasons.append('asr_missing')
        elif unit.reliability<self.cfg['policy']['minimum_asr_reliability']: reasons.append('asr_unreliable')
        if not acoustic_only:
            if self.semantic is None: reasons.append('semantic_missing')
            elif now-self.semantic['source_ms']>self.cfg['semantic']['source_max_age_ms']: reasons.append('semantic_stale')
        audio = self.belief.audio
        if audio is None or now-audio[1]>self.cfg['interaction']['audio_max_age_ms']:
            reasons.append('audio_stale')
        exact = bool(not acoustic_only and unit and self.semantic and unit.text==self.semantic['text'])
        self.belief.complete_boundary = bool(exact and unit.stable==unit.text and unit.text.rstrip().endswith(('.', '!', '?', '。','！','？')))
        self.belief.silence_boundary = bool(exact and unit.final and unit.final_reason=='silence')
        self.reasons = reasons
        self.last_utilities = {}
        if reasons:
            return None
        joint = self.belief.joint_at(now)
        specific = exact and unit.reliability>=self.cfg['interaction']['specific_minimum_reliability']
        function,utilities = self.policy.choose(joint,self.history,now,unit.utterance_id,specific)
        self.last_utilities = utilities
        if function=='none':
            self.reasons = ['utility_prefers_none']
            return None
        summary = self.belief.summary(now)
        degraded = self.prominence is None or now-self.prominence[1]>self.cfg['prominence']['max_audio_age_ms']
        strength = self.cfg['prominence']['default_strength'] if degraded else self.prominence[0]
        action,intensity,expression = self.policy.embody(function,summary,strength,self.neutral_expression)
        if expression not in self.expressions:
            expression='neutral'
        command = self.motor.start(action,intensity,now,unit.utterance_id)
        command['expression'] = expression
        evidence = unit.to_dict() if acoustic_only else self.semantic
        self.history.append({'unit_id':unit.utterance_id,'utterance_id':self.transcripts.current.utterance_id,
            'start':self.units.start,'evidence_text':evidence['text'],
            'text_sha256':hashlib.sha256(evidence['text'].encode()).hexdigest(),
            'semantic_seq':self.last_seq,'action_id':command['action_id'],'function':function,
            'at_ms':now,'status':'dispatched','invalidated':False})
        self.reasons = ['action_selected']
        return {'kind':'command','command':command,'function':function,'belief':self.belief.at(now),
            'mass':self.belief.at(now),'utilities':utilities,'interaction':summary,
            'unit_id':unit.utterance_id,'semantic_seq':self.last_seq,'trigger':'joint_state',
            'evidence_source_ms':evidence['source_ms'],'prominence_degraded':degraded}
