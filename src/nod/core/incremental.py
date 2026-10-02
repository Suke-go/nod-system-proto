"""Finite Bayesian listener: joint observations -> state belief -> feedback ledger."""
from collections import deque
import hashlib
from nod.asr.incremental import IncrementalTranscriptTracker
from nod.belief.listener import ListenerBelief
from nod.config import probability
from nod.core.events import source_time
from nod.policy.incremental import IncrementalPolicy
from nod.semantic.evidence import EvidenceMemory, words
from nod.policy.motor import Motor
from nod.semantic.sensor import SCHEMA, observation, compatibility
from nod.semantic.units import UnitTracker
from nod.timing.opportunity import OpportunityGate


class IncrementalListenerEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transcripts = IncrementalTranscriptTracker(cfg)
        self.units = UnitTracker(cfg['semantic']['max_context_characters'])
        self.belief = ListenerBelief(cfg['listener'])
        self.policy = IncrementalPolicy(cfg['listener']['policy'])
        if cfg['listener']['policy'].get('decision_model'):
            from nod.policy.risk import BayesRiskPolicy
            self.policy=BayesRiskPolicy(cfg['listener']['policy'])
        self.memory = EvidenceMemory(cfg['listener'])
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
        self.asr_boundary = None

    def semantic_input(self):
        return self.units.current, self.units.context

    def process(self, event):
        if event.at_ms < self.now:
            raise ValueError('Ingest time moved backwards')
        self.now = now = event.at_ms
        data = event.data
        if event.kind=='asr_job' and self.cfg['asr'].get('preserve_final_jobs'):
            # Lifecycle/archived text is telemetry only, never new evidence.
            return []
        if event.kind=='presentation' and self.cfg['listener']['version']==4:
            entry=next((h for h in self.history if h['action_id']==data.get('action_id')),None)
            if entry is None or entry.get('presentation_received_ms') is not None:return []
            entry['presentation_received_ms']=now
            return [{'kind':'presentation_observed','action_id':entry['action_id'],
                'dispatch_to_receipt_ms':now-entry['at_ms'],
                'measurement':'browser_animation_frame_callback_plus_return_transport; not photon onset'}]
        records = []
        old_motor = self.motor.state
        self.motor.tick(now)
        if event.kind == 'asr':
            source = source_time(data,now)
            old = self.transcripts.current
            if old is None or old.utterance_id!=data['utterance_id']:
                self.asr_boundary=None
            snapshot = self.transcripts.update(data['utterance_id'],data['text'],source,now,
                data.get('confidence'),data.get('final',False),data.get('final_reason'))
            repaired = bool(old and old.utterance_id==snapshot.utterance_id and
                            old.repair_epoch!=snapshot.repair_epoch)
            unit,new,local_repair = self.units.update(snapshot,self.transcripts.recent_context())
            if self.cfg['listener'].get('commit_scope')=='asr_utterance':
                if old is None or old.utterance_id!=snapshot.utterance_id:
                    self.belief.open_unit(now)
                    self.semantic=None
                elif new or local_repair:
                    # All hypotheses within one ASR utterance share an anchor.
                    # A changing punctuation boundary never commits overlapping
                    # evidence into that anchor. Revisions can revoke it entirely.
                    self.belief.revoke()
                    self.semantic=None
            elif new:
                self.belief.open_unit(now,revoke=repaired)
                self.semantic = None
            elif local_repair:
                self.belief.revoke()
                self.semantic = None
            revoked = self.memory.reconcile(snapshot, now)
            if revoked:
                records.append({'kind':'interpretations_revoked','evidence_ids':revoked})
            if repaired:
                for entry in self.history:
                    if entry['utterance_id'] == snapshot.utterance_id and not entry['invalidated']:
                        start = entry['start']
                        if not words(snapshot.text).startswith(entry['evidence_prefix']):
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
        elif event.kind == 'asr_boundary' and self.cfg['asr'].get('preserve_final_jobs'):
            source_time(data,now)
            current=self.transcripts.current
            if current and current.utterance_id==data['utterance_id']:
                self.asr_boundary=dict(data)
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
                'model':data['result'].get('model'), 'state':self.observation_belief.summary(now),
                'observed_snapshot':data['request']['snapshot'],
                'evidence_rule':'replace_scoped_packet_preserve_valid_prefix',
                'retained':self.memory.describe(now)}
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
                'feedback_count':len(self.history),'retained_interpretations':self.memory.describe(now)})
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
        if not words(unit.text).startswith(words(snapshot['text'])):
            return 'repaired_input'
        source = source_time(snapshot,now)
        if (request['dispatched_ms']>now or now-request['dispatched_ms']>self.cfg['semantic']['total_deadline_ms']
                or now-source>self.cfg['semantic']['source_max_age_ms']):
            return 'expired'
        try:
            if result.get('schema')!=SCHEMA:
                return 'incompatible_semantic_schema'
            values = observation(result['observation'])
            calibration = self.cfg['listener'].get('scalar_calibration')
            if calibration:
                if result.get('model')!=calibration['model']:
                    return 'calibration_model_mismatch'
                from nod.semantic.sensor import calibrate_scalars
                values = calibrate_scalars(values, calibration)
        except (ValueError,KeyError,TypeError):
            return 'invalid_semantic_observation'
        self.last_seq = seq
        self.semantic = {**snapshot, 'observation':values, 'probabilities':compatibility(values)}
        self.observation_belief=ListenerBelief(self.cfg['listener'])
        self.observation_belief.anchor=self.belief.predicted(source)
        self.observation_belief.anchor_ms=source
        self.observation_belief.observe(snapshot['reliability'],values,source)
        self.memory.add(snapshot, values, self.transcripts.current, self.units.start,
            self.belief.predicted(source), now, seq)
        return 'accepted'

    def _decide(self, now):
        unit = self.units.current
        acoustic_only = self.cfg['listener']['condition']=='acoustic_only'
        self.decision_context = {}; self.last_utilities = {}
        exact = bool(unit and self.semantic and words(unit.text)==words(self.semantic['text']))
        fresh = bool(self.semantic and now-self.semantic['source_ms']<=self.cfg['semantic']['source_max_age_ms'])
        obs = self.semantic['observation'] if exact and fresh and not acoustic_only else None
        if unit:
            stamp = min(unit.source_ms,self.semantic['source_ms']) if obs else unit.source_ms
            self.belief.observe(unit.reliability,obs,stamp)
        retained = self.memory.candidates(self.transcripts.current,now) if self.transcripts.current else []
        reasons=[]
        # Upgrade a *completed* continuer when a newer semantic result arrives.
        # Never interrupt an active command, ignore a fault, or bypass the budget.
        upgrade=bool(self.cfg['listener']['version']==4 and self.motor.state=='COOLDOWN'
            and self.history and self.history[-1]['function'] in ('continuer','empathic')
            and self.history[-1]['status']=='completed'
            and now-self.motor.completed_ms>=self.cfg['listener']['policy']['expression_upgrade_gap_ms']
            and self.last_seq>(self.history[-1].get('semantic_seq') or 0))
        if not self.connected: reasons.append('controller_unavailable')
        if not self.input_healthy: reasons.append('input_unhealthy')
        if self.motor.state!='IDLE' and not upgrade: reasons.append('motor_'+self.motor.state.lower())
        if not unit or not unit.text: reasons.append('asr_missing')
        elif now-unit.source_ms>self.cfg['semantic']['source_max_age_ms']: reasons.append('asr_stale')
        elif unit.reliability<self.cfg['policy']['minimum_asr_reliability'] and not retained: reasons.append('asr_unreliable')
        audio=self.belief.audio
        if audio is None or now-audio[1]>self.cfg['listener']['audio_max_age_ms']: reasons.append('audio_stale')
        self.reasons=reasons
        if reasons:return None
        options=[]
        current_id=next((x['id'] for x in retained if words(x['text'])==words(unit.text) and x['unit_id']==unit.utterance_id),None)
        if unit.reliability>=self.cfg['policy']['minimum_asr_reliability']:
            options.append({'belief':self.belief,'observation':obs,'unit_id':unit.utterance_id,
                'retained':False,'id':current_id,'source_ms':self.semantic['source_ms'] if obs else unit.source_ms,
                'text':unit.text,'start':self.units.start,'semantic_seq':self.last_seq if obs else None,
                'reliability':unit.reliability})
        for item in retained:
            # Current exact observation is evaluated once, not twice with two priors.
            if item['id']==current_id and obs is not None:continue
            options.append({**item,'retained':True,'reliability':item['belief'].packet[0]})
        choices=[]
        for option in options:
            q=option['observation'];summary=option['belief'].summary(now)
            if q and self.cfg['listener']['condition']=='argmax':
                q={**q,'completion':float(q['completion']>=.5),'response_demand':float(q['response_demand']>=.5)}
            risk_policy=bool(self.cfg['listener']['policy'].get('decision_model'))
            extra={'supported_expressions':self.expressions} if risk_policy else {}
            f,u,c=self.policy.choose(summary['posterior'],self.history,now,option['unit_id'],q,audio[0],
                bool((unit.final and unit.final_reason=='silence') or
                     (self.asr_boundary and self.asr_boundary['final_reason']=='silence'
                      and now-self.asr_boundary['source_ms']<=self.cfg['semantic']['source_max_age_ms'])),
                bool(q and (risk_policy or option['reliability']>=self.cfg['listener']['specific_minimum_reliability'])),
                retained=option['retained'],evidence_id=option['id'],expressive=self.neutral_expression,
                allow_expression=bool(q and option['reliability']>=self.cfg['listener']['incremental']['retain_reliability']),
                evidence_text=option['text'],expression_only=upgrade,evidence_age_ms=now-option['source_ms'],
                receipt_upgrade=bool(upgrade and (self.cfg['listener']['policy'].get('local_receipt') or risk_policy)
                                     and self.history[-1]['function']=='continuer'),**extra)
            choices.append((u[f]-u['none'],f,u,c,option,summary))
        if not choices:
            self.reasons=['asr_unreliable'];return None
        _,function,self.last_utilities,self.decision_context,selected,summary=max(choices,key=lambda x:x[0])
        if function=='none':
            self.reasons=[self.decision_context['reason']];return None
        degraded=self.prominence is None or now-self.prominence[1]>self.cfg['prominence']['max_audio_age_ms']
        strength=self.cfg['prominence']['default_strength'] if degraded else self.prominence[0]
        if self.cfg['listener']['policy'].get('decision_model'):
            action,intensity,expression=self.policy.realize(function,summary,strength,self.neutral_expression,self.decision_context)
        else:
            action,intensity,expression=self.policy.embody(function,summary,strength,self.neutral_expression)
        if expression not in self.expressions: expression='neutral'
        if upgrade:self.motor.state='IDLE'
        command=self.motor.start(action,intensity,now,selected['unit_id']);command['expression']=expression
        self.history.append({'unit_id':selected['unit_id'],'utterance_id':self.transcripts.current.utterance_id,
            'start':selected['start'],'evidence_text':selected['text'],
            'evidence_prefix':selected.get('prefix',words(self.transcripts.current.text[:selected['start']]+selected['text'])),
            'text_sha256':hashlib.sha256(selected['text'].encode()).hexdigest(),
            'semantic_seq':selected['semantic_seq'],'evidence_id':selected['id'],
            'action_id':command['action_id'],'function':function,'at_ms':now,'status':'dispatched','invalidated':False})
        if self.cfg['listener']['version']==4:self.history[-1]['expression']=expression
        self.reasons=['action_selected']
        return {'kind':'command','command':command,'function':function,'belief':selected['belief'].at(now),
            'mass':selected['belief'].at(now),'utilities':self.last_utilities,'listener':summary,
            'unit_id':selected['unit_id'],'semantic_seq':selected['semantic_seq'],
            'trigger':'listener_belief','decision_context':self.decision_context,
            'evidence_source_ms':selected['source_ms'],'evidence_text':selected['text'],
            'evidence_start':selected['start'],'evidence_id':selected['id'],
            'retained_evidence':selected['retained'],'prominence_degraded':degraded}
