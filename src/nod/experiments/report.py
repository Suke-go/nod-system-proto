from collections import Counter
from pathlib import Path
from nod.config import strict_json
from nod.experiments.common import quantiles


def analyze_session(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        header = strict_json(next(stream))
        commands, accepted, started, completed = {}, {}, {}, {}
        dispositions, errors, actions, suppressed = Counter(), Counter(), Counter(), Counter()
        semantic_latencies, input_ages = [], []
        events, degraded = 0, 0
        for line in stream:
            row = strict_json(line); event = row['event']; events += 1
            if row['ingest_seq'] != events:
                raise ValueError('Broken event sequence')
            data, now = event['data'],event['at_ms']
            if event['kind'] == 'semantic_result':
                semantic_latencies.append(now-data['request']['dispatched_ms'])
            if event['kind'] == 'semantic_error': errors[data['reason']] += 1
            if event['kind'] in ('asr','opportunity'):
                input_ages.append(now-data['source_ms'])
            if event['kind'] == 'feedback':
                destination = {'accepted':accepted,'started':started,'completed':completed}.get(data['status'])
                if destination is not None: destination.setdefault(data['action_id'],now)
            for derived in row['derived']:
                if derived['kind'] == 'decision_status': suppressed.update(derived['reasons'])
                if derived['kind'] == 'semantic_disposition': dispositions[derived['reason']] += 1
                if derived['kind'] == 'command':
                    command = derived['command']; commands[command['action_id']] = command
                    actions[derived['function']] += 1; degraded += derived['prominence_degraded']
        def delays(notifications):
            return quantiles([t-commands[key]['at_ms'] for key,t in notifications.items() if key in commands])
        return {'events':events,'backend':header['backend'],'experiment':header['config'].get('experiment'),
                'commands':len(commands),'functions':dict(actions),'prominence_degraded_commands':degraded,
                'completed_actions':len(set(commands)&set(completed)),
                'actions_without_completion':sorted(set(commands)-set(completed)),
                'semantic_dispositions':dict(dispositions),'semantic_errors':dict(errors),
                'suppression_reason_event_counts':dict(suppressed),
                'semantic_success_return_ms':quantiles(semantic_latencies),'input_observation_age_ms':quantiles(input_ages),
                'command_to_accepted_ms':delays(accepted),'command_to_started_ms':delays(started),
                'measures_real_animation_onset':False,'measures_end_to_end_live_audio_latency':False}
