import asyncio
from copy import deepcopy
import json
from pathlib import Path
import pytest
from nod.config import load_config
from nod.production import configure
from nod.runtime import Runtime
from nod.semantic.frames import SCHEMA,fixture_frame,function_distribution
from nod.telemetry import replay
from nod.experiments.interaction import review,score


def test_typed_state_to_real_websocket_ack_and_research_export(tmp_path):
    class Backend:
        async def evaluate(self,state):
            name='negative_experience_complete' if 'lost' in state['stable_transcript'] else 'explanation_complete'
            p=fixture_frame(name)
            return {'schema':SCHEMA,'frames':p,'probabilities':function_distribution(p),'model':'fixture','confidence':.98}
        async def close(self):pass
    async def run():
        c=configure(load_config());c['output']['port']=0;c['motor']['cooldown_after_completion_ms']=50
        r=Runtime(c,'mock',tmp_path/'session.jsonl');r.semantic.backend=Backend()
        async def feed():
            for uid,text in [('u1','Here is the explanation.'),('u2','I lost my job and feel terrible.')]:
                r.emit('asr',{'utterance_id':uid,'text':text,'final':True,'final_reason':'silence',
                              'confidence':.99,'source_ms':r.clock.now_ms()})
                for _ in range(15):
                    r.emit('opportunity',{'score':.1,'source_ms':r.clock.now_ms()})
                    await asyncio.sleep(.1)
        summary=await r.run(feed)
        return summary,r.commands
    result,actions=asyncio.run(run())
    assert result['functions']==['understanding','empathic']
    assert actions[-1]['command']['expression']=='concerned'
    assert result['motor_state']!='FAULT' and replay(tmp_path/'session.jsonl')['matched']
    exported=review(tmp_path/'session.jsonl',tmp_path/'review')
    assert exported['annotation_items']==2
    annotations=tmp_path/'review/annotations.jsonl';predictions=tmp_path/'review/predictions.jsonl'
    labels=[json.loads(x) for x in annotations.read_text(encoding='utf-8').splitlines()]
    assert all(r['label'] is None and 'frames' not in r for r in labels)
    assert score(annotations,predictions)['metrics'] is None
    labels[0].update(label='explanation_complete',reviewed=True,annotator='test-reviewer')
    annotations.write_text('\n'.join(json.dumps(x) for x in labels),encoding='utf-8')
    assert score(annotations,predictions)['metrics']['accuracy']==1


def test_temperature_calibration_never_uses_unreviewed_or_test_split(tmp_path):
    a=tmp_path/'a.jsonl';p=tmp_path/'p.jsonl'
    labels=[{'id':str(i),'label':'explanation_complete','reviewed':True,'annotator':'fixture',
             'split':'calibration'} for i in range(20)]
    preds=[{'id':str(i),'frames':fixture_frame('explanation_complete',.7)} for i in range(20)]
    a.write_text('\n'.join(json.dumps(r) for r in labels),encoding='utf-8')
    p.write_text('\n'.join(json.dumps(r) for r in preds),encoding='utf-8')
    result=score(a,p,True)
    assert result['temperature']<1 and result['semantic_schema']==SCHEMA
    labels[0]['split']='test'
    a.write_text('\n'.join(json.dumps(r) for r in labels),encoding='utf-8')
    with pytest.raises(ValueError,match='calibration split'):score(a,p,True)
