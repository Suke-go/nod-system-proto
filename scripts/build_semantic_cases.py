"""Author-created diagnostics; expected labels are not independent human validation."""
import json
from pathlib import Path

examples=[
('ja','水は加熱すると蒸発します。','explanation_complete'),
('ja','水を加熱すると、その結果は','explanation_open'),
('ja','この設計は前の案より使いやすいと思います。','evaluation_complete'),
('ja','この設計について私が思うのは','evaluation_open'),
('ja','何度も落ちた試験にようやく合格できて、本当に嬉しいです。','positive_experience_complete'),
('ja','大切にしていた仕事を失って、とてもつらいです。','negative_experience_complete'),
('ja','昇進できて嬉しい反面、仲間と離れるのが寂しいです。','mixed_experience_complete'),
('ja','最初は成功したと思ったのですが、実は','repair_open'),
('ja','さっき成功したと言いましたが、訂正します。実際には失敗でした。','repair_complete'),
('ja','あなたはこの結果をどう思いますか？','question_complete'),
('en','Heating the water makes it evaporate.','explanation_complete'),
('en','When we heat the water, what happens next is','explanation_open'),
('en','I think this design is easier to use than the previous one.','evaluation_complete'),
('en','What I think about this design is','evaluation_open'),
('en','After failing the exam three times, I finally passed and I am so happy.','positive_experience_complete'),
('en','I lost the job I loved and I am devastated.','negative_experience_complete'),
('en','I am happy about my promotion but sad to leave my friends.','mixed_experience_complete'),
('en','I said it succeeded, but actually','repair_open'),
('en','I need to correct what I said: it failed, not succeeded.','repair_complete'),
('en','What do you think of this result?','question_complete'),
]
path=Path(__file__).resolve().parents[1]/'experiments/pilot/semantic-cases.jsonl'
with path.open('w',encoding='utf-8') as stream:
    for i,(language,text,label) in enumerate(examples):
        stream.write(json.dumps({'id':f'{language}-{i+1:02d}','language':language,
            'state':{'stable_transcript':text,'current_partial':'','recent_context':[]},
            'label':label,'label_origin':'author_created_development_fixture'},ensure_ascii=False)+'\n')
