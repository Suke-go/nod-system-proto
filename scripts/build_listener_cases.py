from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
cases={
 'ja':[
  ('音声を文字に変換し、その文字から話の意味を推定します。',{'interpretability':'resolved','appraisal':'neutral','completion':True,'response_demand':False}),
  ('何度も練習した発表が、ようやくうまくできました。本当に嬉しいです。',{'interpretability':'resolved','appraisal':'positive','completion':True,'response_demand':False}),
  ('一生懸命に準備した計画が、中止になってしまいました。とても残念です。',{'interpretability':'resolved','appraisal':'negative','completion':True,'response_demand':False}),
  ('その方法は使えそうですが、',{'completion':False}),
  ('この方法を採用することに賛成ですか。',{'response_demand':True}),
  ('数式中の正の値を、正の値同士で足します。',{'appraisal':'neutral'})],
 'en':[
  ('We convert speech into text and estimate its meaning from that text.',{'interpretability':'resolved','appraisal':'neutral','completion':True,'response_demand':False}),
  ('After practicing so many times, my presentation finally went well. I am really happy.',{'interpretability':'resolved','appraisal':'positive','completion':True,'response_demand':False}),
  ('The project I worked so hard on was cancelled. I am really disappointed.',{'interpretability':'resolved','appraisal':'negative','completion':True,'response_demand':False}),
  ('The method looks promising, but',{'completion':False}),
  ('Do you agree that we should adopt this method?',{'response_demand':True}),
  ('The equation adds two positive numbers.',{'appraisal':'neutral'})]}
rows=[{'id':f'{lang}-{i+1}','language':lang,'state':{'stable_transcript':text,'current_partial':'','recent_context':[]},
       'expected':expected,'label_origin':'authored_diagnostic_not_human_validation'}
      for lang,items in cases.items() for i,(text,expected) in enumerate(items)]
(root/'experiments/pilot/listener-cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
