import asyncio
import csv
import json
import math
import time
from importlib.resources import files
from pathlib import Path
from nod.config import CLASSES, distribution, strict_json
from nod.experiments.common import quantiles, save_json, sha256
from nod.semantic.backend import JevBackend, MockBackend, SemanticError


def load_cases(path):
    cases = []
    seen = set()
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            case = strict_json(line)
            if not isinstance(case.get('id'), str) or not case['id'] or case['id'] in seen:
                raise ValueError('Each evaluation case needs a unique nonempty id')
            seen.add(case['id'])
            state = case['state']
            if set(state) != {'stable_transcript','current_partial','recent_context'}:
                raise ValueError('Invalid semantic state fields')
            if not all(isinstance(state[k], str) for k in ('stable_transcript','current_partial')) or len(state['stable_transcript']+state['current_partial']) > 2000:
                raise ValueError('Invalid or oversized transcript')
            if not isinstance(state['recent_context'], list) or any(not isinstance(x,dict) or set(x) != {'speaker','text'} or x['speaker'] != 'user' or not isinstance(x['text'],str) for x in state['recent_context']):
                raise ValueError('Invalid recent context')
            if sum(len(x['text']) for x in state['recent_context']) > 2000:
                raise ValueError('Context exceeds the runtime budget')
            if case.get('reference_label') not in CLASSES or case.get('label_status') not in ('draft','reviewed'):
                raise ValueError('Cases require a reference label and draft/reviewed status')
            cases.append(case)
    if not cases:
        raise ValueError('Evaluation file is empty')
    return cases


def summarize(rows):
    successful = [r for r in rows if r['status'] == 'ok']
    reviewed = [r for r in successful if r['label_status'] == 'reviewed']
    confusion = {ref:{pred:0 for pred in CLASSES} for ref in CLASSES}
    brier, nll, correct = [], [], 0
    for row in reviewed:
        reference = row['reference_label']; p = row['probabilities']
        confusion[reference][row['prediction']] += 1
        correct += reference == row['prediction']
        brier.append(sum((p[i] - int(key == reference))**2 for i,key in enumerate(CLASSES)))
        nll.append(-math.log(max(1e-12, p[CLASSES.index(reference)])))
    return {'attempted':len(rows), 'succeeded':len(successful), 'failed':len(rows)-len(successful),
            'latency_ms_all_attempts':quantiles([r['latency_ms'] for r in rows]),
            'latency_ms_successful':quantiles([r['latency_ms'] for r in successful]),
            'reviewed_label_count':len(reviewed), 'draft_labels_excluded':len(successful)-len(reviewed),
            'accuracy':correct/len(reviewed) if reviewed else None,
            'brier_multiclass_sum':sum(brier)/len(brier) if brier else None,
            'negative_log_likelihood':sum(nll)/len(nll) if nll else None,
            'confusion_reference_rows_prediction_columns':confusion}


async def evaluate_cases(cfg, cases_path, output_dir, semantic='mock', limit=10, backend=None):
    cases = load_cases(cases_path)
    if not 1 <= limit <= 1000:
        raise ValueError('limit must be between 1 and 1000')
    model = backend or (JevBackend(cfg['semantic']) if semantic == 'jev' else MockBackend())
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except BaseException:
        await model.close()
        raise
    rows = []
    question = files('nod').joinpath('resources/jev-request.json').read_bytes()
    import hashlib
    save_json(output/'run.json', {'schema_version':1, 'semantic':semantic, 'limit':limit,
        'dataset_sha256':sha256(cases_path), 'question_sha256':hashlib.sha256(question).hexdigest(),
        'config':cfg, 'source':'independent_text_cases', 'mock_is_not_model_quality':semantic == 'mock'})
    try:
        with (output/'results.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
            for case in cases[:limit]:
                begin = time.perf_counter()
                row = {'id':case['id'],'reference_label':case['reference_label'],
                       'label_status':case['label_status'],'state':case['state']}
                try:
                    result = await model.evaluate(case['state'])
                    p = distribution(result['probabilities'])
                    row.update(status='ok', probabilities=p, prediction=CLASSES[max(range(4),key=lambda i:p[i])],
                               model=result['model'], provider_confidence=result.get('confidence'))
                except SemanticError as exc:
                    row.update(status='error',reason=exc.reason)
                row['latency_ms'] = (time.perf_counter()-begin)*1000
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                rows.append(row)
                # The first failure stops this experiment. No retries or extra billed calls.
                if row['status'] != 'ok':
                    break
                await asyncio.sleep(max(0, cfg['semantic']['min_dispatch_interval_ms']/1000-(time.perf_counter()-begin)))
    finally:
        await model.close()
    report = summarize(rows)
    report.update(semantic=semantic, model_quality_measured=semantic != 'mock' and report['reviewed_label_count'] > 0,
                  requested=min(limit,len(cases)), not_attempted=min(limit,len(cases))-len(rows))
    save_json(output/'summary.json',report)
    with (output/'review.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id','reference_label','label_status','status','prediction','latency_ms',*CLASSES])
        for row in rows:
            writer.writerow([row['id'],row['reference_label'],row['label_status'],row['status'],
                             row.get('prediction',''),row['latency_ms'],*row.get('probabilities',['']*4)])
    return report
