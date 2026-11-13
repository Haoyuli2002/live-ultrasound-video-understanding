"""Streaming first-response metrics. Lexical scores are not clinical correctness."""
from collections import defaultdict
from stage2.reconstruct import text_metrics


def score(tasks, predictions):
    groups = defaultdict(list)
    known = {t['task_id']: t for t in tasks}
    seen = set()
    for row in predictions:
        key = (row['task_id'], row['time'])
        if key in seen or row['task_id'] not in known:
            raise ValueError('Duplicate decision or unknown task')
        seen.add(key); groups[row['task_id']].append(row)
    tp = tn = fp = fn = 0; outcomes = []; lexical = []
    for tid, task in known.items():
        rows = sorted(groups[tid], key=lambda r:r['time'])
        expected = list(range(int(task['query_time']), int(task['evaluation_horizon'])+1))
        if [r['time'] for r in rows] != expected:
            raise ValueError(f'Missing/extra decision points for {tid}')
        a = task['answer_time']; first = None
        for row in rows:
            gold = a is not None and row['time'] >= a
            pred = row['status'] == 'ANSWER'
            tp += int(gold and pred); tn += int(not gold and not pred)
            fp += int(not gold and pred); fn += int(gold and not pred)
            if pred and first is None: first = row
        if a is None:
            outcome = 'correct_wait' if first is None else 'false_trigger'
        elif first is None: outcome = 'miss'
        elif first['time'] < a: outcome = 'early'
        elif first['time'] == a: outcome = 'on_time'
        else: outcome = 'late'
        entry = {'task_id':tid, 'outcome':outcome,
                 'first_answer_time':None if first is None else first['time'],
                 'gold_answer_time':a,
                 'latency':None if first is None or a is None else first['time']-a}
        if first is not None and a is not None:
            entry['text'] = text_metrics(first['prediction'], task['reference_answer'])
            # Score all answerable tasks end-to-end: missing/early outputs receive zero credit below.
        if a is not None:
            lexical.append(entry.get('text',{}).get('word_overlap_f1',0.) if outcome in {'on_time','late'} else 0.)
        outcomes.append(entry)
    n = tp+tn+fp+fn
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    counts = {name:sum(o['outcome']==name for o in outcomes)
              for name in ['early','on_time','late','miss','correct_wait','false_trigger']}
    return {'decision':{'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':(tp+tn)/n if n else 0,
                       'precision':precision,'recall':recall,
                       'f1':2*precision*recall/(precision+recall) if precision+recall else 0.},
            'task_outcomes':counts,'tasks':outcomes,
            'e2e_word_f1_zero_for_early_or_miss':sum(lexical)/len(lexical) if lexical else None,
            'interpretation':'Same-video overfit diagnostic, not held-out performance. Text overlap is not clinical correctness. Only the first predicted answer is delivered; later raw decisions are counterfactual diagnostic probes.'}
