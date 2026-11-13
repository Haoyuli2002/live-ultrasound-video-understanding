"""Strict scoring on the reviewed endpoint schedule, supporting fractional seconds."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def score(samples,predictions):
 known={s['sample_id']:s for s in samples}
 if len(known)!=len(samples):raise ValueError('Duplicate sample IDs')
 predicted={}
 for r in predictions:
  if r.get('sample_id') not in known or r['sample_id'] in predicted:raise ValueError('Unknown or duplicate prediction ID')
  if r.get('status') not in {'WAIT','ANSWER'}:raise ValueError('Invalid status')
  if r['status']=='ANSWER' and not isinstance(r.get('prediction'),str):raise ValueError('ANSWER requires prediction text')
  predicted[r['sample_id']]=r
 if set(predicted)!=set(known):raise ValueError('Missing predictions for reviewed endpoints')
 confusion=Counter();bytask=defaultdict(list);outcomes=Counter()
 for sid,s in known.items():
  pred=predicted[sid];gold=bool(s['meta']['y']);positive=pred['status']=='ANSWER'
  confusion['tp' if gold and positive else 'fn' if gold else 'fp' if positive else 'tn']+=1
  bytask[s['task_id']].append((s,pred))
 details=[]
 for tid,items in bytask.items():
  items.sort(key=lambda pair:pair[0]['video_window'][1]);a=items[0][0]['meta']['answer_time']
  first=next((s['video_window'][1] for s,p in items if p['status']=='ANSWER'),None)
  outcome='miss' if first is None else 'early' if first<a else 'on_time_at_reviewed_checkpoint' if first==a else 'late'
  outcomes[outcome]+=1;details.append({'task_id':tid,'first_observed_answer_time':first,'gold_answer_time':a,'outcome':outcome})
 n=len(samples)
 return {'decision':dict(confusion),'accuracy':(confusion['tp']+confusion['tn'])/n if n else None,
  'task_outcomes':dict(outcomes),'tasks':details,
  'scope':'Reviewed endpoints only; no claim about unchecked intermediate seconds, exact source-frame onset, clinical correctness, or repeated notifications. This is not dense streaming first-trigger evaluation.'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',type=Path,required=True)
 p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();r=score(rows(a.samples),rows(a.predictions))
 if a.output.exists():raise FileExistsError('Use a new evaluation output filename')
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='tasks'},indent=2))
