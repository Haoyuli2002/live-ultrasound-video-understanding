"""Validate release files and exercise the actual project's loader without GPU imports."""
import argparse, ast, hashlib, json, math
from collections import Counter
from pathlib import Path
import tempfile

def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_contract(project):
 # Execute the real function bodies, not a copied/reimplemented schema validator.
 # torch/model imports and the training entry point are intentionally not executed.
 scope={'Path':Path,'math':math,'json':json}
 names={project/'stage3/model.py':['answerability_label'],
        project/'stage3/train.py':['load_examples','decision_question']}
 for path,wanted in names.items():
  tree=ast.parse(path.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
  if {n.name for n in functions}!=set(wanted):raise ValueError(f'Unknown trainer contract: {path}')
  scope.update({'WAIT':'<WAIT>','ANSWER':'<ANSWER>'})
  exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),scope)
 return scope

def validate(package,project,video_map=None):
 scope=load_contract(project);manifest=json.loads((package/'dataset_manifest.json').read_text())
 paths=json.loads((video_map or package/'video_path_map.local.json').read_text())
 assets=json.loads((package/'media_manifest.json').read_text())
 assert set(paths)==set(assets), 'Video map must cover exactly the release video IDs'
 if any(not Path(p).is_absolute() or not Path(p).is_file() for p in paths.values()):raise ValueError('Missing local source video')
 counts={};all_task_ids=set();split_vids={s:set() for s in ['train','eval']}
 for group,summary in manifest['groups'].items():
  count=0
  for split in ['train','eval']:
   source=package/'data'/group/f'{split}.jsonl';examples=rows(source)
   tasks={t['task_id']:t for t in rows(package/'data'/group/f'{split}_tasks.jsonl')}
   loaded=scope['load_examples'](source) if examples else {};index={}
   for v,points in loaded.items():
    assert [r['time'] for r in points]==sorted(r['time'] for r in points)
    for r in points:
     assert set(r)=={'video_id','question','task_id','target','time','query_time','y'}
     assert 'answer_time' not in scope['decision_question'](r)
     index[(r['task_id'],r['time'])]=r
   assert len(index)==len(examples)
   expected={(t['task_id'],time) for t in tasks.values() for time in t['evaluation_times']}
   assert set(index)==expected
   for sample in examples:
    t=tasks[sample['task_id']];current=sample['video_window'][1]
    assert sample['video_id']==t['video_id'] and sample['question']==t['question']
    assert sample['meta']['split']==split and assets[t['video_id']]['split']==split
    assert current<=assets[t['video_id']]['duration']+.001
    assert sample['target']==('<ANSWER> '+t['reference_answer'] if current>=t['answer_time'] else '<WAIT>')
    assert sample['meta']['y']==int(current>=t['answer_time'])
   count+=len(examples)
   if group=='all':
    split_vids[split]|={t['video_id'] for t in tasks.values()}
    assert not (all_task_ids&tasks.keys());all_task_ids.update(tasks)
  assert count==summary['decision_samples'];counts[group]=count
 assert not (split_vids['train']&split_vids['eval'])
 assert len(paths)==manifest['high_medium_videos']==223
 # Real loader rejects a premature ANSWER and a clock/window mismatch.
 bad={'video_id':'v','task_id':'bad','question':'What appears?', 'video_window':[0,3.25],
      'target':'<ANSWER> premature', 'meta':{'query_time':3.25,'answer_time':4.5,'current_time':3.25,'qa_type':'event_notification'}}
 with tempfile.TemporaryDirectory() as d:
  f=Path(d)/'bad.jsonl'
  for row in [bad,dict(bad,target='<WAIT>',meta=dict(bad['meta'],current_time=3.5))]:
   f.write_text(json.dumps(row)+'\n')
   try:scope['load_examples'](f)
   except ValueError:pass
   else:raise AssertionError('Actual trainer accepted an invalid causal sample')
 if (package/'checksums.json').exists():
  for name,digest in json.loads((package/'checksums.json').read_text()).items():
   assert sha(package/name)==digest, f'Changed release file: {name}'
 return {'status':'passed','groups_samples_loaded_by_actual_loader':counts,
  'source_loader':str(project/'stage3/train.py'),'source_loader_sha256':sha(project/'stage3/train.py'),
  'execution_scope':'Actual AST-extracted load_examples, decision_question and answerability_label functions. No full torch import, GPU execution, model/checkpoint load or clinical re-review.',
  'train_videos':len(split_vids['train']),'eval_videos':len(split_vids['eval']),
  'train_eval_video_overlap':0,'native_media_paths_checked':len(paths),
  'negative_contract_checks':['reject premature ANSWER','reject current_time/window mismatch']}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1])
 p.add_argument('--project-root',type=Path,required=True)
 p.add_argument('--video-path-map',type=Path)
 a=p.parse_args();report=validate(a.package,a.project_root,a.video_path_map)
 # Validation output is deliberately outside the checksum set.
 (a.package/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
