"""Build a new, non-destructive AV annotation package for the current Stage3 trainer."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

SOURCES = [
 'stage3_av50_20261105', 'stage3_av_next50_20261106',
 'stage3_av_visualfirst_next50_20261110',
 'stage3_av_visualfirst_regen14_reviewed_20261111',
 'stage3_remaining79/export_accepted79_20261111',
]
AUDIO_DEPENDENT = {'audio', 'audio_video', 'audio_video_joint'}
IMMEDIATE = {'current_observation', 'history_recall'}
DELAYED = {'temporal_comparison', 'event_notification'}

def read_rows(path):
 return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def write_json(path, obj):
 path.parent.mkdir(parents=True, exist_ok=True)
 path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def write_rows(path, rows):
 path.parent.mkdir(parents=True, exist_ok=True)
 path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def probe(path):
 r=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
  '-show_entries','stream=width,height:format=duration','-of','json',str(path)]))
 return {'duration':float(r['format']['duration']),
         'native_resolution':[r['streams'][0]['width'],r['streams'][0]['height']]}

def asr_segments(path):
 # These are annotation artifacts, never automatically included in model prompts.
 if path is None: return []
 segments=[]
 if path.with_suffix('.json').is_file():
  parsed=json.loads(path.with_suffix('.json').read_text())
  matches=[(s['start'],s['end'],s['text']) for s in parsed['segments']]
 else:
  text=path.read_text()
  matches=re.findall(r'^\[([\d.]+)-([\d.]+)\]\s+([^\n]+)', text, re.M)
  if not matches:
   matches=re.findall(r'^\[\d+\]\s+([\d.]+)-([\d.]+)\s+([^\n]+)', text, re.M)
  if text.strip() and not matches:
   raise ValueError(f'Unrecognized ASR format: {path}')
 for lo,hi,text in matches:
  lo,hi=float(lo),float(hi)
  if not (math.isfinite(lo) and math.isfinite(hi) and 0<=lo<=hi):
   raise ValueError(f'Invalid ASR segment in {path}')
  segments.append({'start':lo,'end':hi,'text':text})
 return segments

def build(workspace, dest):
 if (dest/'checksums.json').exists():
  raise FileExistsError('Use a new output directory. Existing releases are immutable.')
 labels=json.loads((workspace/'outputs/video_value_stats_20261104.json').read_text())
 profiles={r['video_id']:r for r in labels['videos']}
 target={v for v,p in profiles.items() if p['stage_value']['stage3']['value'] in {'high','medium'}}
 rows=[];provenance=[]
 for source in SOURCES:
  for f in sorted((workspace/'outputs'/source).glob('*tasks.jsonl')):
   batch=read_rows(f)
   if not batch: raise ValueError(f'Empty source {f}')
   provenance.append({'source':str(f),'sha256':sha(f),'qa':len(batch)})
   for index,r in enumerate(batch,1): rows.append((r,source,str(f),index))
 override_file=dest/'annotations/review_overrides.jsonl'
 if override_file.exists():
  overrides={r['task_id']:r for r in read_rows(override_file)}
  assert set(overrides)<={r['task_id'] for r,*_ in rows}
  rows=[(overrides.get(r['task_id'],r),source,file,index) for r,source,file,index in rows]
  provenance.append({'source':str(override_file),'sha256':sha(override_file),'qa':len(overrides),'role':'review_corrections_overlay'})
 if len({r['task_id'] for r,*_ in rows})!=len(rows): raise ValueError('Duplicate task IDs')
 if {r['video_id'] for r,*_ in rows}!=target: raise ValueError('Coverage differs from High/Medium inventory')
 # Source versions must be disjoint by video; do not mix old visual-only versions.
 video_sources={}
 for r,source,*_ in rows:
  if video_sources.setdefault(r['video_id'],source)!=source: raise ValueError('Overlapping video versions')
 video_paths={}
 for r,*_ in rows:
  v=r['video_id'];p=Path(r['source_video'])
  if not p.is_absolute() or not p.is_file(): raise ValueError(f'Missing source video {v}')
  if video_paths.setdefault(v,str(p))!=str(p): raise ValueError('Conflicting video paths')
 with ThreadPoolExecutor(max_workers=6) as pool:
  probes=dict(zip(video_paths,pool.map(probe,video_paths.values())))
 assets={}
 for v in sorted(video_paths):
  source=video_sources[v];split=profiles[v]['split'];source_rows=[r for r,*_ in rows if r['video_id']==v]
  candidates=[workspace/'outputs'/source/split/v/'asr.txt']
  candidates += [workspace/'outputs'/n/split/v/'asr.txt' for n in SOURCES[:3]]
  asr_path=next((p for p in candidates if p.is_file()),None)
  segments=asr_segments(asr_path)
  if segments:
   write_json(dest/'annotations/asr'/f'{v}.json',{'video_id':v,'annotation_only':True,
     'full_video_asr_allowed_at_inference':False,'source':str(asr_path),'segments':segments})
  audio_path=next((r.get('source_audio') for r in source_rows if r.get('source_audio') and Path(r['source_audio']).is_file()),None)
  assets[v]={'video_id':v,'split':split,'stage3_value':profiles[v]['stage_value']['stage3']['value'],
   'video_path':video_paths[v],**probes[v],'audio_path':audio_path,
   'audio_source_fallback':'original video audio track; not extracted or validated in this packaging run' if not audio_path else None,
   'asr_source_path':str(asr_path) if asr_path else None,'asr_segments':len(segments),
   'asr_annotation_file':f'annotations/asr/{v}.json' if segments else None,
   'source_dataset':source}
 canonical=[]
 for raw,source,file,line in rows:
  v=raw['video_id'];q=float(raw['query_time']);a=float(raw['answer_time']);kind=raw['qa_type']
  if not all(map(math.isfinite,[q,a])) or not 0<q<=a<=assets[v]['duration']+.001: raise ValueError('Invalid times')
  if kind not in IMMEDIATE|DELAYED or ((q==a)!=(kind in IMMEDIATE)): raise ValueError('QA type/timing mismatch')
  if raw.get('language')!='en' or raw['split']!=profiles[v]['split']: raise ValueError('Language or split mismatch')
  if not raw['question'].strip() or not raw['reference_answer'].strip(): raise ValueError('Empty QA')
  modality=raw.get('evidence_modality','visual_primary_asr_supplemental_policy')
  partial='partial' in raw.get('review_status','')
  reason=[]
  if modality in AUDIO_DEPENDENT: reason.append('source_audio_or_joint_tag_requires_visual_sufficiency_review_not_an_automatic_rejection')
  if partial: reason.append('source_review_explicitly_partial')
  group='audio_or_joint_tagged' if modality in AUDIO_DEPENDENT else ('partial_review' if partial else 'memory_candidates')
  canonical.append({'schema_version':'stage3_training_task_v1','task_id':raw['task_id'],
   'video_id':v,'split':raw['split'],'language':'en','qa_type':kind,
   'question':raw['question'],'reference_answer':raw['reference_answer'],
   'query_time':q,'answer_time':a,'response_group':'immediate' if q==a else 'delayed',
   'qa_mode':'Immediate QA' if q==a else 'Proactive QA',
   'review_horizon':a,'evaluation_times':[q] if q==a else [q,a],
   'evidence_modality':modality,'evidence_modality_origin':'source_field' if 'evidence_modality' in raw else 'source_generation_policy_not_new_per_task_modality_audit',
   'source_review_status':raw.get('review_status'),'current_trainer_group':group,
   'review_revision':raw.get('review_revision'),
   'compatibility_notes':reason,'source_dataset':source,'source_file':file,'source_line':line,
   'native_evidence_review_does_not_validate_224px_input':True})
 def samples(tasks):
  result=[]
  for t in tasks:
   for current in t['evaluation_times']:
    answer=current>=t['answer_time']
    result.append({'schema_version':'stage3_train_sample_v1',
     'sample_id':f"{t['task_id']}@{current:g}",'task_id':t['task_id'],'video_id':t['video_id'],
     'question':t['question'],'video_window':[0,current],
     'target':'<ANSWER> '+t['reference_answer'] if answer else '<WAIT>',
     'meta':{'query_time':t['query_time'],'answer_time':t['answer_time'],
      'current_time':current,'qa_type':t['qa_type'],'y':int(answer),'split':t['split'],
      'decision_sampling':'reviewed_query_and_answer_endpoints_only'}})
  return sorted(result,key=lambda r:(r['video_id'],r['video_window'][1],r['task_id']))
 groups={}
 for name in ['all','memory_candidates','audio_or_joint_tagged','partial_review']:
  ts=canonical if name=='all' else [t for t in canonical if t['current_trainer_group']==name]
  ss=samples(ts)
  for split in ['train','eval']:
   write_rows(dest/'data'/name/f'{split}_tasks.jsonl',[t for t in ts if t['split']==split])
   write_rows(dest/'data'/name/f'{split}.jsonl',[r for r in ss if r['meta']['split']==split])
  groups[name]={'videos':len({t['video_id'] for t in ts}),'qa':len(ts),'decision_samples':len(ss),
   'qa_by_split':dict(Counter(t['split'] for t in ts)),
   'samples_by_split':dict(Counter(r['meta']['split'] for r in ss)),
   'qa_by_type':dict(Counter(t['qa_type'] for t in ts)),
   'decision_labels':dict(Counter('ANSWER' if r['meta']['y'] else 'WAIT' for r in ss))}
 write_rows(dest/'annotations/source_tasks.jsonl',[r for r,*_ in rows])
 for name in ['train.jsonl','eval.jsonl','train_tasks.jsonl','eval_tasks.jsonl']:
  shutil.copyfile(dest/'data/all'/name,dest/name)
 write_json(dest/'video_path_map.local.json',video_paths)
 write_json(dest/'media_manifest.json',assets)
 write_json(dest/'source_manifest.json',provenance)
 write_json(dest/'dataset_manifest.json',{
  'dataset_version':'stage3_av_training_20261112','groups':groups,
  'proactive_qa':sum(t['qa_mode']=='Proactive QA' for t in canonical),
  'immediate_qa':sum(t['qa_mode']=='Immediate QA' for t in canonical),
  'review4_completed':True,'review_correction_count':4,
  'language':'en','high_medium_videos':len(target),
  'source_datasets':SOURCES,'excluded_visual_only_source':'stage3_batch (superseded AV sibling retained instead)',
  'train_eval_video_overlap':sorted({t['video_id'] for t in canonical if t['split']=='train'}&{t['video_id'] for t in canonical if t['split']=='eval'}),
  'native_resolution_preserved':True,'no_video_reencoding':True,
  'media_included':False,'media_storage':'external originals; local map and rebase helper provided',
  'asr_annotation_available_videos':sum(bool(x['asr_segments']) for x in assets.values()),
  'asr_annotation_missing_video_ids':[v for v,x in assets.items() if not x['asr_segments']],
  'model_inputs':['causal_visual_memory','question','query_time','current_time'],
  'annotation_only':['full ASR','gold answer_time','reference/evidence','review notes'],
  'training_default':'memory_candidates: source-tag compatible candidates, not a new semantic or clinical review',
  'all_view_warning':'325 tasks carry old audio/joint tags. Tags alone do not establish that these tasks are unanswerable visually; this packaging run does not reassess their visual sufficiency. The current trainer consumes no ASR. All tasks remain in the full dataset.',
  'sampling':'1 ANSWER at query for immediate tasks; WAIT at query plus ANSWER at answer for delayed tasks. No invented intermediate WAITs, post-completion answers, or untriggered tasks.',
  'evaluation_contract':'Endpoint classification/reconstruction only. Existing stage3.overfit and stage3.metrics assume two videos and integer dense grids and must not be used unchanged for these fractional endpoint files.'})
 return groups

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[3])
 p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1])
 args=p.parse_args();print(json.dumps(build(args.workspace.resolve(),args.output.resolve()),indent=2))
