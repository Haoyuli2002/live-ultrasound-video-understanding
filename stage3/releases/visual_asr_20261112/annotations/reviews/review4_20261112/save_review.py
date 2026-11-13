import copy,json
from pathlib import Path
p=Path(__file__).resolve().parent
original=[json.loads(l) for l in (p/'original_tasks.jsonl').read_text().splitlines()]
opened={
'pClYyRZlYE8':[330,335,336,337,338,339,340,340.25,340.5,340.75,341,342,343,344,345,346,347],
'OJB8ZNtVAvA':[740,745,750,790,791,792,793,794,795,796,797,798,799,800,801,801.0625,801.125,801.25,801.5,801.75,802]}
updates=[];logs=[]
for old in original:
 r=copy.deepcopy(old);v=r['video_id'];tid=r['task_id'];a=r['answer_time'];q=r['query_time']
 if tid=='s3av_v2_pClYyRZlYE8_01':
  r['question']='In this transverse sonogram, where is the larger dark lumen relative to the smaller dark lumen?'
  r['reference_answer']='The larger dark lumen is above and to the right of the smaller dark lumen.'
  times=[330];finding='The two lumen positions are visible at330. Removed needle-approach clause: the dotted guide must not be conflated with an actually localized needle.'
 elif tid=='s3av_v2_pClYyRZlYE8_02':
  r['question']='Notify me when a separate bright echo becomes clearly visible inside the larger dark lumen, below its upper wall.'
  r['reference_answer']='A small bright echo is now visible in the upper part of the larger dark lumen, below and separate from its upper wall.'
  r['answer_time']=340.5;times=[335,336,337,338,339,340,340.25,340.5]
  finding='335 query: main bright echoes are above the lumen; no distinct separate intraluminal focus.340.25 insufficiently distinct;340.5 a small bright echo is distinguishable below the upper wall.346 was late. Reworded to observable echo, without claiming exact physical needle-tip location or successful cannulation.'
  r['query_time_missing_evidence']='At335 the bright echoes above the lumen do not establish a separate bright focus within its dark interior.'
  r['answer_time_new_evidence']='At340.5 a separate small bright focus is visible in the upper dark interior, below the upper boundary.'
 elif tid=='s3av_v2_OJB8ZNtVAvA_01':
  times=[745];finding='745 original full frame shows probe held on exposed anterior chest and ultrasound image at lower left. Original question and answer retained.'
 else:
  r['answer_time']=801.125
  times=[790,791,792,793,794,795,796,797,798,799,800,801,801.0625,801.125]
  finding='790 query and inspected791–801.0625 show color imaging without lower spectral panel. At801.125 an already populated spectral trace is clearly visible below the small heart image.802 was late; moved to801.125.'
  r['query_time_missing_evidence']='At790 the lower-left inset is a color-flow image without a lower spectral trace.'
  r['answer_time_new_evidence']='At801.125 the inset shows a small heart image above a lower spectral trace with multiple bright peaks.'
 r['visual_evidence']=[json.loads((p/'frames'/v/f'{t:g}.json').read_text()) for t in times]
 assert all(e['source_pts_seconds']<e['checkpoint']<=r['answer_time'] for e in r['visual_evidence'])
 r['visual_evidence_seconds']=[e['source_pts_seconds'] for e in r['visual_evidence']]
 r['evidence']=r['visual_evidence']
 r['first_answer_time']=r['answer_time']
 r['answer_time_bracket']={'last_checked_insufficient':None if q==r['answer_time'] else times[-2],
  'first_checked_sufficient':r['answer_time'],'precision':'Question-led native-frame sampled checkpoints; not an exhaustive source-frame onset claim.'}
 r['audio_evidence_ranges']=[pair for pair in r.get('audio_evidence_ranges',[]) if pair[1]<=r['answer_time']]
 r['review_status']='accepted_targeted_native_frame_self_review'
 r['review_revision']='stage3_review4_20261112'
 r['review_note']=finding
 r['full_continuous_video_review']=False;r['direct_audio_listening']=False;r['independent_review']=False
 r['student_input_visibility']='not_checked'
 r['review_scope']='Same-assistant question-led native-resolution full-frame checkpoint review; prior draft and future context visible. Not blind or continuous replay. No direct listening claim.'
 updates.append(r)
 changed={k:{'before':old.get(k),'after':r.get(k)} for k in ['question','reference_answer','query_time','answer_time','review_status'] if old.get(k)!=r.get(k)}
 logs.append({'task_id':tid,'video_id':v,'finding':finding,'changes':changed,'qa_evidence_checkpoints':times,
  'all_actually_opened_context_frames_for_video':[json.loads((p/'frames'/v/f'{t:g}.json').read_text()) for t in opened[v]],
  'query_prefix_result':'ANSWER' if q==r['answer_time'] else 'WAIT','answer_prefix_result':'ANSWER',
  'shortcut_risk':'ASR, labels and prior instruction can cue interpretation; no transcript-only or model-input ablation was performed.',
  'result':'accepted_after_targeted_native_frame_review'})
(p/'reviewed_tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in updates))
(p/'review_log.json').write_text(json.dumps({'reviewed_tasks':4,'native_frames_opened':sum(map(len,opened.values())),
'review_scope':'Same-assistant question-led sampled native-frame review, not blind or continuous replay.','tasks':logs},ensure_ascii=False,indent=2)+'\n')
print([(r['task_id'],r['query_time'],r['answer_time']) for r in updates])
