"""CPU-only audit of frozen WAIT/ANSWER annotations and exact model input frames."""
import argparse,collections,hashlib,json,math,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from decord import VideoReader,cpu
from .data import causal_indices

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):
 try:return ImageFont.truetype(FONT,n)
 except OSError:return ImageFont.load_default()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
 r=args.source_run;o=args.output_dir;o.mkdir(parents=True,exist_ok=False);(o/'sheets').mkdir();(o/'frames256').mkdir();(o/'boundary_native').mkdir()
 tasks=[json.loads(s) for s in (r/'tasks.jsonl').read_text().splitlines()];samples=[json.loads(s) for s in (r/'samples.jsonl').read_text().splitlines()]
 audits={x['video_id']:x for x in json.loads((r/'frame_audit.json').read_text())};videos=json.loads((r/'video_map.json').read_text())
 pred={x['sample_id']:x for x in json.loads((r/'eval_epoch_100.json').read_text())['predictions']}
 selected=[t for t in tasks if t['qa_type'] in ('event_notification','temporal_comparison')]
 groups=collections.defaultdict(list)
 for t in selected:groups[(t['video_id'],t['query_time'],t['answer_time'])].append(t)
 results=[];evidence_results=[]
 for gi,((vid,q,a),ts) in enumerate(groups.items()):
  reader=VideoReader(videos[vid],ctx=cpu(0),num_threads=2);pts=reader.get_frame_timestamp(np.arange(len(reader)))[:,0].tolist()
  group_id=f'{gi:02d}_{vid}_{q}_{a}';end=a if a is not None else max(ts[0]['evaluation_times'])
  eval_times=sorted({float(s['video_window'][1]) for s in samples if s['task_id'] in {t['task_id'] for t in ts}})
  # Dense quarter-second boundary grid, plus every actual 1 fps sample in the query/answer interval.
  times=set(eval_times)
  if a is not None:times.update(float(x) for x in range(math.floor(q),math.floor(a)+1));times.update(round(a-1+i*.25,4) for i in range(9) if a-1+i*.25>=0)
  else:times.update(float(e['time']) for t in ts for e in t.get('evidence',[]))
  times.update(float(e['time']) for t in ts for e in t.get('evidence',[]))
  times=sorted(times);indices=causal_indices(pts,times);imgs={};small={};info={}
  original=reader.get_batch(indices).asnumpy()
  for time,idx,arr in zip(times,indices,original):
   im=Image.fromarray(arr);im256=ImageOps.pad(im,(256,256),method=Image.Resampling.BICUBIC);imgs[time]=im;small[time]=im256
   name=f'{vid}_{time:g}.png';im256.save(o/'frames256'/name)
   info[time]=dict(requested_time=time,actual_pts=pts[idx],frame_index=idx,shape=list(arr.shape),model_image='frames256/'+name,
                   pixel_sha256=hashlib.sha256(np.array(im256).tobytes()).hexdigest())
   old=audits[vid];lookup={t:i for i,t in enumerate(old['requested_times'])}
   if time in lookup:
    j=lookup[time];info[time]['matches_training_audit']=idx==old['frame_indices'][j] and abs(pts[idx]-old['actual_pts'][j])<1e-5
  lastwait=max(x for x in eval_times if a is None or x<a)
  for t in ts:
   for role in ('evidence','waiting_evidence'):
    for ev in t.get(role,[]):
     path=Path(ev['path']);at=float(ev['time']);idx=causal_indices(pts,[at])[0]
     er=dict(task_id=t['task_id'],role=role,time=at,source_path=str(path),exists=path.exists(),annotation_frame=ev.get('frame_index'),actual_frame=idx,frame_matches=idx==ev.get('frame_index'),actual_pts=pts[idx],annotation_pts=ev.get('approx_pts'))
     if path.exists():er['file_sha256_matches']=hashlib.sha256(path.read_bytes()).hexdigest()==ev.get('sha256')
     evidence_results.append(er)
  def sheet(sheet_times,name,title):
   cols=4;header=110;w=1024;h=header+math.ceil(len(sheet_times)/cols)*285
   canvas=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(canvas)
   d.text((8,5),title,fill='black',font=font(17))
   text=' | '.join(t['task_id'] for t in ts)+'\n'+ts[0]['question']
   for j,line in enumerate(textwrap.wrap(text,110)):d.text((8,30+j*19),line,fill='black',font=font(15))
   for j,tm in enumerate(sheet_times):
    x=(j%cols)*256;y=header+(j//cols)*285;canvas.paste(small[tm],(x,y))
    label='ANSWER' if a is not None and tm==a else 'WAIT' if tm in eval_times and (a is None or tm<a) else 'context/unlabeled'
    d.text((x+3,y+258),f'{tm:g}s PTS {info[tm]["actual_pts"]:.3f} {label}',fill='black',font=font(13))
   canvas.save(o/'sheets'/name)
  actualtimes=sorted(t for t in times if t.is_integer() and q<=t<=end) if a is not None else eval_times
  sheet(actualtimes,group_id+'_timeline.jpg',f'{group_id}: EXACT 256px inputs, 1fps query through answer')
  if a is not None:
   boundarytimes=sorted(t for t in times if a-1<=t<=a+1);sheet(boundarytimes,group_id+'_boundary.jpg',f'{group_id}: 0.25s inspection (fractional frames NOT training inputs)')
   wim=imgs[lastwait];aim=imgs[a]
   canvas=Image.new('RGB',(1280,440),'white');d=ImageDraw.Draw(canvas)
   for j,(tm,im) in enumerate([(lastwait,wim),(a,aim)]):
    thumb=ImageOps.contain(im,(640,400));canvas.paste(thumb,(640*j,30));d.text((640*j+4,5),f'{tm:g}s {"WAIT" if j==0 else "ANSWER"} source {im.size}',fill='black',font=font(19))
    im.save(o/'boundary_native'/f'{vid}_{tm:g}.png')
   canvas.save(o/'sheets'/f'{group_id}_native.jpg')
  pair=[]
  for t in ts:
   rows=[s for s in samples if s['task_id']==t['task_id']]
   for s in rows:
    tm=float(s['video_window'][1]);pp=pred[s['sample_id']]
    pair.append(dict(task_id=t['task_id'],time=tm,label='ANSWER' if pp['expected_ready'] else 'WAIT',probability=pp['probability'],**info[tm]))
  ma=float(np.abs(np.array(small[lastwait],dtype=float)-np.array(small[a],dtype=float)).mean()) if a is not None else None
  results.append(dict(group_id=group_id,video_id=vid,query=q,answer=a,last_wait=lastwait,tasks=ts,sample_frames=pair,frames=list(info.values()),last_pair_pixel_mae_0_255=ma,last_pair_identical_pixels=info[lastwait]['pixel_sha256']==info[a]['pixel_sha256'] if a is not None else None))
  print(group_id,'done',flush=True)
 write(o/'groups.json',results);write(o/'evidence_audit.json',evidence_results)
 summary=dict(tasks=len(selected),triggered_tasks=sum(t['answer_time'] is not None for t in selected),groups=len(results),training_sample_frames=sum(len(x['sample_frames']) for x in results),training_alignment_failures=[x for g in results for x in g['sample_frames'] if not x.get('matches_training_audit',False)],evidence_records=len(evidence_results),evidence_frame_mismatches=[x for x in evidence_results if not x['frame_matches']],missing_evidence=[x for x in evidence_results if not x['exists']],evidence_hash_mismatches=[x for x in evidence_results if x.get('file_sha256_matches') is False],exact_duplicate_boundaries=[x['group_id'] for x in results if x['last_pair_identical_pixels']],source=str(r),model_preprocessing='1fps plus exact decision endpoint; causal PTS<=time; PIL ImageOps.pad 256 BICUBIC, identical to training before normalization')
 write(o/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
