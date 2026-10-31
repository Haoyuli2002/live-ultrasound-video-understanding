"""Single-GPU causal WAIT/ANSWER training from the existing Stage 3 QA JSONL."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import torch

from scripts.gpu_runtime import cuda_info
from stage2.data import VideoReader
from .model import DecisionModel, answerability_label, load_frozen_memory_model


def load_examples(path: Path) -> dict[str, list[dict]]:
    videos = {}
    for number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        meta=row.get('meta',{})
        window=row.get('video_window')
        if not isinstance(window,list) or len(window)!=2 or window[0]!=0:
            raise ValueError(f'line {number}: Stage 3 requires a causal video prefix [0,time]')
        current=float(window[1]);query=float(meta.get('query_time',row.get('query_time',-1)))
        answer=float(meta.get('answer_time',row.get('answer_time',-1)))
        if not all(math.isfinite(t) for t in (current,query,answer)) or not 0<=query<=current or current<=0:
            raise ValueError(f'line {number}: invalid query/decision time')
        if 'current_time' in meta and abs(float(meta['current_time'])-current)>1e-6:
            raise ValueError(f'line {number}: current_time differs from video prefix')
        if answer<=query:
            raise ValueError(f'line {number}: answer must require evidence after query time')
        label=answerability_label(str(row.get('target','')))
        if (label==0 and current>=answer) or (label==1 and current<answer):
            raise ValueError(f'line {number}: WAIT/ANSWER marker conflicts with answer time')
        vid=str(row.get('video_id','')).strip();question=str(row.get('question','')).strip()
        if not vid or not question:raise ValueError(f'line {number}: video_id/question required')
        # Reference evidence, answers and annotations never enter the model input.
        videos.setdefault(vid,[]).append({'video_id':vid,'question':question,
            'target':row['target'],'time':current,'query_time':query,'y':label})
    if not videos:raise ValueError('No Stage 3 examples')
    for rows in videos.values():rows.sort(key=lambda r:r['time'])
    return videos


def causal_memories(memory, reader, points):
    """Replay once per video, yielding detached memory at each decision time.

    At fractional times append one temporary latest frame to the decision input;
    do not alter the persistent integer 1-FPS recurrence with that probe.
    """
    second=0;long_state=None;short=[];last_time=None;cached=None
    duration=reader.count/reader.fps
    for row in points:
        with torch.no_grad():
            current=row['time']
            if current>duration+1e-6 or (last_time is not None and current<last_time):
                raise ValueError('Decision points must be chronological and within the video')
            if current!=last_time:
                while second+1<=current+1e-9:
                    short.append(memory.encode_frame(reader.frame_ending_at(second)).detach())
                    second+=1
                    if len(short)==60:
                        long_state=memory.update_long(long_state,torch.stack(short,dim=1)).detach()
                        short.clear()
                parts=[] if long_state is None else [long_state]
                if short:parts.append(torch.stack(short,dim=1))
                if current-second>1e-9:
                    parts.append(memory.encode_frame(reader.frame_before(current)).detach().unsqueeze(1))
                if not parts:raise ValueError('No causal visual memory available at the decision time')
                cached=torch.cat(parts,dim=1).detach()
                last_time=current
        yield row,cached


def train_video(decision, memory, reader, points, optimizer, metrics, *, epoch,
                step=0,max_steps=-1,max_grad_norm=1.0,decision_weight=1.0,text_weight=1.0):
    parameters=[p for p in decision.parameters() if p.requires_grad]
    for row,state in causal_memories(memory,reader,points):
        optimizer.zero_grad(set_to_none=True)
        bce,ntp,logit=decision.forward_answerability(state,row['question'],row['target'],row['y'])
        loss=decision_weight*bce+text_weight*ntp
        if not bool(torch.isfinite(loss).item()):raise RuntimeError('Non-finite Stage 3 training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm)
        optimizer.step();step+=1
        metrics.write(json.dumps({'step':step,'epoch':epoch,'video_id':row['video_id'],
            'decision_time':row['time'],'bce':float(bce.detach()),'ntp':float(ntp.detach()),
            'loss':float(loss.detach()),'logit':float(logit.detach().reshape(-1)[0]),
            'grad_norm':float(norm)})+'\n');metrics.flush()
        if max_steps>0 and step>=max_steps:break
    return step


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-jsonl',type=Path,required=True)
    parser.add_argument('--video-path-map',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stage1-adapter',required=True)
    parser.add_argument('--stage2-adapter',required=True)
    parser.add_argument('--model-name',default='Qwen/Qwen3.5-2B')
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--dtype',choices=('bf16','fp32'),default='bf16')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--max-grad-norm',type=float,default=1.0)
    parser.add_argument('--lora-rank',type=int,default=16)
    parser.add_argument('--frame-size',type=int,default=224)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-steps',type=int,default=-1)
    parser.add_argument('--decision-weight',type=float,default=1.0)
    parser.add_argument('--text-weight',type=float,default=1.0)
    args=parser.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Stage 3 currently supports one GPU per two-model training process')
    if args.epochs<1 or args.max_steps==0 or args.max_steps < -1 or args.max_grad_norm<=0:
        raise ValueError('Invalid epochs, max_steps or gradient norm')
    if args.decision_weight<0 or args.text_weight<0 or args.decision_weight+args.text_weight<=0:
        raise ValueError('Loss weights must be nonnegative and not both zero')
    hardware=cuda_info()
    videos=load_examples(args.train_jsonl)
    paths=json.loads(args.video_path_map.read_text())
    if any(vid not in paths or not Path(paths[vid]).is_file() for vid in videos):
        raise ValueError('Video map must cover every training video with an existing absolute path')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new Stage 3 output directory; optimizer-state resume is not implemented')
    random.seed(args.seed);torch.manual_seed(args.seed)
    dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[args.dtype]
    memory=load_frozen_memory_model(args.model_name,stage1_adapter=args.stage1_adapter,
                                   stage2_adapter=args.stage2_adapter,dtype=dtype)
    memory.model.to('cuda').eval()
    decision=DecisionModel(args.model_name,stage1_adapter=args.stage1_adapter,
        stage2_adapter=args.stage2_adapter,dtype=dtype,rank=args.lora_rank).to('cuda')
    for model in (memory.model,decision.model):
        if hasattr(model.config,'use_cache'):model.config.use_cache=False
    decision.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    decision.train()
    optimizer=torch.optim.AdamW((p for p in decision.parameters() if p.requires_grad),lr=args.learning_rate)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'training_config.json').write_text(json.dumps({
        **{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'hardware':hardware},indent=2)+'\n')
    step=0
    with (args.output_dir/'training_metrics.jsonl').open('w') as metrics:
        for epoch in range(args.epochs):
            for vid,points in videos.items():
                reader=VideoReader(paths[vid],args.frame_size)
                try:
                    step=train_video(decision,memory,reader,points,optimizer,metrics,epoch=epoch,
                        step=step,max_steps=args.max_steps,max_grad_norm=args.max_grad_norm,
                        decision_weight=args.decision_weight,text_weight=args.text_weight)
                finally:reader.close()
                if args.max_steps>0 and step>=args.max_steps:break
            decision.save(args.output_dir/f'epoch-{epoch+1}')
            if args.max_steps>0 and step>=args.max_steps:break
    decision.save(args.output_dir/'final')


if __name__=='__main__':main()
