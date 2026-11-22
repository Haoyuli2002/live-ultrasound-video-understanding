"""Fixed video holdout monitoring without optimizer updates or test-set access."""
import json
import math
from pathlib import Path
import statistics

import torch


def record_gpu_memory(output, rank, step, phase, reset=False):
    path=Path(output)/f'gpu_memory.rank{rank}.jsonl'
    row=dict(rank=rank,global_step=step,phase=phase,gpu=torch.cuda.get_device_name(),
             allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    summary={p:{key:max(r[key] for r in rows if r['phase']==p) for key in ('peak_allocated_bytes','peak_reserved_bytes')}
             for p in {r['phase'] for r in rows}}
    path.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    if reset:torch.cuda.reset_peak_memory_stats()


def split_videos(videos, holdout_ids):
    ids=list(holdout_ids)
    if not ids or len(set(ids))!=len(ids) or not set(ids)<set(videos):
        raise ValueError('Holdout must be a unique proper subset of TRAIN videos')
    selected=set(ids)
    return ({v:b for v,b in videos.items() if v not in selected},
            {v:videos[v] for v in ids})


def plot_curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    output=Path(output); bins=defaultdict(list)
    for path in output.glob('training_metrics.rank*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line);bins[((row['global_step']-1)//100+1)*100].append(row['loss'])
    reports=[json.loads(p.read_text()) for p in sorted((output/'holdout').glob('step-*/report.json'))]
    reports.sort(key=lambda r:r['global_step'])
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(sorted(bins),[statistics.mean(bins[b]) for b in sorted(bins)],label='Train: target-mean CE, 100-step bins')
    ax.plot([r['global_step'] for r in reports],[r['loss'] for r in reports],'o-',label='Holdout: target-mean CE, eval mode')
    ax.set(xlabel='Optimizer step',ylabel='Cross-entropy loss',title='Stage 2 train / fixed 16-video holdout')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(output/'loss_curves.png',dpi=160);fig.savefig(output/'loss_curves.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for kind in ('short','block','global'):
        ax.plot([r['global_step'] for r in reports],[r['scores']['real/'+kind]['rouge_l_f1'] for r in reports],'o-',label=kind)
    ax.set(xlabel='Optimizer step',ylabel='ROUGE-L F1',title='Holdout free-running reconstruction (lexical metric)')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'holdout_quality.png',dpi=160);plt.close(fig)
    (output/'holdout/history.json').write_text(json.dumps(reports,indent=2))


def monitor(memory,videos,paths,output,step,rank,world,frame_size=256,max_new_tokens=160):
    from .data import VideoReader
    from .reconstruct import collect_memories,greedy_reconstruct,text_metrics,summarize
    from .provenance import digest
    output=Path(output);folder=output/'holdout'/f'step-{step:08d}'
    if (folder/'report.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    # Balance whole videos by target count; never break recurrent chronology.
    counts={v:sum(len(b['local_sub_summaries'])+2*(b['block_window'][1]-b['block_window'][0]==60) for b in blocks)
            for v,blocks in videos.items()}
    assignments=[[] for _ in range(world)];sizes=[0]*world
    for vid in sorted(videos,key=lambda v:(-counts[v],v)):
        i=min(range(world),key=lambda j:(sizes[j],j));assignments[i].append(vid);sizes[i]+=counts[vid]
    record_gpu_memory(output,rank,step,'train',reset=True)
    # Preserve training RNG and mode; previous training Long is owned by caller.
    try:
        memory.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode(), (folder/f'rank-{rank}.jsonl').open('w') as stream:
            for vid in assignments[rank]:
                reader=VideoReader(paths[vid],frame_size)
                try:records=collect_memories(memory,videos[vid],reader)
                finally:reader.close()
                for record in records:
                    loss=float(memory.reconstruction_loss(record['memory'],record['reference'],record['kind']))
                    if not math.isfinite(loss):raise FloatingPointError('Nonfinite holdout loss')
                    generated=greedy_reconstruct(memory,record['memory'],record['kind'],max_new_tokens)
                    row=dict(video_id=vid,kind=record['kind'],window=record['window'],condition='real',
                             reference=record['reference'],loss=loss,**generated)
                    row.update(text_metrics(row['prediction'],row['reference']))
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(f'[holdout] step={step} rank={rank} video={vid} targets={len(records)}',flush=True)
                if records:del record
                records.clear()
    finally:
        memory.reset_short_memory();memory.train()
        record_gpu_memory(output,rank,step,'holdout',reset=True)
    if world>1:torch.distributed.barrier()
    if rank==0:
        rows=[]
        for i in range(world):rows.extend(json.loads(s) for s in (folder/f'rank-{i}.jsonl').read_text().splitlines())
        keys=[(r['video_id'],r['kind'],tuple(r['window'])) for r in rows]
        if len(rows)!=sum(counts.values()) or len(set(keys))!=len(rows):raise ValueError('Holdout coverage mismatch')
        report=dict(global_step=step,videos=len(videos),targets=len(rows),loss=statistics.mean(r['loss'] for r in rows),
                    loss_by_kind={k:statistics.mean(r['loss'] for r in rows if r['kind']==k) for k in ('short','block','global')},
                    scores=summarize(rows),empty_outputs=sum(not r['prediction'].strip() for r in rows),
                    reference_used_in_generation=False,loss_reduction='mean of per-target token-mean CE',
                    interpretation='Fixed training holdout; not final test. Train curve uses train mode; holdout uses eval mode.')
        (folder/'report.json').write_text(json.dumps(report,indent=2))
        with (folder/'review.txt').open('w') as f:
            for r in rows:f.write(f"{r['video_id']} {r['kind']} {r['window']}\nREF: {r['reference']}\nGEN: {r['prediction']}\n\n")
        plot_curves(output)
        print('[holdout-summary]',json.dumps(report),flush=True)
    if world>1:torch.distributed.barrier()
