"""Export train/holdout loss curves and separate per-GPU utilization traces."""
import argparse
import json
from pathlib import Path


def rows(path):
    result=[]
    if not path.exists():return result
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue
    return result


def render_plots(run):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from .dashboard import gpu_samples
    run=Path(run);output=run/'plots';output.mkdir(exist_ok=True)
    training=[r for r in rows(run/'training.jsonl') if r.get('phase')=='train']
    holdout=rows(run/'holdout_metrics.jsonl')
    if not holdout:holdout=rows(run/'eval_metrics.jsonl')
    fields=[('Total loss','loss','training_objective_fullset'),('Answer NTP','answer_loss','answer_ce'),
            ('Readiness','readiness_loss','readiness_objective_loss'),('BCE','readiness_weighted_bce','readiness_weighted_bce'),
            ('CL','readiness_contrastive','readiness_contrastive'),('TR (before coefficient)','readiness_temporal','readiness_temporal')]
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for ax,(title,tk,hk) in zip(axes.flat,fields):
        ts=[r for r in training if r.get(tk) is not None];hs=[r for r in holdout if r.get(hk) is not None]
        if ts:ax.plot([r['step'] for r in ts],[r[tk] for r in ts],label='Train minibatch',alpha=.75,linewidth=1)
        if hs:ax.plot([r['optimizer_step'] for r in hs],[r[hk] for r in hs],'o-',label='Holdout full set',linewidth=1.6)
        ax.set(title=title,xlabel='Optimizer step',ylabel='Loss');ax.grid(alpha=.2)
        if ts or hs:ax.legend()
    fig.suptitle('Train and holdout: same loss definitions, different data and averaging populations')
    tmp=output/'loss_curves.tmp.png';fig.savefig(tmp,dpi=150);tmp.replace(output/'loss_curves.png');plt.close(fig)
    samples=gpu_samples(run/'logs/gpu.csv')
    devices=sorted({r['uuid'] for r in samples})
    if devices:
        fig,axes=plt.subplots(len(devices),3,figsize=(14,3.2*len(devices)),squeeze=False,constrained_layout=True)
        origin=min(r['time'] for r in samples)
        for row,uuid in zip(axes,devices):
            data=[r for r in samples if r['uuid']==uuid];stride=max(1,len(data)//4000);data=data[::stride]
            times=[(r['time']-origin)/60 for r in data]
            for ax,title,values in zip(row,['GPU utilization (%)','VRAM capacity used (%)','VRAM used (GiB)'],
                    [[r['gpu'] for r in data],[100*r['used']/r['total'] for r in data],[r['used'] for r in data]]):
                ax.plot(times,values,linewidth=.7);ax.set(title=title,xlabel='Minutes since first sample',ylabel=uuid[-12:]);ax.grid(alpha=.2)
                if '%' in title:ax.set_ylim(0,100)
        tmp=output/'gpu_curves.tmp.png';fig.savefig(tmp,dpi=150);tmp.replace(output/'gpu_curves.png');plt.close(fig)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args();render_plots(args.run_dir)
