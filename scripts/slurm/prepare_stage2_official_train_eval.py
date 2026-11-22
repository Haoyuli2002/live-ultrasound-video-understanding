"""Freeze working-tree bytes for official-base train/eval; does not submit."""
import random
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def prepare(repo, output, gpus=4, epochs=1, eval_each_epoch=False):
    from stage2.release import validate_release
    from stage2.dataset_versions import default_release, validate_dataset_version
    repo=repo.resolve(); output=output.resolve()
    release=default_release('visual_asr', repo, Path('/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding'))
    manifest=validate_release(release/'manifest.json')
    validate_dataset_version(manifest, 'visual_asr')
    full=repo/'cluster_data/experiments/stage2/full_qformer_20261112T104925Z_9FcvVy'
    paths=json.loads((full/'video_map.json').read_text())
    assert all(Path(paths[v['video_id']]).is_file() for v in manifest['videos'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir();(output/'code').mkdir()
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
    extra=['stage2/holdout.py','stage2/test_holdout.py','stage2/train_eval_pipeline.py','stage2/test_train_eval_pipeline.py',
           'scripts/slurm/run_stage2_official_train_eval.sbatch',
           'scripts/slurm/run_stage2_official_train_eval_8a100_3ep.sbatch',
           'scripts/slurm/prepare_stage2_official_train_eval.py']
    hashes={}
    for name in sorted(set(tracked+extra)):
        src=repo/name
        if not name or not src.is_file():continue
        dest=output/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest);hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    shutil.copytree(release,output/'release')
    shutil.copy2(full/'video_map.json',output/'video_map.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
    (output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    (output/'working_tree.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=repo))
    holdout=sorted(random.Random(20261115).sample(sorted(set(manifest['splits']['train'])-{'P3FoSFIrfQA'}),16))
    train_ids=sorted(set(manifest['splits']['train'])-set(holdout))
    records={v['video_id']:v for v in manifest['videos']}
    train_targets=sum(records[v]['targets'] for v in train_ids)
    holdout_targets=sum(records[v]['targets'] for v in holdout)
    (output/'holdout_ids.json').write_text(json.dumps(holdout,indent=2))
    (output/'split_manifest.json').write_text(json.dumps(dict(seed=20261115,train=train_ids,holdout=holdout,test=manifest['splits']['eval'],train_targets=train_targets,holdout_targets=holdout_targets,test_targets=sum(records[v]['targets'] for v in manifest['splits']['eval'])),indent=2))
    plan=dict(model_name='Qwen/Qwen3.5-2B',initialization='official_checkpoint',init_adapter=None,
              architecture='qformer_secondwise_v2',epochs=epochs,gpus=gpus,per_rank_targets=1,
              learning_rate=1e-4,save_steps=100,epoch_checkpoints=True,
              partition='lrz-dgx-a100-80x8' if gpus==8 else 'lrz-hgx-h100-94x4',qos='gpu',walltime='48:00:00',
              dataset_version='visual_asr',frame_size=256,train_videos=len(train_ids),train_targets=train_targets,
              holdout_videos=len(holdout),holdout_targets=holdout_targets,holdout_every=1000,holdout_at_end=True,
              eval_videos=len(manifest['splits']['eval']),eval_targets=sum(records[v]['targets'] for v in manifest['splits']['eval']),eval_conditions=['real','zero','other_window'],
              expected_eval_records=3*sum(records[v]['targets'] for v in manifest['splits']['eval']),eval_each_epoch=eval_each_epoch,
              eval_checkpoints=[f'epoch-{i}' for i in range(1,epochs+1)] if eval_each_epoch else ['final'],smoke_epochs=0,smoke_enabled=False,
              base_commit=commit,snapshot='working-tree bytes bound by source_hashes.json')
    (output/'submission.json').write_text(json.dumps(plan,indent=2))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--gpus',type=int,choices=[4,8],default=4)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--eval-each-epoch',action='store_true')
    args=parser.parse_args();prepare(args.repo,args.output,args.gpus,args.epochs,args.eval_each_epoch)
