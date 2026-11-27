"""Full-data planning and delayed loss selection, without loading a model."""
import copy
import json
from pathlib import Path
from .data import digest, load_examples, sample_times
from .optimization import distribute_batches, task_batches


def batch_plan(rows,batch_size,epochs,seed,world_size=1,video_block_size=0):
    if batch_size<1 or epochs<1:raise ValueError('Positive batch size and epochs required')
    orders=[task_batches(rows,batch_size,seed+epoch,video_block_size) for epoch in range(1,epochs+1)]
    expected={r['sample_id'] for r in rows}
    for batches in orders:
        ids=[r['sample_id'] for batch in batches for r in batch]
        if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('Incomplete or repeated training endpoints')
        assignment={r['task_id']:i for i,batch in enumerate(batches) for r in batch}
        if any(any(assignment[r['task_id']]!=i for r in batch) for i,batch in enumerate(batches)):
            raise ValueError('A question was split across optimizer updates')
    if world_size < 1:raise ValueError('world_size must be positive')
    if world_size > 1:
        orders = [distribute_batches(batches, world_size, bool(video_block_size), seed + epoch)
                  for epoch, batches in enumerate(orders, 1)]
    counts=[len(batches) for batches in orders]
    return dict(batch_size=batch_size,batch_size_per_gpu=batch_size,global_batch_capacity=batch_size*world_size,
                gradient_accumulation=1,gpus=world_size,epochs=epochs,
                optimizer_steps_per_epoch=counts,total_optimizer_steps=sum(counts),
                batch_order=([[[[r['sample_id'] for r in batch] for batch in ranks] for ranks in steps] for steps in orders]
                             if world_size > 1 else [[[r['sample_id'] for r in batch] for batch in batches] for batches in orders]))


def validate_official_split(train,holdout,manifest):
    for name,rows in [('train',train),('holdout',holdout)]:
        ids={r['video_id'] for r in rows}
        if not ids.issubset(set(manifest[name])):raise ValueError(f'{name} contains videos outside official split')
        if ids & set(manifest['test']):raise ValueError('Test video leakage')
        if any(r['split']!=name for r in rows):raise ValueError('Split label mismatch')
    if {r['video_id'] for r in train}&{r['video_id'] for r in holdout}:raise ValueError('Train/holdout overlap')
    if {r['video_id'] for r in holdout}!=set(manifest['holdout']):raise ValueError('Missing holdout videos')


def feature_storage_estimate(rows,tokens=64,width=2048,bytes_per_value=2):
    per_video={}
    endpoint_frames=0
    for r in rows:
        times=sample_times(r['current_time']);endpoint_frames+=len(times)
        per_video.setdefault(r['video_id'],set()).update(times)
    unique=sum(map(len,per_video.values()))
    size=tokens*width*bytes_per_value
    return dict(unique_frames=unique,duplicated_endpoint_frames=endpoint_frames,
                shared_features_gib=unique*size/2**30,old_endpoint_features_gib=endpoint_frames*size/2**30,
                assumption='64 patches/frame x 2048 dimensions x BF16; actual encoder audit confirms shape at runtime')


def resolve_recipe(cfg):
    """Read a completed selection; never fall back to a provisional loss."""
    root=Path(cfg['readiness_recipe_selection'])
    status=json.loads((root/'status.json').read_text())
    if status.get('status')!='completed':raise ValueError('Loss ablation/selection is not completed; full training remains prepared only')
    decision=json.loads((root/'selection.json').read_text())
    if decision['action'] not in ('reuse_bce_checkpoint','fresh_all_pairs_training'):
        raise ValueError('Unrecognized selection outcome')
    source=Path(decision['source']);selected=json.loads((source/'experiment.json').read_text())
    plan=json.loads((root/'selection_plan.json').read_text())
    arm=next(a for a in plan['arms'] if a['name']==decision['winner'])
    if Path(arm['run_dir']).resolve()!=source.resolve() or digest(source/'experiment.json')!=arm['config_sha256']:
        raise ValueError('Selected recipe source differs from frozen comparison')
    result=copy.deepcopy(cfg)
    for key in ['readiness_objective','timing_lambda_reg','readiness_bce_positive_weight','readiness_bce_weight']:
        result[key]=selected[key]
    result['timing_pair_mode']='all_pairs'
    result['model']['readiness_threshold']=decision['threshold']
    result['readiness_recipe_resolved']=dict(winner=decision['winner'],source=str(source),
        source_config_sha256=digest(source/'experiment.json'),selection_sha256=digest(root/'selection.json'),
        initialization='original pretrained assets and fresh optimizer; selected ultrasound checkpoint is NOT restored')
    if result.get('init_checkpoint') is not None or result.get('restore_optimizer'):
        raise ValueError('Full fresh run must not restore overfit weights/optimizer')
    return result
