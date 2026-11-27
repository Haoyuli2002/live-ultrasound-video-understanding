"""Scratch-backed frozen per-video features with bounded mmap handle retention."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import torch


class VideoFeatureCache:
    def __init__(self, root, max_open_videos=4):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        if max_open_videos<1:raise ValueError('max_open_videos must be positive')
        self.max_open_videos=max_open_videos
        self.videos={};self.endpoints={};self.open=OrderedDict()

    def add_video(self, video_id, tensor, endpoint_indices):
        if video_id in self.videos:raise ValueError('Duplicate video features')
        if tensor.ndim!=4 or tensor.shape[0]!=1:raise ValueError('Expected [1,T,N,D] features')
        for sample_id,indices in endpoint_indices.items():
            if sample_id in self.endpoints:raise ValueError('Duplicate sample ID')
            if not indices or any(i<0 or i>=tensor.shape[1] for i in indices):
                raise ValueError('Invalid causal feature indices')
        filename=hashlib.sha256(video_id.encode()).hexdigest()+'.pt'
        path=self.root/filename
        if path.exists():raise ValueError('Refuse to reuse unverified cached features')
        temporary=path.with_suffix('.partial')
        torch.save(tensor.detach().cpu().contiguous(),temporary);temporary.replace(path)
        self.videos[video_id]=dict(file=filename,shape=list(tensor.shape),dtype=str(tensor.dtype))
        for sid,indices in endpoint_indices.items():self.endpoints[sid]=(video_id,list(indices))

    def __getitem__(self,sample_id):
        video_id,indices=self.endpoints[sample_id]
        if video_id in self.open:features=self.open.pop(video_id)
        else:features=torch.load(self.root/self.videos[video_id]['file'],map_location='cpu',weights_only=True,mmap=True)
        self.open[video_id]=features
        while len(self.open)>self.max_open_videos:self.open.popitem(last=False)
        # Only this endpoint's own grid and final fractional frame are gathered.
        # A contiguous 1 FPS prefix can share the read-only mmap storage.
        # Some manifests interleave other questions' fractional endpoints.
        if (indices[-1] - indices[0] + 1 == len(indices)
                and all(index == indices[0] + offset for offset, index in enumerate(indices))):
            return features[:, indices[0]:indices[-1] + 1]
        return features[:,indices].contiguous()

    def write_manifest(self):
        (self.root/'manifest.json').write_text(json.dumps(dict(videos=self.videos,endpoints=self.endpoints,
            note='Frozen encoder features only; trainable projections/memory recomputed each optimizer step.'),indent=2)+'\n')


def prepare_distributed_features(builder,model,examples,root,size,chunk,rank,world):
    import torch.distributed as dist
    videos=sorted({e['video_id'] for e in examples});owned=set(videos[rank::world])
    directory=Path(root)/'feature_shards'/f'rank_{rank}';directory.mkdir(parents=True,exist_ok=True)
    local=builder(model,[e for e in examples if e['video_id'] in owned],directory,size,chunk,'video_mmap')
    del local
    dist.barrier()
    cache=VideoFeatureCache(root)
    audit=[]
    for worker in range(world):
        shard=Path(root)/'feature_shards'/f'rank_{worker}'
        data=json.loads((shard/'features/manifest.json').read_text())
        if set(cache.videos)&set(data['videos']) or set(cache.endpoints)&set(data['endpoints']):
            raise ValueError('Duplicate distributed feature shard')
        cache.videos.update({v:dict(info,file=str((shard/'features'/info['file']).relative_to(root))) for v,info in data['videos'].items()})
        cache.endpoints.update(data['endpoints'])
        audit.extend(json.loads((shard/'frame_audit.json').read_text()))
    if set(cache.endpoints)!={e['sample_id'] for e in examples}:raise ValueError('Missing distributed features')
    if rank==0:(Path(root)/'frame_audit.json').write_text(json.dumps(sorted(audit,key=lambda r:r['video_id']),indent=2)+'\n')
    return cache
