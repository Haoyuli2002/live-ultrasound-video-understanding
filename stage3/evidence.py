"""Frozen visual extraction and causal evidence materialization."""
import torch


@torch.no_grad()
def visual_features(memory, images):
    if hasattr(memory, 'extract_visual_features'):
        return memory.extract_visual_features(images)
    pixels = memory.processor.image_processor(images=images, return_tensors='pt')
    inputs = {k:v.to(memory.device) if torch.is_tensor(v) else v for k,v in pixels.items()}
    out = memory.model.get_base_model().get_image_features(
        pixel_values=inputs['pixel_values'], image_grid_thw=inputs['image_grid_thw'], return_dict=True)
    frames = out.pooler_output
    if not isinstance(frames, (list,tuple)) or len(frames) != len(images) or any(f.ndim != 2 for f in frames):
        raise ValueError('Expected per-frame merged visual patches')
    return frames


@torch.no_grad()
def materialize(memory, reader, bank, windows, current_time):
    """Load only selected anchor frames, not every frame in displayed windows."""
    result = []
    for w in windows:
        if not 0 <= w['start'] < w['end'] <= current_time or not w['frames']:
            raise ValueError('Invalid/future evidence window')
        tokens = []
        for record in w['frames']:
            if record.source_video != bank.source_video or bank.get(record.end,current_time) is not record:
                raise ValueError('Evidence frame belongs to another bank')
            if not w['start'] <= record.source_pts < w['end']:
                raise ValueError('Anchor outside evidence window')
            expected = min(reader.count-1, int(record.end*reader.fps-1e-6))
            if expected != record.frame_index:
                raise ValueError('Reader does not match indexed frame timing')
            tokens.append(visual_features(memory,[reader.frame_before(record.end)])[0].detach())
        result.append({'evidence_id':w['evidence_id'],'start':w['start'],'end':w['end'],
                       'tokens':torch.cat(tokens,dim=0)})
    return result
