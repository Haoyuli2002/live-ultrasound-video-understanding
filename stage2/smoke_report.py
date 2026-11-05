"""Functional Q-Former smoke gate: gradients, reload, losses and free decoding."""
import argparse
import json
import math
from pathlib import Path
from .provenance import digest
from .qformer import SECONDWISE_ARCHITECTURE


def check(run):
    run=Path(run);cfg=json.loads((run/'training/training_config.json').read_text())
    if cfg['memory_architecture']!='qformer':raise ValueError('Not a Q-Former smoke')
    metrics=[json.loads(x) for x in (run/'training/training_metrics.jsonl').read_text().splitlines() if x.strip()]
    if len(metrics)!=16*cfg['epochs'] or not all(math.isfinite(r['loss']) for r in metrics):
        raise ValueError('Incomplete/nonfinite smoke training')
    gradient=json.loads((run/'training/compressor_training_report.json').read_text())
    if any(not gradient[k]['all_finite'] or gradient[k]['changed_parameters']<1 for k in ['short','long']):
        raise ValueError('Both Short and Long compressors must learn finite parameters')
    checkpoint=run/'training/final'
    for file in ['adapter_config.json','adapter_model.safetensors','qformer_config.json','qformer.pt']:
        if not (checkpoint/file).is_file() or not (checkpoint/file).stat().st_size:
            raise FileNotFoundError(file)
    qformer_config=json.loads((checkpoint/'qformer_config.json').read_text())
    if (qformer_config.get('architecture') != SECONDWISE_ARCHITECTURE
            or qformer_config.get('short_queries') != 1
            or qformer_config.get('short_update_seconds') != 1):
        raise ValueError('Secondwise Q-Former needs a new single-query smoke; legacy smoke cannot pass')
    for name in ['epoch-1','final']:
        values=[json.loads(x) for x in (run/f'reconstruction_{name}.jsonl').read_text().splitlines() if x.strip()]
        if len(values)!=16 or not all(math.isfinite(r['loss']) for r in values):
            raise ValueError('Failed checkpoint reload/teacher-forced evaluation')
    recon=run/'reconstruct_final'
    manifest=json.loads((recon/'manifest.json').read_text())
    values=[json.loads(x) for x in (recon/'predictions.jsonl').read_text().splitlines() if x.strip()]
    if len(values)!=48 or manifest['reference_used_in_decoding'] is not False or any('skipped' in r for r in values):
        raise ValueError('Incomplete free-running memory-control evaluation')
    for kind in ['short','block','global']:
        if not any(r['kind']==kind and r['condition']=='real' and r['prediction'].strip() for r in values):
            raise ValueError(f'All real {kind} outputs are empty')
    inp=json.loads((run/'input_manifest.json').read_text())
    result=dict(status='functional_smoke_passed',training_updates=len(metrics),generated_outputs=len(values),
        architecture='qformer',model_name=cfg['model_name'],frame_size=cfg['frame_size'],
        init_adapter_sha256=digest(Path(cfg['init_adapter'])/'adapter_model.safetensors'),
        code_commit=(run/'code_commit.txt').read_text().strip(),source_sha256=inp['source_sha256'],
        implementation_sha256={name:digest(Path(__file__).parent/name)
                               for name in ['model.py','qformer.py','short_encoding.py','train_overfit_ab.py']},
        qformer_config=qformer_config,
        truncations=sum(r['stop_reason']=='max_new_tokens' for r in values),
        empty_outputs=sum(not r['prediction'].strip() for r in values),
        compressor_updates=gradient,quality_approved=False,
        interpretation='Functional smoke only: not clinical quality, convergence or generalization.')
    (run/'smoke_report.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    print(json.dumps(check(p.parse_args().run_dir),indent=2))
