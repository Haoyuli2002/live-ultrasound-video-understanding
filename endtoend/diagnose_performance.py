"""Read-only CPU/CUDA profile of one warmed-up endpoint batch."""
import functools
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from . import train


def main():
    run = Path(sys.argv[sys.argv.index('--run-dir') + 1])
    cfg = json.loads((run / 'experiment.json').read_text())
    if not (cfg.get('profile_only') and cfg.get('restore_optimizer') and
            cfg.get('profile_warmups') == 1 and cfg.get('profile_repeats') == 1 and
            len(cfg['batch_candidates']) == 1):
        raise ValueError('Requires read-only profiling, one warmup and one timed batch')
    from lavis.models.streamready.module import StreamReadyModule
    from lavis.models.streamready.visual_memory import VisualMemoryTree
    from lavis.models.blip2_models.streamready import StreamReady

    def annotate(cls, name, label):
        original = getattr(cls, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with record_function(label):
                return original(*args, **kwargs)

        setattr(cls, name, wrapped)

    annotate(StreamReadyModule, 'forward', 'phase/core_sample')
    annotate(VisualMemoryTree, 'update', 'phase/memory_update')
    annotate(VisualMemoryTree, '_rebuild_prototypes', 'phase/prototype_rebuild')
    annotate(StreamReadyModule, '_reason_once', 'phase/qformer_reasoning')
    annotate(StreamReady, '_reasoning_loss', 'phase/batched_llm_loss')
    annotate(StreamReady, '_readiness_surrogate', 'phase/readiness_gradient')
    original_forward, original_backward = train.forward_batch, train.backward_objectives
    state = dict(calls=0, profiler=None)

    def forward(*args, **kwargs):
        state['calls'] += 1
        if state['calls'] == 2:
            torch.cuda.synchronize()
            state['profiler'] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            state['profiler'].start()
            state['started'] = time.perf_counter()
        with record_function('phase/forward_batch'):
            return original_forward(*args, **kwargs)

    def backward(*args, **kwargs):
        with record_function('phase/backward_objectives'):
            result = original_backward(*args, **kwargs)
        if state['profiler'] is not None:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - state['started']
            prof = state['profiler']
            prof.stop()
            events = []
            for event in prof.key_averages():
                events.append(dict(name=event.key, count=event.count,
                                   cpu_total_ms=event.cpu_time_total / 1000,
                                   cpu_self_ms=event.self_cpu_time_total / 1000,
                                   device_total_ms=event.device_time_total / 1000,
                                   device_self_ms=event.self_device_time_total / 1000))
            report = dict(profiled_batch_size=cfg['batch_candidates'][0],
                          profiled_wall_seconds=elapsed, optimizer_updates=0,
                          scope='one synthetic worst-case ANSWER batch, forward and isolated backward; excludes H2D input transfer, optimizer, evaluation, file saving',
                          limitation='Profiler adds overhead; nested scopes overlap and must not be summed.',
                          phases=[e for e in events if e['name'].startswith('phase/')],
                          top_cpu_self=sorted(events, key=lambda e: e['cpu_self_ms'], reverse=True)[:30],
                          top_device_self=sorted(events, key=lambda e: e['device_self_ms'], reverse=True)[:20],
                          synchronization=[e for e in events if any(x in e['name'].lower() for x in ('synchron', '_local_scalar', 'aten::item', 'memcpy'))])
            (run / 'performance_diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
            prof.export_chrome_trace(str(run / 'trace.json'))
            print('PERFORMANCE_DIAGNOSIS', json.dumps(report), flush=True)
            state['profiler'] = None
        return result

    train.forward_batch, train.backward_objectives = forward, backward
    train.main()


if __name__ == '__main__':
    main()
