import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .holdout import split_videos,plot_curves
from .train_full import schedules


class HoldoutTests(unittest.TestCase):
    def test_holdout_is_absent_from_all_optimizer_schedules(self):
        blocks=[dict(block_window=[0,60],local_sub_summaries=[dict(local_summary_target='x') for _ in range(6)],block_summary_target='b',global_summary_target='g')]
        videos={str(i):blocks for i in range(196)}
        selected=list(map(str,range(16)))
        train,holdout=split_videos(videos,selected)
        self.assertEqual((len(train),len(holdout)),(180,16))
        for epoch in range(3):
            tasks=[t for rank in schedules(train,4,42,epoch) for t in rank]
            self.assertEqual(len(tasks),180*8)
            self.assertFalse(set(selected)&{t['video_id'] for t in tasks})

    def test_duplicate_unknown_and_full_holdout_are_rejected(self):
        for ids in (['a','a'],['missing'],['a','b'],[]):
            with self.assertRaises(ValueError):split_videos({'a':[], 'b':[]},ids)

    def test_plots_contain_train_and_holdout_and_quality_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'holdout/step-00001000';folder.mkdir(parents=True)
            (root/'training_metrics.rank0.jsonl').write_text(json.dumps(dict(global_step=1000,loss=2.0))+'\n')
            report=dict(global_step=1000,loss=2.5,scores={f'real/{k}':dict(rouge_l_f1=.2) for k in ('short','block','global')})
            (folder/'report.json').write_text(json.dumps(report))
            plot_curves(root)
            for f in ['loss_curves.png','loss_curves.svg','holdout_quality.png','holdout/history.json']:
                self.assertGreater((root/f).stat().st_size,0)

    def test_monitor_restores_training_mode_without_changing_weights(self):
        from contextlib import nullcontext
        import torch
        from .holdout import monitor
        class Memory:
            def __init__(self):
                self.weight=torch.nn.Parameter(torch.tensor(2.0));self.training=True;self.reset=False
            def eval(self):self.training=False
            def train(self):self.training=True
            def reset_short_memory(self):self.reset=True
            def reconstruction_loss(self,memory,reference,kind):
                assert not self.training
                return self.weight.square()
        kinds=['short']*6+['block','global']
        records=[dict(kind=k,window=[i*10,(i+1)*10],memory=torch.ones(1,10,1),reference='vessel') for i,k in enumerate(kinds)]
        blocks=[dict(block_window=[0,60],local_sub_summaries=[{}]*6)]
        generated=dict(prediction='vessel',stop_reason='eos')
        with tempfile.TemporaryDirectory() as d, \
             patch('stage2.data.VideoReader'), \
             patch('stage2.reconstruct.collect_memories',return_value=records), \
             patch('stage2.reconstruct.greedy_reconstruct',return_value=generated) as decode, \
             patch('torch.cuda.current_device',return_value=0), \
             patch('torch.random.fork_rng',return_value=nullcontext()), \
             patch('stage2.holdout.record_gpu_memory'), \
             patch('stage2.holdout.plot_curves'):
            memory=Memory();monitor(memory,{'v':blocks},{'v':'unused'},d,1000,0,1)
            self.assertTrue(memory.training and memory.reset)
            self.assertEqual(memory.weight.item(),2.0)
            self.assertIsNone(memory.weight.grad)
            self.assertEqual(decode.call_count,8)
            report=json.loads((Path(d)/'holdout/step-00001000/report.json').read_text())
            self.assertEqual(report['targets'],8)
            self.assertEqual(report['loss'],4.0)
