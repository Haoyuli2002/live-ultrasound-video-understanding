import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .monitoring import evaluation_due,token_mean_answer_ce
from .dashboard import Monitor
from .gpu_stats import read_samples,summarize_devices


class StepMonitoringTests(unittest.TestCase):
    def test_50_steps_plus_final_without_epoch_duplicates(self):
        cfg=dict(epochs=2,eval_epochs=[1,2],eval_every_steps=50)
        calls=[0];step=0
        for epoch in (1,2):
            for _ in range(99):
                step+=1
                if evaluation_due(cfg,step,epoch,last_step=calls[-1]):calls.append(step)
            if evaluation_due(cfg,step,epoch,epoch_end=True,last_step=calls[-1]):calls.append(step)
        self.assertEqual(calls,[0,50,100,150,198])
        self.assertFalse(evaluation_due(cfg,200,2,epoch_end=True,last_step=200))
        self.assertTrue(evaluation_due(dict(epochs=2,eval_epochs=[1,2]),99,1,epoch_end=True,last_step=0))

    def test_holdout_ce_weights_answer_tokens_not_sequences(self):
        records=[dict(answer_token_count=2,answer_token_nll=4),dict(answer_token_count=6,answer_token_nll=24),{}]
        self.assertEqual(token_mean_answer_ce(records),3.5)
        self.assertEqual(token_mean_answer_ce([{}]),0)

    def test_dashboard_separates_devices_and_real_step_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'logs').mkdir()
            (p/'experiment.json').write_text(json.dumps(dict(expected_optimizer_steps=198,world_size=2,epochs=2,holdout={'directory':'holdout'})))
            (p/'data_audit.json').write_text(json.dumps(dict(train=dict(videos=171,qa=5236),holdout=dict(videos=16,qa=464))))
            (p/'logs/gpu.csv').write_text('timestamp, uuid, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%], utilization.memory [%], power.draw [W]\n'
              '2026/11/23 12:00:00.000, GPU-a, A100, 40960 MiB, 81920 MiB, 10 %, 4 %, 100 W\n'
              '2026/11/23 12:00:00.000, GPU-b, A100, 20480 MiB, 81920 MiB, 90 %, 8 %, 200 W\n')
            m=Monitor({'x':p})
            with patch.object(m,'scheduler',return_value={'state':'RUNNING'}):s=m.snapshot('x')
            self.assertEqual(s['planned_steps'],198)
            self.assertEqual(s['gpu_devices']['GPU-a']['last_60s_mean_util'],10)
            self.assertEqual(s['gpu_devices']['GPU-b']['last_60s_mean_util'],90)
            self.assertEqual(s['holdout_data']['qa'],464)
            stats=summarize_devices(read_samples(p/'logs/gpu.csv'))
            self.assertEqual(stats['GPU-a']['memory_occupancy_peak_pct'],50)
            self.assertEqual(stats['GPU-b']['memory_occupancy_peak_pct'],25)

if __name__=='__main__':unittest.main()
