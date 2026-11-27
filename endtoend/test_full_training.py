import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .feature_cache import VideoFeatureCache
from .full_data import batch_plan,resolve_recipe,validate_official_split
from .monitoring import evaluation_state,validate_split_separation


def rows(video,split,labels=(0,0,1)):
    return [dict(video_id=video,task_id=video,sample_id=f'{video}-{i}',split=split,
                 video_path=f'/data/{video}.mp4',current_time=float(i),ready=y) for i,y in enumerate(labels)]


class FullTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_mmap_cache_preserves_exact_endpoint_frames_and_lru(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=VideoFeatureCache(tmp,max_open_videos=1)
            a=torch.arange(6*2*3,dtype=torch.float32).view(1,6,2,3)
            cache.add_video('a',a,{'a1':[0,1,3],'a2':[0,1,2,4],'prefix':[0,1,2]})
            cache.add_video('b',a+100,{'b1':[0,5]})
            prefix = cache['prefix']
            self.assertEqual(prefix.data_ptr(), cache.open['a'].data_ptr())
            torch.testing.assert_close(prefix, a[:, :3])
            torch.testing.assert_close(cache['a1'],a[:,[0,1,3]])
            torch.testing.assert_close(cache['a2'],a[:,[0,1,2,4]])
            torch.testing.assert_close(cache['b1'],(a+100)[:,[0,5]])
            self.assertEqual(list(cache.open),['b'])
            torch.testing.assert_close(cache['a1'],a[:,[0,1,3]])
            self.assertEqual(len(list(Path(tmp).glob('*.pt'))),2)
            cache.write_manifest()
            with self.assertRaises(ValueError):cache.add_video('a',a,{'new':[0]})
            with self.assertRaises(ValueError):cache.add_video('c',a,{'a1':[0]})

    def test_mmap_and_legacy_real_preprocessing_match(self):
        from types import SimpleNamespace
        from .train import prepare_features
        root=Path(__file__).resolve().parents[1]
        video=root/'cluster_data/videos/train_full295/P3FoSFIrfQA.mp4'
        if not video.is_file():self.skipTest('Server video unavailable')
        def encode(pixels):
            # Depend on actual pixels so a frame-selection regression is visible.
            values=pixels.mean(dim=(1,3,4))
            return values[...,None,None].expand(-1,-1,2,3).contiguous()
        model=SimpleNamespace(backbone=SimpleNamespace(encode_frames=encode))
        examples=[dict(video_id='v',video_path=str(video),sample_id='x',current_time=2.3),
                  dict(video_id='v',video_path=str(video),sample_id='y',current_time=3.)]
        with tempfile.TemporaryDirectory() as tmp,patch.object(torch.Tensor,'cuda',lambda x:x):
            p=Path(tmp);(p/'old').mkdir();(p/'new').mkdir()
            old=prepare_features(model,examples,p/'old',256,2)
            new=prepare_features(model,examples,p/'new',256,2,'video_mmap')
            for e in examples:torch.testing.assert_close(old[e['sample_id']],new[e['sample_id']])
            self.assertEqual(json.loads((p/'old/frame_audit.json').read_text()),json.loads((p/'new/frame_audit.json').read_text()))

    def test_split_leakage_rejected(self):
        train=rows('a','train');holdout=rows('b','holdout')
        official=dict(train=['a'],holdout=['b'],test=['c'])
        validate_split_separation(train,holdout);validate_official_split(train,holdout,official)
        with self.assertRaises(ValueError):validate_split_separation(train,rows('a','holdout'))
        bad=[dict(r,video_path='/data/a.mp4') for r in holdout]
        with self.assertRaises(ValueError):validate_split_separation(train,bad)
        with self.assertRaises(ValueError):validate_official_split(rows('c','train'),holdout,official)
        with self.assertRaises(ValueError):validate_split_separation(train,rows('b','test'))

    def test_update_budget_and_complete_question_groups(self):
        train=rows('a','train')+rows('b','train',(1,))+rows('c','train',(0,1))
        plan=batch_plan(train,4,10,17)
        self.assertEqual(plan['total_optimizer_steps'],sum(plan['optimizer_steps_per_epoch']))
        for epoch in plan['batch_order']:
            self.assertEqual(sorted(s for b in epoch for s in b),sorted(r['sample_id'] for r in train))
            self.assertEqual(sum(any(s.startswith('a-') for s in b) for b in epoch),1)
        with self.assertRaises(ValueError):batch_plan(train,2,10,17)

    def test_holdout_restores_mode_rng_and_has_no_gradients(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Dropout(.5)).train()
        params=[p.detach().clone() for p in model.parameters()]
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,'test cleanup'):
            with evaluation_state(model):
                self.assertFalse(model.training)
                self.assertFalse(model(torch.ones(1,3)).requires_grad)
                random.random();np.random.rand();torch.rand(4)
                raise RuntimeError('test cleanup')
        self.assertTrue(model.training);self.assertEqual(random.getstate(),py)
        np.testing.assert_array_equal(np.random.get_state()[1],npstate[1]);torch.testing.assert_close(torch.get_rng_state(),ts)
        for p,expected in zip(model.parameters(),params):
            self.assertIsNone(p.grad);torch.testing.assert_close(p,expected)

    def test_pending_recipe_refuses_training_then_resolves_without_checkpoint(self):
        from .data import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (root/'status.json').write_text('{"status":"prepared"}')
            cfg=dict(readiness_recipe_selection=str(root),model={'readiness_threshold':.5},init_checkpoint=None,restore_optimizer=False)
            with self.assertRaises(ValueError):resolve_recipe(cfg)
            selected=dict(readiness_objective='contrastive_bce_gold',timing_lambda_reg=.1,readiness_bce_positive_weight=1.,readiness_bce_weight=1.)
            (source/'experiment.json').write_text(json.dumps(selected))
            (root/'status.json').write_text('{"status":"completed"}')
            (root/'selection.json').write_text(json.dumps(dict(action='fresh_all_pairs_training',source=str(source),winner='hybrid1',threshold=.35)))
            (root/'selection_plan.json').write_text(json.dumps(dict(arms=[dict(name='hybrid1',run_dir=str(source),config_sha256=digest(source/'experiment.json'))])))
            actual=resolve_recipe(cfg)
            self.assertEqual(actual['timing_pair_mode'],'all_pairs');self.assertEqual(actual['readiness_bce_positive_weight'],1)
            self.assertEqual(actual['model']['readiness_threshold'],.35);self.assertIsNone(actual['init_checkpoint'])
            self.assertEqual(cfg['model']['readiness_threshold'],.5)
            (source/'experiment.json').write_text('{}')
            with self.assertRaises(ValueError):resolve_recipe(cfg)


if __name__=='__main__':unittest.main()
