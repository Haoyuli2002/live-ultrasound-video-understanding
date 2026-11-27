import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from lavis.models.blip2_models.Qformer import BertConfig
from StreamReady.tests.test_qwen35_backbone import tiny_backbone

from .actions import ACTIONS, install_actions
from .answer import AnswerDecoder, DecisionContext
from .checkpoint import restore, save
from .index import EventIndex
from .qformer import EventQFormers
from .retrieve import QwenRelevance
from .train import batch_step, batch_loss, decision_loss, joint_optimizer, run_split
from .progress import TrainingProgress


def components():
    bb = tiny_backbone()
    vocab = {f'w{i}': i for i in range(64)}
    for i, token in {0:'[PAD]', 1:'[UNK]', 2:'[EOS]', 3:'Yes', 4:'No', 5:'WAIT', 6:'ANSWER',
                     7:'<', 8:'>', 60:'[IMAGE]', 62:'[START]', 63:'[END]'}.items():
        del vocab[f'w{i}']
        vocab[token] = i
    raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', eos_token='[EOS]',
                                  pad_token='[PAD]', extra_special_tokens=['[IMAGE]', '[START]', '[END]'])
    tok.chat_template = ("{% for msg in messages %}{% if msg.content is string %}{{ msg.content }}"
                         "{% else %}[START][IMAGE][END] {{ msg.content[1].text }}{% endif %}{% endfor %} w9")
    bb.tokenizer = tok
    decoder = AnswerDecoder(bb)
    cfg = BertConfig(vocab_size=16, hidden_size=24, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=32, encoder_width=12, add_cross_attention=True, cross_attention_freq=2)
    return bb, EventQFormers(cfg, vision_width=32, llm_width=32), decoder


class JointTrainingTests(unittest.TestCase):
    def test_progress_tracks_real_ga_updates_without_changing_training(self):
        torch.manual_seed(123)
        ctx = DecisionContext('w10', 0, 4, 0, 4)
        prepared = [(torch.randn(5, 4, 32), torch.randn(1, 2, 4, 32),
                     torch.ones(1, 2, dtype=torch.bool), ctx, target)
                    for target in ['<WAIT>', '<ANSWER> w11', '<WAIT>', '<ANSWER> w12 w13', '<WAIT>']]
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'video.mp4';video.touch()
            rows = [dict(video_id='v', video_path=str(video), current_time=4, key=i) for i in range(5)]
            def execute(track):
                torch.manual_seed(42)
                bb, towers, decoder = components();decoder.token_mean_ntp = True
                opt = joint_optimizer(bb, towers, 1e-4, 1e-5)
                progress = TrainingProgress(Path(tmp)/'progress', examples=5, batch_size=2, world_size=1,
                    accumulation=2, epochs=1, holdout=True, emit=lambda *a,**kw:None) if track else None
                if track:progress.__enter__();progress.start_epoch(1)
                with patch('event_retrieval.train.EventIndex', return_value=SimpleNamespace(eof=5,close=lambda:None)), \
                     patch('event_retrieval.train.index_source', return_value='fixture'), \
                     patch('event_retrieval.train.prepare_decision', side_effect=lambda index,row,*a,**kw:prepared[row['key']]):
                    value = run_split(rows, Path(tmp), Path(tmp), towers, decoder, opt, seed=9,
                                      batch_size=2, gradient_accumulation=2, progress=progress)
                    if track:
                        self.assertEqual(progress.step,2)
                        self.assertTrue(all(v['step'].item()==2 for v in opt.state.values()))
                        progress.phase('holdout')
                        run_split(rows, Path(tmp), Path(tmp), towers, decoder, batch_size=2, progress=progress)
                        self.assertEqual(progress.step,2)
                        self.assertEqual(progress.holdout_processed,5)
                        progress.finish()
                return value, [p.detach().clone() for module in [bb,towers] for p in module.parameters()]
            baseline, expected = execute(False);observed, actual = execute(True)
            self.assertEqual(baseline,observed)
            for a,b in zip(expected,actual,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
            records = [json.loads(line) for line in (Path(tmp)/'progress/progress.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['global_step']==2 and r['phase']=='holdout' for r in records))

    def test_direct_recent_frames_are_chronological_and_disjoint(self):
        bb, towers, decoder = components()
        towers.set_raw_recent_frames(4)
        towers.eval()
        decoder.separate_action_loss = True
        lengths = [1, 4, 5, 24]
        recent = torch.randn(len(lengths), 24, 64, bb.llm_width)
        valid = torch.arange(24)[None] < torch.tensor(lengths)[:, None]
        recent[~valid] = float("nan")
        events = torch.randn(len(lengths), 1, 2, 64, bb.llm_width)
        event_mask = torch.ones(events.shape[:3], dtype=torch.bool)
        event_mask[0] = False
        prefix, mask = towers(recent, events, recent_mask=valid, event_mask=event_mask)
        self.assertEqual(prefix.shape[1], 416)
        self.assertTrue(torch.isfinite(prefix).all())
        self.assertFalse(mask[0, :128].any())
        for i, n in enumerate(lengths):
            start = max(0, n - 4)
            raw = prefix[i, 160:][mask[i, 160:]]
            torch.testing.assert_close(raw, recent[i, start:n].flatten(0, 1))
            self.assertEqual(mask[i, 128:160].sum().item(), 32 if start else 0)
            if start:
                summary, _ = towers._encode(towers.short, recent[i:i+1, None, :start],
                                            None, history=False)
                torch.testing.assert_close(prefix[i, 128:160], summary[0])
        ctx = DecisionContext('w10', 0, 23, 0, 23)
        loss = decoder.alignment_loss(prefix[-1:], mask[-1:], [ctx], ['<ANSWER> w11'],
                                      average_endpoints=True)
        loss.backward()
        for branch in (towers.short, towers.long):
            self.assertGreater(branch.queries.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn(decoder.generate(prefix[-1], mask[-1], ctx, 5)['action'], ('WAIT', 'ANSWER'))

    def test_accumulated_ntp_update_matches_full_batch(self):
        targets = ['<WAIT>', '<ANSWER> w11 w12', '<WAIT> w13', '<WAIT>', '<ANSWER> w14']
        context = DecisionContext('w10', 0, 4, 0, 4)
        rows = [(torch.randn(5, 4, 32), torch.randn(1, 2, 4, 32),
                 torch.ones(1, 2, dtype=torch.bool), context, target) for target in targets]
        def updates(accumulate):
            torch.manual_seed(42)
            bb, towers, decoder = components()
            towers.eval()
            bb.eval()
            decoder.token_mean_ntp = True
            params = [p for group in (bb.parameters(), towers.parameters()) for p in group if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.001)
            sums = torch.zeros(2)
            for group in (rows[:4], rows[4:]):
                start = sums[1].clone()
                if accumulate:
                    for offset in range(0, len(group), 2):
                        batch_step(group[offset:offset+2], towers, decoder, optimizer, loss_sums=sums,
                                   accumulation_start=start, finish_update=offset+2 >= len(group))
                else:
                    batch_step(group, towers, decoder, optimizer, loss_sums=sums)
            return [p.detach().clone() for p in params], sums
        full, full_loss = updates(False)
        micro, micro_loss = updates(True)
        torch.testing.assert_close(full_loss, micro_loss)
        for expected, actual in zip(full, micro, strict=True):
            torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)

    def test_plain_ntp_averages_all_supervised_tokens(self):
        bb, towers, decoder = components()
        decoder.token_mean_ntp = True
        towers.set_raw_recent_frames(4)
        towers.eval()
        ctx = DecisionContext('w10', 0, 4, 0, 4)
        rows = []
        for target in ('<WAIT>', '<ANSWER> w11 w12'):
            rows.append((torch.randn(5, 4, 32), torch.randn(1, 2, 4, 32),
                         torch.ones(1, 2, dtype=torch.bool), ctx, target))
        captures = []
        hook = bb.model.get_output_embeddings().register_forward_hook(lambda m, x, y: captures.append(y))
        sums = torch.zeros(2)
        try:
            loss = batch_loss(rows, towers, decoder, endpoint_weights=[1, 1], loss_sums=sums)
        finally:
            hook.remove()
        gold = torch.tensor([*decoder.format.tags['WAIT'], *decoder.format.tags['ANSWER'],
                             *bb.tokenizer.encode(' w11 w12', add_special_tokens=False), bb.tokenizer.eos_token_id])
        reference = torch.nn.functional.cross_entropy(captures[0].float(), gold)
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(sums[0], reference.detach() * len(gold))
        self.assertEqual(sums[1].item(), len(gold))
        torch.testing.assert_close(torch.autograd.grad(loss, captures[0], retain_graph=True)[0],
                                   torch.autograd.grad(reference, captures[0])[0])

    def test_batched_mixed_windows_match_endpoint_means(self):
        torch.manual_seed(3)
        bb, towers, decoder = components()
        towers.eval()
        inputs = []
        for n, slots, target in [(2, 0, '<WAIT>'), (5, 2, '<ANSWER> w11 w12')]:
            recent = torch.randn(n, 4, 32)
            history = torch.randn(slots, 3 if slots else 0, 4, 32)
            valid = torch.ones(history.shape[:2], dtype=torch.bool)
            ctx = DecisionContext('w10', 20, 25, 26 - n, 25)
            inputs.append((recent, history, valid, ctx, target))
        singles = [batch_loss([sample], towers, decoder) for sample in inputs]
        params = [towers.short.queries, towers.long.queries]
        # Include a long wait and a final partial batch of size one.
        for weights in (None, [1 / 4, 38 / 4]):
            with self.subTest(weights=weights):
                actual = batch_loss(inputs, towers, decoder, endpoint_weights=weights)
                expected = sum(w * loss for w, loss in zip(weights or [1, 1], singles)) / len(inputs)
                torch.testing.assert_close(actual, expected)
                grads = torch.autograd.grad(actual, params)
                refs = torch.autograd.grad(expected, params, retain_graph=True)
                for g, ref in zip(grads, refs):
                    torch.testing.assert_close(g, ref, atol=2e-6, rtol=1e-4)
        single = batch_loss(inputs[1:], towers, decoder, endpoint_weights=[38 / 4])
        torch.testing.assert_close(single, singles[1] * (38 / 4))

    def test_action_weight_does_not_scale_body_gradients(self):
        torch.manual_seed(3)
        bb, towers, decoder = components()
        decoder.separate_action_loss = True
        towers.eval()
        joint_optimizer(bb, towers, 1e-4, 1e-5)
        ctx = DecisionContext('w10', 20, 25, 24, 25)
        features = (torch.randn(2, 4, 32), torch.randn(1, 2, 4, 32),
                    torch.ones(1, 2, dtype=torch.bool), ctx)
        for targets, weights in [(['<WAIT>', '<ANSWER> w11 w12'], [1., 38.]),
                                 (['<WAIT>'], [1.]), (['<WAIT> w11'], [1.]),
                                 (['<ANSWER> w11 w12'], [38.])]:
            with self.subTest(targets=targets):
                logits = []
                hook = bb.model.get_output_embeddings().register_forward_hook(
                    lambda module, args, output: logits.append(output))
                sums = torch.zeros(2)
                try:
                    loss = batch_loss([(*features, target) for target in targets], towers, decoder,
                                      endpoint_weights=weights, loss_sums=sums)
                finally:
                    hook.remove()
                offset = 0
                action = logits[0].new_zeros(())
                body = logits[0].new_zeros(())
                for target, weight in zip(targets, weights):
                    tag, *words = target.split()
                    ids = decoder.format.tags[tag[1:-1]].copy()
                    if words:
                        ids += bb.tokenizer.encode(' ' + ' '.join(words), add_special_tokens=False)
                        ids.append(bb.tokenizer.eos_token_id)
                    ce = torch.nn.functional.cross_entropy(logits[0][offset:offset + len(ids)],
                                                           torch.tensor(ids), reduction='none')
                    action = action + weight * ce[0]
                    if words:
                        body = body + ce[1:].mean()
                    offset += len(ids)
                reference = (action + body) / len(targets)
                torch.testing.assert_close(sums, torch.stack([action, body]).detach())
                torch.testing.assert_close(loss, reference)
                torch.testing.assert_close(torch.autograd.grad(loss, logits, retain_graph=True)[0],
                                           torch.autograd.grad(reference, logits)[0])

    def test_actions_joint_update_restore_and_generate(self):
        torch.manual_seed(17)
        bb, towers, decoder = components()
        ids = [bb.tokenizer.encode(s, add_special_tokens=False) for s in ACTIONS]
        self.assertEqual(ids, [[64], [65]])
        self.assertEqual(install_actions(bb), dict(zip(ACTIONS, [64, 65])))
        params = [towers.short.queries, towers.long.queries,
                  next(bb.model.model.language_model.layers.parameters()),
                  bb.model.get_input_embeddings().weight, bb.model.get_output_embeddings().weight]
        initial = [p.detach().clone() for p in params]
        opt = joint_optimizer(bb, towers, 1e-3, 1e-3)
        index = EventIndex(':memory:', 'fixture', 'tiny')
        self.addCleanup(index.close)
        for sec in range(27):
            frame = torch.zeros(4, 32)
            frame[:, int(sec >= 4)] = 1
            index.push(sec, float(sec), frame)
        row = dict(question='w10', current_time=26, query_time=25, ready=False, answer='')
        loss = decision_loss(index, row, towers, decoder, QwenRelevance(bb))
        row.update(ready=True, answer='w11')
        loss = loss + decision_loss(index, row, towers, decoder, QwenRelevance(bb))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        opt.step()
        for p, old in zip(params, initial):
            self.assertFalse(torch.equal(p, old))
        for p, old in zip(params[-2:], initial[-2:]):
            self.assertTrue((p[64:66] != old[64:66]).any(dim=1).all())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in bb.model.model.visual.parameters()))

        root = Path('/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/event_retrieval')
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / 'latest.pt'
            save(path, towers, bb, opt, 1, {'model': bb.model_path})
            other, towers2, dec2 = components()
            opt2 = joint_optimizer(other, towers2, 1e-3, 1e-3)
            self.assertEqual(restore(path, towers2, other, opt2, {'model': bb.model_path}), 1)
            self.assertTrue(opt2.state)
            torch.testing.assert_close(other.model.get_output_embeddings().weight, bb.model.get_output_embeddings().weight)
            torch.testing.assert_close(towers2.long.queries, towers.long.queries)
            visual, mask = towers2.eval().encode_short(torch.randn(1, 3, 4, 32))
            ctx = DecisionContext('w10', 25, 26, 3, 26)
            self.assertEqual(dec2.generate(visual[0], mask[0], ctx, 9),
                             decoder.generate(visual[0], mask[0], ctx, 9))
