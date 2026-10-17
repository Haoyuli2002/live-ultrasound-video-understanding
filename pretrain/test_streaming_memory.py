"""CPU tests for chronological state carry, optimizer schedule, and detach boundaries."""
import unittest
import torch
from pretrain.streaming_memory import prepare_blocks, train_stream


def block(vid, index):
    start = index * 60
    return dict(video_id=vid, block_window=[start, start+60],
        block_summary_target='block', global_summary_target='global',
        local_sub_summaries=[dict(window=[start+i*10,start+(i+1)*10],
            local_summary_target='short') for i in range(6)])


class StreamingTests(unittest.TestCase):
    def test_chronology_gradients_and_reset(self):
        weight = torch.nn.Parameter(torch.tensor(0.1))
        optimizer = torch.optim.SGD([weight], lr=0.0001)
        reports, prior, short_calls, states = [], [], [], []
        def encode(row, second):
            state = weight * (1 + second/1000)
            state.retain_grad()
            states.append(state)
            return state
        def decode_short(memory, target, task):
            if task == 'block_summary':
                self.assertEqual(len(memory),60)
                self.assertTrue(all(not x.requires_grad for x in memory[:50]))
                self.assertTrue(all(x.requires_grad for x in memory[50:]))
            else:
                self.assertEqual(len(memory),10)
            short_calls.append(task)
            return (torch.stack(memory).mean() + weight).square()
        def compress(short, previous):
            prior.append(len(previous))
            self.assertTrue(all(not x.requires_grad for x in previous))
            value = torch.stack(short).mean() + weight
            if previous:
                value = value + torch.stack(previous).mean()
            return [value]*60
        def backward(loss):
            old_grads = [None if x.grad is None else x.grad.clone() for x in states[:-10]]
            loss.backward()
            for x, old in zip(states[:-10], old_grads):
                self.assertIsNotNone(old)
                torch.testing.assert_close(x.grad, old)
            self.assertTrue(all(x.grad is not None for x in states[-10:]))
        def report(vid, time, losses, total):
            expected = {'short','local','global'} if time%60 == 0 else {'short'}
            self.assertEqual(set(losses),expected)
            torch.testing.assert_close(total, sum(losses.values()))
            reports.append((vid,time))
        rows = list(prepare_blocks([block('a',2),block('b',0),block('a',0),block('a',1)]))
        count = train_stream(rows, encode_second=encode, decode_short=decode_short,
            encode_long=compress, decode_global=lambda mem,target: (torch.stack(mem).mean()+weight).square(),
            optimizer=optimizer, backward=backward, clip_grad=lambda: None, report=report)
        self.assertEqual(count,24)
        self.assertEqual(prior,[0,60,60,0])
        self.assertEqual(reports[:6],[('a',i) for i in range(10,61,10)])
        self.assertEqual(short_calls.count('block_summary'),4)
        self.assertEqual(len(states),240)  # No history recomputation.

    def test_tail_only_trains_short_and_no_long_update(self):
        row = block('tail',0)
        row['block_window']=[0,20]
        row['local_sub_summaries']=row['local_sub_summaries'][:2]
        del row['block_summary_target']; del row['global_summary_target']
        rows=list(prepare_blocks([row]))
        weight=torch.nn.Parameter(torch.tensor(1.0))
        optimizer=torch.optim.SGD([weight],lr=0.01)
        reports=[]
        def no_long(*args):
            self.fail('Incomplete minute must not update or decode long memory')
        count=train_stream(rows,encode_second=lambda row,t: weight*1,
            decode_short=lambda states,target,task: torch.stack(states).mean().square(),
            encode_long=no_long,decode_global=no_long,optimizer=optimizer,
            backward=lambda loss: loss.backward(),clip_grad=lambda: None,
            report=lambda vid,t,losses,total: reports.append((t,set(losses))))
        self.assertEqual(count,2)
        self.assertEqual(reports,[(10,{'short'}),(20,{'short'})])

    def test_query_token_positions_are_unambiguous(self):
        from pretrain.memory_collator import long_encode_messages, LONG_MEMORY_TOKEN, SHORT_MEMORY_TOKEN
        messages=long_encode_messages(60,60,60)
        self.assertNotIn(LONG_MEMORY_TOKEN,messages[0]['content'])
        text=messages[1]['content'][0]['text']
        self.assertEqual(text.count(LONG_MEMORY_TOKEN),120)
        self.assertEqual(text.count(SHORT_MEMORY_TOKEN),60)

    def test_invalid_annotations_fail(self):
        for rows in [[block('a',1)], [block('a',0),block('a',0)],
                     [block('a',0),block('a',2)]]:
            with self.assertRaises(ValueError): list(prepare_blocks(rows))
        for key in ['local_sub_summaries','block_summary_target','global_summary_target']:
            row = block('a',0); del row[key]
            with self.assertRaises(ValueError): list(prepare_blocks([row]))
        row = block('a',0); row['local_sub_summaries'][0]['window']=[0,11]
        with self.assertRaises(ValueError): list(prepare_blocks([row]))


if __name__ == '__main__':
    unittest.main()
