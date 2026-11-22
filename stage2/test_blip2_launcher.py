"""Regression for torchrun consuming the overfit worker's --run option."""
import contextlib
import io
from pathlib import Path
import shlex
import unittest

from torch.distributed.run import get_args_parser
from .blip2_memory import ARCHITECTURE
from .blip2_overfit import build_parser


class Blip2LauncherTests(unittest.TestCase):
    def batch_args(self, arm):
        path = Path(__file__).resolve().parents[1] / 'scripts/slurm/run_stage2_blip2_ablation.sbatch'
        line = next(line.strip() for line in path.read_text().splitlines() if line.strip().startswith('torchrun '))
        return [part.replace('$RUN_DIR', '/tmp/frozen run').replace('$arm', arm)
                for part in shlex.split(line)[1:]]

    def test_both_arms_survive_actual_batch_command_parsing(self):
        for arm in (ARCHITECTURE, 'qformer'):
            with self.subTest(arm=arm):
                outer = get_args_parser().parse_args(self.batch_args(arm))
                self.assertTrue(outer.module)
                self.assertEqual(outer.nproc_per_node, '4')
                self.assertEqual(outer.training_script, 'stage2.blip2_overfit')
                worker = build_parser().parse_args(outer.training_script_args)
                self.assertEqual(worker.run, Path('/tmp/frozen run'))
                self.assertEqual(worker.arm, arm)

    def test_separator_also_protects_legacy_worker_argument(self):
        args = ['--run' if part == '--run-dir' else part for part in self.batch_args('qformer')]
        outer = get_args_parser().parse_args(args)
        self.assertEqual(build_parser().parse_args(outer.training_script_args).run, Path('/tmp/frozen run'))

    def test_old_launcher_reproduces_observed_ambiguity(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            get_args_parser().parse_args(['--standalone', '--nproc_per_node=4', '-m',
                                         'stage2.blip2_overfit', '--run', '/tmp/frozen', '--arm', 'qformer'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('ambiguous option: --run', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
