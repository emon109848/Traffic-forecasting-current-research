"""CPU tests for the GPU command's probe and truthful failure reporting."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from traffic_benchmark.evaluation.feasibility import optimizer_probe, sweep_batches


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.]))

    def forward(self, x, calendar):
        return self.weight * x


class FeasibilityTests(unittest.TestCase):
    def test_one_optimizer_step_changes_disposable_cpu_model(self):
        model = ToyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, foreach=False)
        report = optimizer_probe(model, torch.ones((1,2,1,1)), None, optimizer)
        self.assertEqual(report['optimizer_steps'], 1)
        self.assertTrue(report['checked_parameter_changed'])
        self.assertEqual(report['output_shape'], [1,2,1,1])
        self.assertEqual(int(optimizer.state[model.weight]['step']), 1)

    def test_nonfinite_output_fails_before_update(self):
        model = ToyModel()
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, 'Nonfinite prediction'):
            optimizer_probe(model, torch.full((1,1,1,1),float('nan')), None, optimizer)
        self.assertEqual(model.weight.item(),2)
        self.assertEqual(len(optimizer.state),0)

    def test_missing_gradient_fails_before_update(self):
        model = ToyModel()
        model.unused = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError,'gradient'):
            optimizer_probe(model,torch.ones((1,1,1,1)),None,optimizer)
        self.assertEqual(model.weight.item(),2)

    def test_oom_stops_sweep_and_preserves_prior_success(self):
        calls=[]
        def run(batch):
            calls.append(batch)
            return {'batch_size':batch,'status':'passed' if batch==1 else 'oom'}
        report=sweep_batches(run)
        self.assertEqual(calls,[1,2])
        self.assertEqual(report['largest_passed_batch'],1)
        self.assertFalse(report['all_requested_passed'])
        self.assertEqual(report['results'][-1]['status'],'not_attempted')

    def test_generic_failure_does_not_masquerade_as_oom(self):
        report=sweep_batches(lambda batch:{'batch_size':batch,'status':'failed'})
        self.assertIsNone(report['largest_passed_batch'])
        self.assertEqual([r['status'] for r in report['results']],['failed','not_attempted','not_attempted'])

    def test_all_pass_and_malformed_reports(self):
        report=sweep_batches(lambda batch:{'batch_size':batch,'status':'passed'})
        self.assertTrue(report['all_requested_passed'])
        self.assertEqual(report['largest_passed_batch'],4)
        with self.assertRaises(ValueError):
            sweep_batches(lambda batch:{'batch_size':batch,'status':'maybe'})
        with self.assertRaises(ValueError):
            sweep_batches(lambda batch:{'batch_size':batch,'status':'passed'},[2,1])

    def test_unavailable_cuda_cli_writes_no_measurements_and_refuses_overwrite(self):
        spec=importlib.util.spec_from_file_location('gpu_check',Path('scripts/check_patchstg_gpu.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'gpu'
            argv=['check_patchstg_gpu.py','--output-dir',str(output)]
            with patch('sys.argv',argv),patch.object(module.torch.cuda,'is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(),2)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'],'cuda_unavailable')
            self.assertEqual([r['status'] for r in report['results']],['not_attempted']*3)
            self.assertIsNone(report['largest_passed_batch'])
            self.assertNotIn('peak_allocated_bytes',report)
            with patch('sys.argv',argv),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
                module.main()
            self.assertEqual(exc.exception.code,2)


if __name__=='__main__':
    unittest.main()
