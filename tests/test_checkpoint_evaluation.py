"""Frozen checkpoint selection, provenance, map integrity, and scoring behavior."""

from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows
from traffic_benchmark.data.spatial import build_spatial_maps
from traffic_benchmark.evaluation import MaskedForecastMetrics
from traffic_benchmark.evaluation.checkpoint import validate_checkpoint, restore_frozen_model
from traffic_benchmark.models import PatchSTG, PatchSTGConfig
from traffic_benchmark.training.trainer import evaluate


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.scaler = GlobalStandardScaler(2., 1.)
        self.hashes = {'flowsd.npz': 'data', 'sd_meta.csv': 'meta'}
        self.splits = {'train': (0, 8), 'val': (8, 12), 'test': (12, 16)}
        identity = {'source_sha256': self.hashes, 'splits': self.splits,
                    'scaler': asdict(self.scaler), 'implementation_sha256': {'file': 'hash'},
                    'spatial': {'source_sha256': self.hashes, 'scaler': asdict(self.scaler)}}
        self.report = {'status': 'completed', 'diagnostic_subset': False,
                       'identity': identity, 'best_epoch': 2, 'best_validation_mae': 1.,
                       'history': [{'epoch': e, 'validation': {'overall_pooled': {'mae': score}}}
                                   for e, score in ((1, 2.), (2, 1.), (3, 1.))]}
        self.checkpoint = {'kind': 'best', 'epoch': 2, 'validation_mae': 1.,
                           'scaler': asdict(self.scaler), 'identity': deepcopy(identity)}

    def check(self, checkpoint=None, report=None, hashes=None, splits=None, scaler=None):
        with patch('traffic_benchmark.evaluation.checkpoint.file_sha256', return_value='hash'):
            return validate_checkpoint(checkpoint or self.checkpoint, report or self.report,
                                       hashes or self.hashes, splits or self.splits, scaler or self.scaler)

    def test_first_validation_minimum_wins_tie_and_json_tuple_equivalence(self):
        self.checkpoint['identity']['splits'] = {k: list(v) for k, v in self.splits.items()}
        self.check()
        self.checkpoint['epoch'] = 3
        with self.assertRaisesRegex(ValueError, 'selection'):
            self.check()

    def test_reject_last_diagnostic_and_incomplete_runs(self):
        for key, value in (('status', 'running'), ('diagnostic_subset', True)):
            report = deepcopy(self.report); report[key] = value
            with self.assertRaises(ValueError):
                self.check(report=report)
        self.checkpoint['kind'] = 'last'
        with self.assertRaisesRegex(ValueError, 'best checkpoint'):
            self.check()

    def test_reject_changed_inputs_boundaries_scaler_and_code(self):
        for kwargs in ({'hashes': {'other': 'hash'}}, {'splits': {'test': (11, 16)}},
                       {'scaler': GlobalStandardScaler(3., 1.)}):
            with self.assertRaises(ValueError):
                self.check(**kwargs)
        with patch('traffic_benchmark.evaluation.checkpoint.file_sha256', return_value='different'):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                validate_checkpoint(self.checkpoint, self.report, self.hashes, self.splits, self.scaler)

    def test_reject_checkpoint_report_identity_mismatch(self):
        self.checkpoint['identity']['source_sha256']['flowsd.npz'] = 'other'
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            self.check()

    def test_restoration_freezes_weights_and_scores_all_windows_without_changes(self):
        maps = build_spatial_maps(np.array([[0., 0.], [1., 1.], [2., 2.]]), np.eye(3) + 1,
                                  depth=1, leaf_size=2, merge_factor=1)
        cfg = PatchSTGConfig(nodes=3, history=2, signal_dim=2, node_dim=2,
                            slot_dim=2, weekday_dim=2, layers=1, dropout=.5)
        original = PatchSTG(cfg, maps)
        checkpoint = {'identity': {'model_config': asdict(cfg)}, 'model': original.state_dict()}
        model = restore_frozen_model(checkpoint, maps)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        values = np.arange(48, dtype=float).reshape(16, 3, 1)
        data = TrafficWindows(values, (12, 16), self.scaler, history=2, horizon=2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = evaluate(model, DataLoader(data, batch_size=2), self.scaler, torch.device('cpu'), 2)
        metrics = MaskedForecastMetrics(2)
        with torch.no_grad():
            for i in range(len(data)):
                sample = data[i]
                pred = self.scaler.inverse_transform(model(torch.from_numpy(sample['x'])[None],
                                                          torch.from_numpy(sample['x_calendar'])[None]))
                metrics.update(pred.numpy(), sample['y'][None])
        self.assertEqual(result['windows'], 3)  # Includes final partial batch.
        self.assertAlmostEqual(result['overall_pooled']['mae'], metrics.compute()['overall_pooled']['mae'], places=5)
        for k, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[k]))
        checkpoint['model']['gather_indices'] = torch.flip(checkpoint['model']['gather_indices'], [0])
        with self.assertRaisesRegex(ValueError, 'spatial buffer mismatch'):
            restore_frozen_model(checkpoint, maps)


if __name__ == '__main__':
    unittest.main()
