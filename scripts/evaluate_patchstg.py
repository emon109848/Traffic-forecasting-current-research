"""Evaluate a frozen validation-selected PatchSTG checkpoint on the full SD test split."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows, load_processed_sd, make_splits
from traffic_benchmark.data.spatial import load_spatial_maps
from traffic_benchmark.evaluation.checkpoint import file_sha256, validate_checkpoint, restore_frozen_model
from traffic_benchmark.training.trainer import evaluate, seed_everything


def git_output(*args):
    try:
        return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw/largest_sd'))
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path. Do not overwrite test results.')
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; no automatic CPU fallback.')
    report = json.loads((args.run_dir / 'report.json').read_text())
    checkpoint_path = args.run_dir / 'best.pt'
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    values, _ = load_processed_sd(args.data_dir)
    splits = make_splits(len(values))
    scaler = GlobalStandardScaler.fit(values, splits['train'][1])
    hashes = {name: file_sha256(args.data_dir / name) for name in ('flowsd.npz', 'sd_meta.csv')}
    identity = validate_checkpoint(checkpoint, report, hashes, splits, scaler)
    config = identity['model_config']
    if config['nodes'] != 716 or config['history'] != 12 or config['slots_per_day'] != 96:
        raise ValueError('Expected fixed SD 716-node, 12-to-12, 15-minute protocol.')
    seed = report['config']['seed']
    torch.set_num_threads(2)
    seed_everything(seed)
    maps = load_spatial_maps(args.run_dir / 'spatial_maps.npz', identity['spatial'])
    model = restore_frozen_model(checkpoint, maps).to(args.device)
    dataset = TrafficWindows(values, splits['test'], scaler)
    batch_size = report['config']['batch_size']
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if args.device == 'cuda:0':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    result = evaluate(model, loader, scaler, torch.device(args.device), 12)
    if result['windows'] != len(dataset):
        raise ValueError('Incomplete test evaluation.')
    result['per_horizon'] = [{**row, 'minutes': row['step'] * 15} for row in result['per_horizon']]
    output = {
        'status': 'completed', 'model': 'PatchSTG',
        'implementation': 'code-aligned shared-protocol adaptation; not an official reproduction',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': {'path': str(checkpoint_path), 'sha256': checkpoint_hash,
                       'epoch': checkpoint['epoch'], 'validation_mae': checkpoint['validation_mae'],
                       'selection': 'lowest full validation pooled MAE; first epoch wins ties'},
        'training_report_sha256': file_sha256(args.run_dir / 'report.json'),
        'spatial_maps_sha256': file_sha256(args.run_dir / 'spatial_maps.npz'),
        'training_identity': identity, 'seed': seed, 'scaler': asdict(scaler),
        'config': {'split': 'test', 'splits': splits, 'history': 12, 'horizon': 12,
                   'stride': 1, 'batch_size': batch_size, 'shuffle': False, 'num_workers': 0,
                   'device': args.device, 'precision': identity['precision'],
                   'target_mask': 'nonzero targets only; no clipping; MAPE percent without epsilon',
                   'prediction_units': 'original flow units', 'test_used_for_tuning': False},
        'source_hashes': hashes,
        'software': {'python': platform.python_version(), 'torch': str(torch.__version__),
                     'numpy': np.__version__, 'cuda_build': torch.version.cuda,
                     'cudnn': torch.backends.cudnn.version()},
        'hardware': {'platform': platform.platform(), 'processor': platform.processor()},
        'git': {'revision': git_output('rev-parse', 'HEAD'), 'status': git_output('status', '--porcelain')},
        'implementation_hashes': {p: file_sha256(p) for p in (
            'scripts/evaluate_patchstg.py', 'src/traffic_benchmark/evaluation/checkpoint.py')},
        'parameters': sum(p.numel() for p in model.parameters()),
        'results': {'test': result},
        'timing_scope': 'Single full test pass: loading batches, transfers, forward, inverse scaling, metrics; '
                        'excludes setup; no warmup; not an isolated inference-speed benchmark',
    }
    if args.device == 'cuda:0':
        props = torch.cuda.get_device_properties(0)
        output['hardware']['gpu'] = {'name': props.name, 'total_memory_bytes': props.total_memory,
                                    'compute_capability': list(torch.cuda.get_device_capability(0))}
        output['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        output['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    # Verify originals remained unchanged throughout inference.
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError('Checkpoint changed during evaluation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Test:', json.dumps(result['overall_pooled']))
    print('Saved', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
